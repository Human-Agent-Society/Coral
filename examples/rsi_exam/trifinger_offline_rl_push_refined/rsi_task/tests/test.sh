#!/usr/bin/env bash
# Harbor verifier entry. Runs AFTER the agent finishes. Computes the reward via the two-phase
# grade.py (de-privileged child runs the submitted Policy on sealed episodes -> action traces;
# trusted root parent replays + scores), which writes /logs/verifier/reward.txt (and reward.json).
# grade.py owns process isolation; this file is the outermost fail-closed gate.
#
# NOTE the missing `-e`: we must survive a non-zero grade.py to run the gate below, so the exit code
# is captured explicitly instead (and `|| true`, which used to swallow it, is gone).
set -uo pipefail

export PYTHONDONTWRITEBYTECODE=1   # no __pycache__ copies of sealed modules left readable in /tests

mkdir -p /logs/verifier
chmod 0755 /logs/verifier          # normalise the bind mount: root writes, uid 4242 does not

# pre-clean. A reward file EXISTING can never be evidence that a reward was EARNED —
# the untrusted phase could have seeded it before killing the grader.
rm -f /logs/verifier/reward.txt /logs/verifier/reward.json /logs/verifier/grade_debug.json \
      /logs/verifier/score_details.json

# The submission is mounted from the agent side and may carry a restrictive mode; the de-privileged
# child must still be able to READ it. Read bits only — never write, never chown.
chmod -R a+rX /app/methods /app/submission 2>/dev/null || true

python3 /tests/grade.py
RC=$?

# Fail-closed gate: a reward counts only if grade.py exited cleanly AND left both reward files.
# This is what catches an uncatchable SIGKILL of the grader, which no in-process handler can.
# reward is still forced to 0.0, but a `grader failed:` marker from the previous file is
# inherited (and synthesized if absent), so a broken grader is not silently recorded as
# "the agent scored 0".
if [ "$RC" -ne 0 ] || [ ! -f /logs/verifier/reward.txt ] || [ ! -f /logs/verifier/reward.json ]; then
  python3 - "$RC" <<'PYEOF'
import json, math, os, sys
p = "/logs/verifier/reward.json"
# reward.json may contain FINITE NUMBERS ONLY: harbor's VerifierResult is a pydantic
# dict[str, float|int], so a string value raises ValidationError and the whole run is
# dropped rather than scored. The marker text goes to score_details.json; only the numeric
# flag grader_failed=1 stays here. Both "error" (str) and "errors" (list) shapes are checked.
mark = ""
# The marker is searched for across all three files, not just reward.json: that one holds
# finite numbers only (text is stripped by _reward_json_numeric), so a reward.json-only
# search never hits and degrades every failure to a generic synthesized string, discarding
# the real cause grade.py already diagnosed. The text lives in grade_debug.json /
# score_details.json.
for _src in ("/logs/verifier/grade_debug.json", "/logs/verifier/score_details.json", p):
    try:
        _prev = json.load(open(_src))
    except Exception:
        continue
    if not isinstance(_prev, dict):
        continue
    _cands = []
    _e = _prev.get("error")
    if isinstance(_e, str):
        _cands.append(_e)
    _es = _prev.get("errors")
    if isinstance(_es, list):
        _cands.extend(x for x in _es if isinstance(x, str))
    elif isinstance(_es, str):
        _cands.append(_es)
    for _c in _cands:
        if _c.strip().lower().startswith("grader failed:"):
            mark = _c
            break
    if mark:
        break
if not mark:
    mark = "grader failed: grade.py exited %s without a usable reward.json" % sys.argv[1]

# reward is always forced to 0.0: a reward.json seeded by the submission must not survive
# a grader crash.
d = {'reward': 0.0, 'mean_return': 0.0}
d = dict((k, int(v) if isinstance(v, bool) else v) for k, v in d.items()
         if isinstance(v, (int, float)) and not isinstance(v, bool)
         or isinstance(v, bool))
d = dict((k, v) for k, v in d.items() if isinstance(v, (int, float)) and math.isfinite(v))
d["reward"] = 0.0
d["grader_failed"] = 1
with open(p, "w") as fh:
    json.dump(d, fh)

# The human-readable text goes to score_details.json, which the results archiver reads
# straight from the trial directory.
try:
    _sd = os.path.join(os.path.dirname(p), "score_details.json")
    _doc = {}
    if os.path.exists(_sd):
        try:
            _doc = json.load(open(_sd))
        except Exception:
            _doc = {}
    if not isinstance(_doc, dict):
        _doc = {}
    _doc["error"] = mark
    with open(_sd, "w") as fh:
        json.dump(_doc, fh)
except Exception:
    pass
PYEOF
  echo 0 > /logs/verifier/reward.txt
fi
cat /logs/verifier/reward.txt
