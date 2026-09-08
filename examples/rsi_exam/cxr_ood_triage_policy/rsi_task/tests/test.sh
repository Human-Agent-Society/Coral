#!/usr/bin/env bash
# Harbor verifier entry. Copied to /tests/test.sh and run with bash AFTER the agent finishes.
# Computes the reward via the TWO-PHASE grade.py, which writes /logs/verifier/reward.txt
# (and reward.json). grade.py owns the parent/child process isolation; Harbor only runs this
# script and reads the reward.
#
# NOTE: deliberately `set -uo pipefail` and NOT `-e`. We must survive a non-zero grade.py and
# run the gate below ourselves; `-e` would abort the script before the gate ever executed.
set -uo pipefail

mkdir -p /logs/verifier
# Normalise the bind-mount's mode: root (the trusted parent) writes here, the uid-4242 child
# must not. Do not tighten to 0700 -- the host side may read the reward as a non-root user.
chmod 0755 /logs/verifier 2>/dev/null || true

# The submission tree comes from the agent side and may carry a restrictive mode; the dropped
# child still has to READ it. Add read/traverse only -- never write, never chown (making
# /app/methods writable by the child would reopen the overwrite-the-scorer hole in a new place).
chmod -R a+rX /app/methods 2>/dev/null || true

# Clear the field BEFORE grading. A reward file that merely EXISTS is never evidence of a
# score: an untrusted child could have planted one and then killed the grader.
# score_details.json is cleared too: grade.py writes it as the marker's text carrier, and a
# leftover copy from an earlier run would be read as this run's evidence by the fallback below.
rm -f /logs/verifier/reward.txt /logs/verifier/reward.json /logs/verifier/grade_debug.json \
      /logs/verifier/score_details.json

python3 /tests/grade.py
RC=$?

# Fail-closed gate: only a clean exit that actually produced a reward counts. Anything else
# (grade.py SIGKILLed by the submission, crashed, wrote nothing) collapses to 0.
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
# The marker cannot be inherited from reward.json alone: it holds only numbers (the flag
# grader_failed=1), while the prefixed TEXT lands in grade_debug.json / score_details.json.
# Searching reward.json only would never hit, degrading every failure to a generic
# synthesized marker that loses what actually broke.
mark = ""
_d = os.path.dirname(p)
for _src in (p, os.path.join(_d, "grade_debug.json"), os.path.join(_d, "score_details.json")):
    if mark:
        break
    try:
        _prev = json.load(open(_src))
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
    except Exception:
        pass
if not mark:
    mark = "grader failed: grade.py exited %s without a usable reward.json" % sys.argv[1]

# reward is always forced to 0.0: a reward.json seeded by the submission must not survive
# a grader crash.
d = {'reward': 0.0, 'triage_utility_pct': 0.0}
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
