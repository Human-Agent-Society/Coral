#!/usr/bin/env bash
# Harbor verifier entry for finscope_dcf_valuation. Copied to /tests/test.sh and run with bash
# AFTER the agent finishes. grade.py writes /logs/verifier/reward.txt (and reward.json).
set -uo pipefail

mkdir -p /logs/verifier
# Normalise the bind-mounted reward dir: root writes here, the de-privileged solve subprocess
# (uid 4242, see grade.py / tests/Dockerfile) must not be able to. Do not tighten past 0755 —
# the host side may read these files as a non-root user.
chmod 0755 /logs/verifier 2>/dev/null || true

# The submission arrives from the agent side with unknown ownership/mode and uid 4242 has to be
# able to READ it. Add read/traverse only — never write, never chown (that re-opens the write hole).
chmod -R a+rX /app/methods 2>/dev/null || true

# clear the slate BEFORE grading. An existing reward file is never evidence of a
# score — the submission ran earlier and may have planted one (measured: it could, and could
# then SIGKILL the grader so nothing ever overwrote it).
# score_details.json is cleared as well: grade.py writes it, and the archiver reads the
# `grader failed:` text from it, so a copy planted by the submission must not survive.
rm -f /logs/verifier/reward.txt /logs/verifier/reward.json \
      /logs/verifier/grade_debug.json /logs/verifier/score_details.json

python3 /tests/grade.py   # NOTE: no `|| true`, and no `set -e`, so we can read $? below
RC=$?

# Fail-closed on BOTH signals: the grader must have exited cleanly AND produced both reward
# files. Anything else (crash, SIGKILL, partial write) scores 0.
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
# Since reward.json holds numbers only, the marker text can never be in it. The search
# therefore covers score_details.json / grade_debug.json first, otherwise the fallback
# always synthesizes a generic string with no actual failure reason.
mark = ""
_dir = os.path.dirname(p)
for _src in ("score_details.json", "grade_debug.json", p):
    if mark:
        break
    try:
        _prev = json.load(open(_src if os.path.isabs(_src) else os.path.join(_dir, _src)))
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
if not mark:
    mark = "grader failed: grade.py exited %s without a usable reward.json" % sys.argv[1]

# reward is always forced to 0.0: a reward.json seeded by the submission must not survive
# a grader crash.
d = {'reward': 0.0, 'valuation_mave': 0.0}
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
