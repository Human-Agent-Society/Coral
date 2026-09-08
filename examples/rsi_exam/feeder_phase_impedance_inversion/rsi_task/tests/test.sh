#!/bin/bash
# Hardened verifier entrypoint (second layer of defence; grade.py and the uid-4242 privilege
# drop are the first). Deliberately NOT `set -e` -- we must run past a non-zero grade.py so
# the exit-code gate below can still execute.
set -uo pipefail

# The log/reward directory is hardcoded and never read from the host env: `-e VERIFIER_LOG_DIR=...`
# would otherwise send reward.json somewhere the harness cannot read it (a silent fail-open).
# Keep in sync with grade.py's LOG_DIR.
LOGDIR="/logs/verifier"
mkdir -p "$LOGDIR"
# Normalise the bind-mount perms: root writes reward files, the dropped solver uid (4242)
# must NOT be able to (a 0777 host mount would otherwise let the child seed reward.json).
chmod 0755 "$LOGDIR" 2>/dev/null || true

# Fail-closed: a reward file that merely EXISTS is never evidence of a score. Wipe any value
# the submitted method may have seeded before grading starts. grade_debug.json is wiped too --
# the fallback below inherits the `grader failed:` marker from it, and a stale copy would
# mislabel this run as an infra failure.
rm -f "$LOGDIR/reward.txt" "$LOGDIR/reward.json" "$LOGDIR/score_details.json" \
      "$LOGDIR/grade_debug.json"

# The submission is mounted from the agent side and may carry restrictive perms; let uid 4242
# read it (read-only, never chown, never +w -- adding write would reopen a hole-B variant).
chmod -R a+rX /app/methods 2>/dev/null || true

python3 /tests/grade.py
RC=$?

# Exit-code gate: only a clean exit that produced reward.json counts. Anything else -> 0.
# This backstops the case where grade.py is SIGKILLed before it can write: even if the
# drop-priv layer somehow let the child kill the parent, disk still ends up 0.
# reward is still forced to 0.0, but any `grader failed:` marker is inherited, so a broken
# grader is not silently recorded as "the agent scored 0".
if [ "$RC" -ne 0 ] || [ ! -f "$LOGDIR/reward.json" ]; then
  # The heredoc below is <<'PYEOF' (quoted = no expansion), so "$LOGDIR" would be a LITERAL
  # string here. The path must stay a literal absolute path, or the fallback silently writes
  # a file named `$LOGDIR` in the CWD and the real reward.json is never touched.
  python3 - "$RC" <<'PYEOF'
import json, math, os, sys
p = "/logs/verifier/reward.json"
# reward.json may contain FINITE NUMBERS ONLY: harbor's VerifierResult is a pydantic
# dict[str, float|int], so a string value raises ValidationError and the whole run is
# dropped instead of scored. The marker text goes to score_details.json; only the numeric
# flag grader_failed=1 stays here. Both "error" (str) and "errors" (list) shapes are checked.
mark = ""
try:
    _prev = json.load(open(p))
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
# Since reward.json holds numbers only, the prefixed TEXT lives in score_details.json /
# grade_debug.json. Inherit from those too, otherwise the specific failure reason grade.py
# already recorded is overwritten by the least informative synthesized string.
if not mark:
    for _f in ("score_details.json", "grade_debug.json"):
        try:
            _doc0 = json.load(open(os.path.join(os.path.dirname(p), _f)))
        except Exception:
            continue
        if not isinstance(_doc0, dict):
            continue
        _c2 = []
        _e0 = _doc0.get("error")
        if isinstance(_e0, str):
            _c2.append(_e0)
        _es0 = _doc0.get("errors")
        if isinstance(_es0, list):
            _c2.extend(x for x in _es0 if isinstance(x, str))
        for _c in _c2:
            if _c.strip().lower().startswith("grader failed:"):
                mark = _c
                break
        if mark:
            break
if not mark:
    mark = "grader failed: grade.py exited %s without a usable reward.json" % sys.argv[1]

# reward is always forced to 0.0: a reward.json seeded by the submission must not survive
# a grader crash.
d = {'reward': 0.0}
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
  echo 0 > "$LOGDIR/reward.txt"
fi

cat "$LOGDIR/reward.json"
