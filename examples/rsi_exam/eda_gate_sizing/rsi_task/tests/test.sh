#!/usr/bin/env bash
# Harbor verifier entry. Copied to /tests/test.sh and run AFTER the agent finishes.
# grade.py owns the two-phase parent/child isolation and writes the reward.
set -uo pipefail

mkdir -p /logs/verifier

# the submission shares this filesystem, so it can seed a reward file
# during solve(). Harbor reads reward.json in preference to reward.txt and never
# looks at exit codes — so BOTH files must be removed before grading, not just
# the txt one.
rm -f /logs/verifier/reward.json /logs/verifier/reward.txt

python3 /tests/grade.py
RC=$?

# Fail-closed, but never clobber the infra-failure marker. Reward
# is forced to 0.0 no matter what is on disk (so a seeded 1.0 cannot survive a
# grader crash), while an existing "grader failed: ..." error string is carried
# over so the run gets classified as infra_failure instead of "the agent scored 0".
if [ ! -s /logs/verifier/reward.json ] || [ "$RC" -ne 0 ]; then
  ERR=$(python3 - <<'PY'
import json, pathlib
p = pathlib.Path("/logs/verifier/reward.json")
msg = ""
try:
    msg = json.loads(p.read_text()).get("error", "")
except Exception:
    pass
if not msg.startswith("grader failed: "):
    msg = "grader failed: grade.py exited without a usable reward file"
print(msg)
PY
)
  python3 - "$ERR" <<'PY'
import json, pathlib, sys
pathlib.Path("/logs/verifier/reward.json").write_text(
    json.dumps({"reward": 0.0, "mean_score": 0.0, "error": sys.argv[1]}))
pathlib.Path("/logs/verifier/reward.txt").write_text("0.0\n")
PY
fi

cat /logs/verifier/reward.json
