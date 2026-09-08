#!/usr/bin/env bash
set -uo pipefail

mkdir -p /logs/verifier

# Clear any reward file the submission side may have planted, before grading. harbor's
# verifier reads reward.json in preference to reward.txt, and /logs/verifier is a host
# directory shared between the agent and verifier containers -- so if the grader fails for any
# reason (OOM, timeout, bad data; no attack required) a pre-planted file would be taken as
# this run's score.
rm -f /logs/verifier/reward.json /logs/verifier/reward.txt \
      /logs/verifier/score_details.json /logs/verifier/grade_debug.json

python3 /tests/grade.py
RC=$?

# The fallback keys on grade.py's exit code, not on whether a file exists (the submission can
# create files itself). reward is always zeroed; only a `grader failed:` marker is inherited,
# and synthesized when absent, so a grader fault stays distinguishable from a genuine zero
# without handing the submission any scoring advantage.
if [ "$RC" -ne 0 ] || [ ! -s /logs/verifier/reward.json ]; then
  python3 - <<'PY'
import json
from pathlib import Path

log = Path("/logs/verifier")
marker = ""
for name in ("score_details.json", "grade_debug.json"):
    try:
        err = json.loads((log / name).read_text()).get("error", "")
    except Exception:
        continue
    if isinstance(err, str) and err.startswith("grader failed: "):
        marker = err
        break
marker = marker or "grader failed: grade.py produced no reward"

log.mkdir(parents=True, exist_ok=True)
(log / "reward.txt").write_text("0.0\n")
(log / "reward.json").write_text(json.dumps({"reward": 0.0}))
(log / "score_details.json").write_text(
    json.dumps({"reward": 0.0, "correct": False, "error": marker}, indent=2)
)
PY
fi

cat /logs/verifier/reward.txt
