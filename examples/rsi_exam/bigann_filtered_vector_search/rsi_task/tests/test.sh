#!/usr/bin/env bash
# Harbor verifier entry. Runs the TWO-PHASE grade.py, which writes /logs/verifier/reward.json
# (+ reward.txt). grade.py owns parent/child process isolation; Harbor only runs this + reads reward.
set -uo pipefail

mkdir -p /logs/verifier
python3 /tests/grade.py || true   # grade.py writes reward 0 on its own failures

if [ ! -f /logs/verifier/reward.txt ]; then
  echo 0 > /logs/verifier/reward.txt
fi
cat /logs/verifier/reward.txt
