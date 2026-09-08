#!/usr/bin/env bash
# Harbor verifier entry. Copied to /tests/test.sh and run with bash AFTER the agent finishes.
# Computes the reward via grade.py, which writes /logs/verifier/reward.txt (and reward.json).
set -uo pipefail

mkdir -p /logs/verifier
python3 /tests/grade.py || true   # grade.py writes reward 0 on its own failures

# Fail-closed: if grade.py died before writing a reward, emit 0 so Harbor always sees one.
if [ ! -f /logs/verifier/reward.txt ]; then
  echo 0 > /logs/verifier/reward.txt
fi
cat /logs/verifier/reward.txt
