#!/usr/bin/env bash
set -uo pipefail

mkdir -p /logs/verifier
chmod 0700 /tests
python3 /tests/grade.py || true
if [ ! -f /logs/verifier/reward.txt ]; then
  echo 0 > /logs/verifier/reward.txt
fi
cat /logs/verifier/reward.txt
