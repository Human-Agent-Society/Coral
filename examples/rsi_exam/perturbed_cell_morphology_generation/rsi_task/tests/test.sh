#!/usr/bin/env bash
set -u
mkdir -p /logs/verifier
rm -f /logs/verifier/reward.json /logs/verifier/reward.txt \
  /logs/verifier/score_details.json /logs/verifier/grade_debug.json
python /tests/grade.py
status=$?
if [ "$status" -ne 0 ]; then
  printf '{"reward":0.0,"fid":0.0,"overall_fid":0.0,"grader_failed":1.0}\n' \
    > /logs/verifier/reward.json
  if [ ! -f /logs/verifier/grade_debug.json ]; then
    printf '{"error":"grader failed: test.sh observed grade.py exit %s"}\n' "$status" \
      > /logs/verifier/grade_debug.json
  fi
fi
exit 0
