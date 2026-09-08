#!/usr/bin/env bash
# Verifier entrypoint.  Two jobs beyond calling the grader:
#   1. clear the reward sinks BEFORE grading, so a reward.json / reward.txt /
#      score_details.json planted by the agent phase can never survive into scoring;
#   2. never exit without a reward: `set -e` on python3 used to leave the directory empty
#      when the grader died, and an empty directory is not a 0.
set -uo pipefail
mkdir -p /logs/verifier
rm -f /logs/verifier/reward.json /logs/verifier/reward.txt /logs/verifier/score_details.json

python3 /tests/grade.py
rc="$?"

if [[ "$rc" -ne 0 || ! -f /logs/verifier/reward.json || ! -f /logs/verifier/reward.txt ]]; then
    printf '%s\n' "{\"reward\": 0.0, \"error\": \"grader failed: rc=${rc}\"}" \
        > /logs/verifier/reward.json
    printf '0.0000\n' > /logs/verifier/reward.txt
fi

exit "$rc"
