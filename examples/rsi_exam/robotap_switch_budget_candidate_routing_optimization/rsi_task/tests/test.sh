#!/usr/bin/env bash
set -euo pipefail
umask 077

source /tests/reward_log_owner.sh

mkdir -p /logs/verifier /run/candidate-routing-grade
capture_reward_log_owner /logs/verifier
trap 'restore_reward_log_owner /logs/verifier' EXIT

chmod 0700 /logs/verifier
chmod 0711 /run/candidate-routing-grade
rm -f /logs/verifier/reward.json /logs/verifier/reward.txt /logs/verifier/grade_debug.json /logs/verifier/score_details.json

set +e
python /tests/grade.py
grade_rc="$?"
set -e

outputs_valid=0
if python - <<'PY'
import json
import math
from pathlib import Path

payload = json.loads(Path('/logs/verifier/reward.json').read_text())
if set(payload) != {'reward'}:
    raise SystemExit(1)
raw_value = payload['reward']
if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
    raise SystemExit(1)
value = float(raw_value)
text_value = float(Path('/logs/verifier/reward.txt').read_text().strip())
if not (
    math.isfinite(value)
    and math.isfinite(text_value)
    and 0.0 <= value <= 1.0
    and abs(value - text_value) <= 5e-9
):
    raise SystemExit(1)
PY
then
    outputs_valid=1
fi

if [[ "$grade_rc" -ne 0 || "$outputs_valid" -ne 1 ]]; then
    # This trusted fallback always destroys a planted/high reward. It preserves
    # an existing diagnostic only when grade.py already classified the failure
    # as verifier infrastructure ("grader failed:").
    python /tests/fail_closed.py
fi

chmod 0644 /logs/verifier/reward.json /logs/verifier/reward.txt
for artifact in grade_debug.json score_details.json; do
    if [[ -f "/logs/verifier/$artifact" ]]; then
        chmod 0644 "/logs/verifier/$artifact"
    fi
done
cat /logs/verifier/reward.txt
