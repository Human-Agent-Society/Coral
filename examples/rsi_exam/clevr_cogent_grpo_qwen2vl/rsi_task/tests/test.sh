#!/usr/bin/env bash
# Harbor verifier entry. Runs AFTER the agent finishes: grade.py loads the submitted
# Qwen2-VL model and scores it on the sealed CoGenT-B / SuperCLEVR sets.
#
# Harbor prefers reward.json over reward.txt and ignores the container exit code, so
# the fail-closed path lives here. No `set -e`: the fallback must still run if grade.py dies.
set -uo pipefail

OUT=/logs/verifier
mkdir -p "$OUT"
# Wipe before grading: the submitted artifact shares this filesystem and could seed a
# score for the case where grade.py never gets to overwrite it.
rm -f "$OUT/reward.json" "$OUT/reward.txt"

python3 /tests/grade.py
RC=$?

RC="$RC" python3 - "$OUT" <<'PY'
import json, os, sys

out = sys.argv[1]
path = os.path.join(out, "reward.json")
rc = int(os.environ["RC"])
try:
    payload = json.load(open(path))
    ok = rc == 0 and isinstance(payload.get("reward"), (int, float))
except Exception:
    payload, ok = {}, False

if not ok:
    # Force 0 -- never keep a reward the grader did not compute. Inherit only a
    # grader-failure marker so infra breakage stays distinguishable from a bad model.
    err = payload.get("error", "")
    if not (isinstance(err, str) and err.startswith("grader failed:")):
        err = f"grader failed: grade.py exited {rc} without a usable reward"
    payload = {"reward": 0.0, "error": err}
    json.dump(payload, open(path, "w"))

open(os.path.join(out, "reward.txt"), "w").write(f"{payload['reward']}\n")
print(json.dumps(payload))
PY
