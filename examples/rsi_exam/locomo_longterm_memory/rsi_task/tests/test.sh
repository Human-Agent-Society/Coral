#!/usr/bin/env bash
# Harbor prefers reward.json over reward.txt and ignores the container exit
# code, so the fail-closed path has to live here. No `set -e`: the fallback
# below must still run when grade.py dies.
set -uo pipefail
cd "$(dirname "$0")"

OUT=/logs/verifier
mkdir -p "$OUT"
# Wipe before grading. The submitted code shares this filesystem and can seed a
# score for the case where grade.py never gets to overwrite it.
rm -f "$OUT/reward.json" "$OUT/reward.txt"

python3 grade.py
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
    # grader-failure marker so infra breakage stays distinguishable from a
    # submission that honestly scored 0.
    err = payload.get("error", "")
    if not (isinstance(err, str) and err.startswith("grader failed:")):
        err = f"grader failed: grade.py exited {rc} without a usable reward"
    payload = {"reward": 0.0, "error": err}
    json.dump(payload, open(path, "w"), indent=2)

open(os.path.join(out, "reward.txt"), "w").write(f"{payload['reward']}\n")
print(json.dumps(payload))
PY
