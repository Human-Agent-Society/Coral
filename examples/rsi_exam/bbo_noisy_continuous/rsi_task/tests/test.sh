#!/usr/bin/env bash
set -uo pipefail

REWARD_DIR=/logs/verifier
umask 077
mkdir -p "${REWARD_DIR}"
chmod 0700 "${REWARD_DIR}"
rm -f "${REWARD_DIR}/reward.txt" "${REWARD_DIR}/reward.json" "${REWARD_DIR}/score_details.json" "${REWARD_DIR}/grade_debug.json"

set +e
python3 /tests/grade.py
RC=$?
set -e

VALID_REWARD=0
if python3 - "${REWARD_DIR}" <<'PY_VALIDATE' 2>/dev/null
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])


def load_json(name):
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(
        (root / name).read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(float(value))


reward_doc = load_json("reward.json")
assert type(reward_doc) is dict and set(reward_doc) == {"reward"}
reward = reward_doc["reward"]
assert finite_number(reward) and 0.0 <= float(reward) <= 1.0
assert (root / "reward.txt").read_text(encoding="utf-8") == f"{float(reward)!r}\n"

details = load_json("score_details.json")
assert type(details) is dict
assert set(details) == {"metric", "direction", "aggregation", "instances", "aggregate"}
assert details["metric"] == "best_latent_objective_at_final_query"
assert details["direction"] == "lower"
assert details["aggregation"] == (
    "median over seeds per instance; leaderboard reward uses "
    "mean(0.70*anytime_per_instance+0.30*final_per_instance)"
)
assert type(details["instances"]) is list
instance_keys = {
    "id",
    "raw_metric",
    "floor",
    "upper_bound",
    "score",
    "anytime_score",
    "final_score",
}
numeric_instance_keys = instance_keys - {"id"}
for instance in details["instances"]:
    assert type(instance) is dict and set(instance) == instance_keys
    assert type(instance["id"]) is str and instance["id"]
    assert all(finite_number(instance[key]) for key in numeric_instance_keys)
    assert all(
        0.0 <= float(instance[key]) <= 1.0
        for key in ("score", "anytime_score", "final_score")
    )

aggregate = details["aggregate"]
aggregate_keys = {"raw_metric", "floor", "upper_bound", "reward"}
assert type(aggregate) is dict and set(aggregate) == aggregate_keys
assert all(finite_number(aggregate[key]) for key in aggregate_keys)
assert float(aggregate["reward"]) == float(reward)

debug = load_json("grade_debug.json")
assert type(debug) is dict
assert finite_number(debug.get("reward"))
assert float(debug["reward"]) == float(reward)
assert type(debug.get("correctness")) is bool
assert type(debug.get("errors")) is list
assert all(type(error) is str for error in debug["errors"])
if debug["correctness"]:
    assert debug.get("score_details") == details
else:
    assert float(reward) == 0.0 and details["instances"] == []
PY_VALIDATE
then
  VALID_REWARD=1
fi

if [ "${RC}" -ne 0 ] || [ "${VALID_REWARD}" -ne 1 ]; then
  echo "grader failed: verifier exited ${RC}; writing zero reward" >&2
  python3 - "${REWARD_DIR}" "${RC}" <<'PY_FALLBACK'
import json
import os
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1])
return_code = sys.argv[2]
marker = None
try:
    debug = json.loads((root / "grade_debug.json").read_text(encoding="utf-8"))
    if type(debug) is dict and type(debug.get("errors")) is list:
        for value in debug["errors"]:
            if type(value) is str and value.startswith("grader failed:"):
                marker = value[:1000]
                break
except Exception:
    pass
if marker is None:
    marker = f"grader failed: verifier entrypoint failed closed (exit {return_code})"

documents = {
    "reward.txt": "0.0\n",
    "reward.json": json.dumps({"reward": 0.0}, sort_keys=True) + "\n",
    "score_details.json": json.dumps(
        {
            "metric": "best_latent_objective_at_final_query",
            "direction": "lower",
            "aggregation": (
                "median over seeds per instance; leaderboard reward uses "
                "mean(0.70*anytime_per_instance+0.30*final_per_instance)"
            ),
            "instances": [],
            "aggregate": {
                "raw_metric": 0.0,
                "floor": 0.0,
                "upper_bound": 0.0,
                "reward": 0.0,
            },
        },
        sort_keys=True,
    )
    + "\n",
    "grade_debug.json": json.dumps(
        {
            "reward": 0.0,
            "correctness": False,
            "errors": [marker],
        },
        sort_keys=True,
    )
    + "\n",
}
for name in documents:
    try:
        (root / name).unlink()
    except FileNotFoundError:
        pass
for name, text in documents.items():
    descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=root)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, root / name)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
PY_FALLBACK
fi

chmod 0644 "${REWARD_DIR}/reward.txt" "${REWARD_DIR}/reward.json" \
  "${REWARD_DIR}/score_details.json" "${REWARD_DIR}/grade_debug.json"
cat "${REWARD_DIR}/reward.txt"
