#!/usr/bin/env python3
"""Re-run the submitted harness on sealed seeds and compute per-instance reward."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from runner import evaluate

HIDDEN_CASES = [
    {
        "id": "sealed-space-3",
        "scenario": "Space Sick",
        "difficulty": "Challenge",
        "seed": 3,
    },
    {
        "id": "sealed-space-4",
        "scenario": "Space Sick",
        "difficulty": "Challenge",
        "seed": 4,
    },
    {
        "id": "sealed-chemistry-3",
        "scenario": "Combinatorial Chemistry",
        "difficulty": "Challenge",
        "seed": 3,
    },
    {
        "id": "sealed-chemistry-4",
        "scenario": "Combinatorial Chemistry",
        "difficulty": "Challenge",
        "seed": 4,
    },
]

LOG_DIR = Path("/logs/verifier")
SUBMISSION = Path("/app/methods/main/agent.py")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "120"))

ANCHOR_FILE = Path(os.environ.get("ANCHORS_PATH", "/tests/anchors.json"))


def _load_anchors() -> dict[str, tuple[float, float, float]]:
    themes = json.loads(ANCHOR_FILE.read_text())["themes"]
    out = {}
    for name, v in themes.items():
        b, s, u = float(v["baseline"]), float(v["sota"]), float(v["upper"])
        if not b < s < u:
            raise ValueError(f"anchors out of order for {name}: {v}")
        out[name] = (b, s, u)
    return out


ANCHORS = _load_anchors()


def reward_of(metric: float, baseline: float, sota: float, upper: float) -> float:
    """baseline->0, sota->0.6, upper->1; piecewise linear, clamped."""
    if sota <= baseline or upper <= sota:
        return 0.0
    if metric <= baseline:
        return 0.0
    if metric < sota:
        return 0.6 * (metric - baseline) / (sota - baseline)
    if metric < upper:
        return 0.6 + 0.4 * (metric - sota) / (upper - sota)
    return 1.0


def fail(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "reward.json").write_text(
        json.dumps({"reward": 0.0, "error": message})
    )
    (LOG_DIR / "reward.txt").write_text("0\n")
    print(f"FAIL: {message}")
    raise SystemExit(0)


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not SUBMISSION.exists():
        fail(f"missing harness entry: {SUBMISSION}")
    if not os.environ.get("DISCOVERYWORLD_LLM_API_KEY"):
        fail("DISCOVERYWORLD_LLM_API_KEY is missing")

    started = time.time()
    report = evaluate(
        HIDDEN_CASES,
        harness_path=str(SUBMISSION),
        max_steps=MAX_STEPS,
        log_root=str(LOG_DIR / "instance_logs"),
    )
    details = []
    for item in report["instances"]:
        baseline, sota, upper = ANCHORS[item["scenario"]]
        reward = reward_of(item["raw_metric"], baseline, sota, upper)
        details.append(
            {
                **item,
                "anchors": {
                    "baseline": baseline,
                    "sota": sota,
                    "upper": upper,
                },
                "reward": reward,
            }
        )
    reward = sum(item["reward"] for item in details) / len(details)
    output = {
        "metric": "mean_scientific_progress",
        "reward": reward,
        "mean_raw_metric": report["mean_raw_metric"],
        "mean_procedure": report["mean_procedure"],
        "mean_completion": report["mean_completion"],
        "mean_knowledge": report["mean_knowledge"],
        "max_steps": MAX_STEPS,
        "wall_sec": time.time() - started,
        "instances": details,
    }
    (LOG_DIR / "score_details.json").write_text(json.dumps(output, indent=2))
    (LOG_DIR / "reward.json").write_text(
        json.dumps(
            {
                "reward": reward,
                "mean_raw_metric": report["mean_raw_metric"],
                "mean_procedure": report["mean_procedure"],
                "mean_completion": report["mean_completion"],
                "mean_knowledge": report["mean_knowledge"],
            },
            indent=2,
        )
    )
    (LOG_DIR / "reward.txt").write_text(f"{reward}\n")
    print(
        f"hidden mean_raw_metric={report['mean_raw_metric']:.4f} "
        f"reward={reward:.4f} "
        f"(procedure={report['mean_procedure']:.4f}, "
        f"completion={report['mean_completion']:.4f}, "
        f"knowledge={report['mean_knowledge']:.4f})"
    )


if __name__ == "__main__":
    main()
