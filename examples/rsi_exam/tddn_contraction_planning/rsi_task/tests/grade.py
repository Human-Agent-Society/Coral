"""Harbor verifier for sealed TDDN contraction planning."""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

from evaluate import evaluate_suite, score_suites


SUBMISSION = Path("/app/methods/main")
HELDOUT = Path("/tests/heldout/cases.json")
LOG_DIR = Path(os.environ.get("VERIFIER_LOG_DIR", "/logs/verifier"))
BASELINE = Path("/tests/trusted/baseline/solver.py")
REFERENCE = Path("/tests/trusted/reference/solver.py")

# grader-side failure marker: lets the aggregation layer tell "the grader broke"
# apart from "the submission scored 0"
GRADER_FAILED = "grader failed: "


def score_details(report: dict[str, object]) -> dict[str, object]:
    """Per-case raw metrics plus the anchors they were scored against."""
    def rows(suite: object) -> list[dict[str, object]]:
        if not isinstance(suite, dict):
            return []
        return list(suite.get("cases", []))

    anchors = {
        key: report.get(key)
        for key in (
            "reference_combined_log_gain",
            "reference_peak_speedup_geomean",
            "reference_time_speedup_geomean",
        )
    }
    return {
        "reward": report.get("reward", 0.0),
        "correct": report.get("correct", False),
        "combined_log_gain": report.get("combined_log_gain"),
        "anchors": anchors,
        "per_case": {
            "candidate": rows(report.get("candidate")),
            "baseline": rows(report.get("baseline")),
            "reference": rows(report.get("reference")),
        },
        "error": report.get("error", ""),
    }


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, object]
    try:
        cases = json.loads(HELDOUT.read_text(encoding="utf-8"))
        baseline = evaluate_suite(BASELINE, cases, unprivileged=False)
        reference = evaluate_suite(REFERENCE, cases, unprivileged=False)
        candidate = evaluate_suite(
            SUBMISSION / "solver.py",
            cases,
            unprivileged=True,
        )
        score = score_suites(candidate, baseline, reference)
        report = {
            **score,
            "candidate": candidate,
            "baseline": baseline,
            "reference": reference,
        }
    except Exception as error:
        report = {
            "reward": 0.0,
            "correct": False,
            "error": f"{GRADER_FAILED}{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }

    reward = float(report.get("reward", 0.0))
    (LOG_DIR / "reward.txt").write_text(f"{reward}\n", encoding="utf-8")
    # reward.json carries finite numbers only -- harbor drops the whole trial otherwise
    (LOG_DIR / "reward.json").write_text(
        json.dumps({
            "reward": reward,
            "peak_tdd_nodes_speedup_geomean": float(report.get(
                "peak_tdd_nodes_speedup_geomean", 0.0
            ) or 0.0),
            "total_time_speedup_geomean": float(report.get(
                "total_time_speedup_geomean", 0.0
            ) or 0.0),
        }),
        encoding="utf-8",
    )
    (LOG_DIR / "score_details.json").write_text(
        json.dumps(score_details(report), indent=2),
        encoding="utf-8",
    )
    (LOG_DIR / "grade_debug.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        key: value
        for key, value in report.items()
        if key not in {"candidate", "baseline", "reference"}
    }))


if __name__ == "__main__":
    main()
