"""Harbor verifier for clevr_cogent_grpo_qwen2vl.

Runs AFTER the agent finishes. Harbor uploads the agent's /app/submission (the trained, merged
Qwen2-VL-2B model) into this box; the sealed held-out sets travel with tests/ and were never
visible to the agent. We load the submitted model in a clean, network-isolated container and run
the FIXED eval protocol (eval_common.py) on both hidden test sets, then map the mean accuracy to
the anchored reward and write /logs/verifier/reward.{txt,json}.

Anti-cheat: the artifact here is model WEIGHTS, not code — nothing the agent wrote executes in
this box. The verifier owns the eval and the held-out data (separate-verifier, layer 1).

Reward (anchored, piecewise-linear, clamped):
    baseline   -> 0.0
    upper_bound-> 1.0
A missing, unloadable or wrong-architecture submission scores 0 -- that is the agent's
result, NOT a grader fault, and the two must stay distinguishable.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Hardcoded: a host that can set these with -e can move the score without leaving
# a trace in the repo. Author-side re-measurement edits a copy of this tree.
SUBMISSION_DIR = Path("/app/submission")
HELDOUT = Path("/tests/heldout")
REWARD_DIR = Path("/logs/verifier")
EVAL_BATCH = 8
ANCHORS = Path("/tests/anchors.json")

# Filled by load_anchors() before grading. No env fallback: the old
# os.environ.get("BASELINE", "50.0") silently repriced the whole band whenever
# the variable failed to arrive, and 50.0 looks plausible enough to go unnoticed.
BASELINE = FRONTIER = UPPER_BOUND = None


def load_anchors() -> None:
    """Read the sealed band from tests/anchors.json (root-owned, 0400)."""
    global BASELINE, FRONTIER, UPPER_BOUND
    a = json.loads(ANCHORS.read_text())
    BASELINE, FRONTIER, UPPER_BOUND = float(a["baseline"]), float(a["frontier"]), float(a["upper_bound"])

TEST_SETS = {
    "cogent_b": HELDOUT / "cogent_b_val",
    "superclevr": HELDOUT / "superclevr_test",
}

sys.path.insert(0, "/tests")


def reward_from_metric(acc: float) -> float:
    """baseline -> 0, frontier -> 0.6, upper_bound -> 1 (piecewise linear, clamped)."""
    if acc <= BASELINE:
        return 0.0
    if acc <= FRONTIER:
        return 0.6 * (acc - BASELINE) / (FRONTIER - BASELINE)
    if acc <= UPPER_BOUND:
        return 0.6 + 0.4 * (acc - FRONTIER) / (UPPER_BOUND - FRONTIER)
    return 1.0


class SubmissionError(Exception):
    """The submission is missing/unusable: an honest 0, not a grader fault."""


EXPECTED_ARCH = "Qwen2VLForConditionalGeneration"


def check_submission() -> None:
    """Positively identify a bad submission so it is not excused as infra failure."""
    cfg = SUBMISSION_DIR / "config.json"
    if not cfg.exists():
        raise SubmissionError(f"no model config.json under {SUBMISSION_DIR}")
    try:
        arch = json.loads(cfg.read_text()).get("architectures") or []
    except Exception as exc:
        raise SubmissionError(f"unreadable config.json: {exc}") from exc
    if EXPECTED_ARCH not in arch:
        raise SubmissionError(f"wrong architecture {arch}, expected [{EXPECTED_ARCH}]")


def fail_closed(msg: str) -> None:
    """Grader-side fault: reward 0 with a marker the summariser can tell apart
    from a submission that honestly scored 0."""
    REWARD_DIR.mkdir(parents=True, exist_ok=True)
    (REWARD_DIR / "reward.json").write_text(
        json.dumps({"reward": 0.0, "error": f"grader failed: {msg}"}), encoding="utf-8")
    (REWARD_DIR / "reward.txt").write_text("0.0\n", encoding="utf-8")
    print(f"grader failed: {msg}")
    sys.exit(1)


def main() -> None:
    REWARD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        load_anchors()
    except Exception as exc:
        return fail_closed(f"anchors unreadable ({ANCHORS}): {exc}")
    out = {"metric": None, "reward": 0.0, "correctness": False, "errors": [], "per_set": {}}
    grader_error = None
    try:
        from datasets import load_from_disk
        from eval_common import evaluate_model

        check_submission()

        per_set = {}
        for name, path in TEST_SETS.items():
            ds = load_from_disk(str(path))
            acc, details = evaluate_model(str(SUBMISSION_DIR), ds, batch_size=EVAL_BATCH)
            per_set[name] = {"accuracy_pct": round(acc, 4),
                             "correct": details["correct"], "total": details["total"]}
            print(f"[grade] {name}: {acc:.3f}% ({details['correct']}/{details['total']})")

        mean_acc = sum(v["accuracy_pct"] for v in per_set.values()) / max(1, len(per_set))
        out = {
            "metric": round(mean_acc, 4),
            "reward": round(reward_from_metric(mean_acc), 6),
            "correctness": True,
            "errors": [],
            "per_set": per_set,
        }
    except SubmissionError as exc:      # the agent's result: an honest 0, no marker
        out = {"metric": None, "reward": 0.0, "correctness": False,
               "errors": [f"bad submission: {exc}"], "per_set": {}}
    except Exception as exc:            # noqa: BLE001 — grader-side fault
        grader_error = f"grader failed: {type(exc).__name__}: {str(exc)[:240]}"
        out = {"metric": None, "reward": 0.0, "correctness": False,
               "errors": [grader_error], "per_set": {}}

    rewards = {
        "reward": float(out["reward"]),
        "mean_accuracy_pct": float(out["metric"]) if out.get("metric") is not None else 0.0,
        "cogent_b_acc": float(out["per_set"].get("cogent_b", {}).get("accuracy_pct", 0.0)),
        "superclevr_acc": float(out["per_set"].get("superclevr", {}).get("accuracy_pct", 0.0)),
    }
    # The marker has to ride in reward.json itself: that is the only file harbor and
    # test.sh read. Writing it to grade_debug.json alone dropped it on the floor.
    if grader_error:
        rewards["error"] = grader_error
    (REWARD_DIR / "reward.txt").write_text(f"{out['reward']}\n", encoding="utf-8")
    (REWARD_DIR / "reward.json").write_text(json.dumps(rewards), encoding="utf-8")
    (REWARD_DIR / "grade_debug.json").write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps(out))
    if grader_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
