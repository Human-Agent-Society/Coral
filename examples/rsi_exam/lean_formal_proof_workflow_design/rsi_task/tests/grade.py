"""Grade submitted Lean proof workflows on the frozen theorem suite."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from evaluator import SUBMISSION_DIR, evaluate, load_jsonl, runtime_preflight


HELDOUT = Path("/tests/heldout")
REWARD_DIR = Path("/logs/verifier")
EXPECTED_HELDOUT_SHA256 = "44e117f89551d1a7d6b1e1b614359af81ed4e09d78145f91cce839d35605154b"


def to_reward(value):
    return float(value) / 100.0


def write_result(reward, detail):
    REWARD_DIR.mkdir(parents=True, exist_ok=True)
    reward = round(float(reward), 6)
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        raise ValueError("reward must be finite and in [0, 1]")
    canonical = {"reward": reward}
    errors = []
    if not detail.get("verifier_success", False):
        errors.append(str(detail.get("verifier_error") or detail.get("error")
                          or "verifier did not complete successfully"))
    debug = {
        **detail,
        "correctness": bool(detail.get("verifier_success", False)),
        "errors": errors,
        "reward": reward,
    }
    reward_txt = REWARD_DIR / "reward.txt"
    reward_json = REWARD_DIR / "reward.json"
    grade_debug = REWARD_DIR / "grade_debug.json"
    score_details = REWARD_DIR / "score_details.json"
    reward_txt.write_text(str(reward) + "\n", encoding="utf-8")
    reward_json.write_text(
        json.dumps(canonical, indent=2, allow_nan=False), encoding="utf-8"
    )
    grade_debug.write_text(
        json.dumps(debug, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    score_details.write_text(
        json.dumps({**detail, "reward": reward}, ensure_ascii=False, indent=2,
                   allow_nan=False), encoding="utf-8")
    # These files cross the container/host boundary.  Harbor parses reward.json
    # immediately after test.sh exits, before cleanup can chown bind-mounted
    # files to the host user, so their mode must not depend on the caller's
    # umask.  Hidden per-problem details have already been removed upstream.
    for path in (reward_txt, reward_json, grade_debug, score_details):
        path.chmod(0o644)


def seal(path):
    try:
        path.unlink()
    except OSError:
        try:
            path.chmod(0)
        except OSError:
            pass


def sanitize_hidden_result(result):
    """Remove participant-controlled per-case channels from released hidden evidence."""
    result["per_problem"] = []


def main():
    try:
        model_identity = runtime_preflight()
        hidden_path = HELDOUT / "test.jsonl"
        raw = hidden_path.read_bytes()
        source_sha256 = hashlib.sha256(raw).hexdigest()
        if source_sha256 != EXPECTED_HELDOUT_SHA256:
            raise ValueError("hidden dataset does not match the frozen manifest")
        rows = load_jsonl(hidden_path)
        if len(rows) != 40:
            raise ValueError(f"held-out suite must contain 40 rows, found {len(rows)}")
        seal(hidden_path)
        result = evaluate(rows, SUBMISSION_DIR, model_identity=model_identity)
        sanitize_hidden_result(result)
        result["source_file_sha256"] = source_sha256
        result["heldout_status"] = "reused_legacy_hidden_not_fresh"
        result["grader_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["submission_valid"] = not result.get("invalid") and not result.get("error")
        result["verifier_success"] = result.get("n_total") == 40
        reward = to_reward(result["accuracy_pct"]) if result["submission_valid"] else 0.0
        write_result(reward, result)
        print(f"hidden compile accuracy = {result['n_correct']}/{result['n_total']} "
              f"= {result['accuracy_pct']:.2f}% -> reward {reward:.6f}")
    except Exception as exc:
        detail = {"accuracy_pct": 0.0, "n_correct": 0, "n_total": 0,
                  "verifier_success": False,
                  "verifier_error": f"grader failed: {type(exc).__name__}: {exc}",
                  "protocol_version": "lean-proof-workflow-v2.5-gpt54-fixed-statement"}
        write_result(0.0, detail)
        raise


if __name__ == "__main__":
    main()
