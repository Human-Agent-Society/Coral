from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import protocol
from restricted_runner import restricted_workspace, run_restricted
from submission_guard import violations


TESTS = Path("/tests")
RUNNER = Path("/runner")
# Present only in the author anchor-measurement image (Dockerfile.oracle).
ORACLE = Path("/tests/oracle_mode").is_file()
SUBMISSION = Path("/app/methods/main")
PRODUCTION_BASELINE = Path("/tests/production_baseline")
BRACKET_MAX_MIN_RATIO = 1.25
LOGS = Path("/logs/verifier")



ANCHOR_SPEC = json.loads((Path("/tests") / "anchors.json").read_text())


def assert_expected_gpu():
    expected = ANCHOR_SPEC["measured_on"]
    probe = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.get_device_name(0))"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    actual = probe.stdout.strip()
    if probe.returncode != 0 or not actual:
        raise RuntimeError(f"could not read GPU name: {probe.stderr.strip()}")
    if actual != expected:
        raise RuntimeError(
            f"anchors were measured on {expected!r}, this GPU is {actual!r}; "
            "re-measure the baseline manifest before scoring"
        )


def reward_of(speedup, sota, upper):
    """Piecewise-linear map of one case's raw speedup onto [0, 1]."""
    base = float(ANCHOR_SPEC["anchors"]["baseline"])
    r_sota = float(ANCHOR_SPEC["rewards"]["sota"])
    r_upper = float(ANCHOR_SPEC["rewards"]["upper"])
    if not math.isfinite(speedup) or speedup <= base:
        return 0.0
    if speedup <= sota:
        return r_sota * (speedup - base) / (sota - base)
    return min(r_upper, r_sota + (r_upper - r_sota) * (speedup - sota) / (upper - sota))


def mapped_reward(rows, ok):
    if not ok or not rows:
        return 0.0
    vals = [reward_of(float(r["raw_speedup"]),
                      float(r["human_sota_raw_speedup"]),
                      float(r["upper_bound_raw_speedup"])) for r in rows]
    return math.fsum(vals) / len(vals)


def guard(root):
    return [] if ORACLE else violations(root)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def challenge_seed(case_id: str, panel_sha: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{panel_sha}|{case_id}".encode()).digest()[:8], "big") & 0x7FFF_FFFF


def expected(case, seed):
    args = protocol.make_inputs(case, seed)
    values = protocol.run_reference(args, case)
    arrays = [value.detach().float().cpu().numpy() for value in values]
    del args, values
    gc.collect()
    torch.cuda.empty_cache()
    return arrays


def expected_signature(case, seed, sig_seed):
    args = protocol.make_inputs(case, seed)
    values = protocol.run_reference(args, case)
    sig = protocol.signature(values, sig_seed)
    del args, values
    gc.collect()
    torch.cuda.empty_cache()
    return sig


def execute(submission, case, panel_sha, index):
    cseed = challenge_seed(case["id"], panel_sha)
    with restricted_workspace(submission, guard, case_index=index) as (candidate, runtime, env, uid):
        challenge = runtime / "challenge.json"
        output = runtime / "candidate.json"
        challenge.write_text(json.dumps({"case": case, "challenge_seed": cseed}))
        challenge.chmod(0o444)
        completed = run_restricted(
            ["python", str(RUNNER / "child_eval.py"), str(candidate), str(challenge), str(output)],
            cwd=candidate, env=env, uid=uid, timeout=1200,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{case['id']}: child failed\n{completed.stdout}\n{completed.stderr}")
        report = json.loads(output.read_text())
        report["correctness_arrays"] = [
            np.load(path, allow_pickle=False) for path in report["correctness"]["paths"]
        ]
        report["parent_wall_time_ms"] = completed.wall_time_ms
        return report, cseed


def measure_baseline(case, panel_sha, index):
    report, cseed = execute(PRODUCTION_BASELINE, case, panel_sha, index)
    median = float(report["candidate_median_ms"])
    if report["case_id"] != case["id"] or not math.isfinite(median) or median <= 0.0:
        raise RuntimeError(f"{case['id']}: unusable baseline timing")
    return median


def main():
    # # relaxed-for-dev: assert_expected_gpu() disabled
    LOGS.mkdir(parents=True, exist_ok=True)
    panel_path = TESTS / "heldout" / "test_spec.json"
    baseline_path = TESTS / "heldout" / "baseline_manifest.json"
    panel_sha = sha256(panel_path)
    expected_panel_sha = os.environ.get("HELDOUT_PANEL_SHA256", panel_sha)
    expected_baseline_sha = os.environ.get("BASELINE_MANIFEST_SHA256", sha256(baseline_path))
    if panel_sha != expected_panel_sha or sha256(baseline_path) != expected_baseline_sha:
        raise RuntimeError("frozen panel or baseline manifest hash mismatch")
    panel = json.loads(panel_path.read_text())
    baseline = json.loads(baseline_path.read_text())["cases"]
    details = []
    all_passed = True
    for index, case in enumerate(panel["cases"]):
        before = measure_baseline(case, panel_sha, index)
        report, cseed = execute(SUBMISSION, case, panel_sha, index)
        after = measure_baseline(case, panel_sha, index)
        truth = expected(case, protocol.phase_seed(case, "correctness", 0))
        comparisons = protocol.compare(report["correctness_arrays"], truth, case)
        calls = []
        for row in report["calls"]:
            phase, ridx = row["phase"], int(row["index"])
            offset = 10_000 if phase == "warmup" else 20_000
            sig = expected_signature(
                case, protocol.phase_seed(case, phase, ridx), cseed + offset + ridx
            )
            calls.append({
                "phase": phase, "index": ridx,
                "passed": (
                    row["seed"] == protocol.phase_seed(case, phase, ridx)
                    and row["immutability"]["passed"]
                    and row["structure"]["passed"]
                    and protocol.signature_close(row["signature"], sig)
                ),
            })
        samples = [float(x) for x in report["candidate_repeat_ms"]]
        dispersion = max(samples) / max(min(samples), 1e-9)
        passed = (
            report["case_id"] == case["id"]
            and report["timer"] == {"prebound": True, "fresh_inputs": True, "setup_outside_timing": True}
            and report["correctness"]["immutability"]["passed"]
            and report["correctness"]["structure"]["passed"]
            and all(item["passed"] for item in comparisons)
            and len(calls) == protocol.WARMUPS + protocol.REPEATS
            and all(item["passed"] for item in calls)
            and dispersion <= protocol.MAX_DISPERSION
        )
        candidate_ms = float(report["candidate_median_ms"])
        anchor_row = baseline[case["id"]]
        bracket_drift = max(before, after) / min(before, after)
        bracket_stable = bracket_drift <= BRACKET_MAX_MIN_RATIO
        passed = passed and bracket_stable
        baseline_ms = math.sqrt(before * after)
        speedup = baseline_ms / candidate_ms if passed else 0.0
        all_passed &= passed
        details.append({
            "id": case["id"], "passed": passed,
            "baseline_ms": baseline_ms, "candidate_ms": candidate_ms,
            "baseline_bracket_before_ms": before,
            "baseline_bracket_after_ms": after,
            "baseline_bracket_drift": bracket_drift,
            "baseline_bracket_stable": bracket_stable,
            "raw_speedup": speedup, "dispersion": dispersion,
            "human_sota_raw_speedup": float(anchor_row["human_speedup"]),
            "upper_bound_raw_speedup": float(anchor_row["upper_bound_speedup"]),
            "normalization": None,
            "candidate_repeat_ms": samples,
            "comparisons": comparisons, "calls": calls,
        })
    raw = math.exp(sum(math.log(row["raw_speedup"]) for row in details) / len(details)) if all_passed else 0.0
    payload = {
        "reward": mapped_reward(details, all_passed),
        "raw_speedup": raw,
        "all_cases_passed": int(all_passed),
        "human_sota_raw_speedup": float(os.environ["HUMAN_SOTA_RAW_SPEEDUP"]),
        "upper_bound_raw_speedup": float(os.environ["UPPER_BOUND_RAW_SPEEDUP"])
    }
    (LOGS / "score_details.json").write_text(json.dumps({
        "metric": "geometric_mean_raw_speedup", "aggregate": raw, "cases": details,
    }, indent=2))
    # reward.json carries reward only; everything else is merged into score_details
    try:
        _sd = json.loads((LOGS / "score_details.json").read_text())
    except Exception:
        _sd = {}
    _sd.update({k: v for k, v in payload.items() if k != "reward"})
    (LOGS / "score_details.json").write_text(json.dumps(_sd, indent=2))
    (LOGS / "reward.json").write_text(
        json.dumps({"reward": float(payload["reward"])}, indent=2)
    )
    (LOGS / "reward.txt").write_text(f"{raw:.12f}\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
        Path("/logs/verifier/reward.json").write_text(
            json.dumps({"reward": 0.0}, indent=2))
        raise
