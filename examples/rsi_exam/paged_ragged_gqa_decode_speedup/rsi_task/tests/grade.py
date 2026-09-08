from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

from protocol import (
    case_from_dict,
    correctness_report,
    make_inputs,
    make_metadata,
    phase_seed,
    reference_decode,
    timing_report,
)
from restricted_runner import restricted_case_workspace, run_restricted
from speedup_math import bracketed_raw_speedup, geometric_mean_speedup
from submission_guard import violations


TESTS = Path("/tests")
RUNNER = Path("/runner")
SUBMISSION = Path("/app/methods/main")
PRODUCTION_BASELINE = TESTS / "production_baseline"
LOGS = Path("/logs/verifier")
ANCHOR_FILE = TESTS / "anchors.json"
ANCHOR_SPEC = json.loads(ANCHOR_FILE.read_text())

EXPECTED_GPU_NAME = ANCHOR_SPEC["measured_on"]

EXPECTED_BASELINE_SHA256 = (
    "25d150e3f6c8a14faea71dbb6a9bda45e7dbf54cd778297bb2fdf0c426f351c3"
)
REQUIRED_CPU_AFFINITY = sorted(os.sched_getaffinity(0))
EXPECTED_HASHES = {
    "visible": "ff07368f227e7e6cc26a1c816ebd83c8a16b528b9a2cf173d6d0c31d848db087",
    "calibration": "47d206fb07838f7b4e4940fb5aed8d5282a69430a79b68481cfe129b028fe9d3",
    "heldout": "06e792e160d41e1a6fd48662b14e30655ce9d4dcdfed7ddbfa57d4807a23b388",
}
EXPECTED_HUMAN_SOTA = float(ANCHOR_SPEC["anchors"]["sota"])
EXPECTED_UPPER_BOUND = float(ANCHOR_SPEC["anchors"]["upper"])



def reward_of(speedup, sota, upper):
    """Piecewise-linear map of one case's raw speedup onto [0, 1]."""
    base = float(ANCHOR_SPEC["anchors"]["baseline"])
    r_sota = float(ANCHOR_SPEC["rewards"]["sota"])
    r_upper = float(ANCHOR_SPEC["rewards"]["upper"])
    if not math.isfinite(speedup) or speedup <= base:
        return 0.0
    if speedup <= sota:
        return r_sota * (speedup - base) / (sota - base)
    return min(
        r_upper,
        r_sota + (r_upper - r_sota) * (speedup - sota) / (upper - sota),
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_results(reward: dict, details: dict) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "reward.json").write_text(json.dumps(reward, indent=2) + "\n")
    (LOGS / "score_details.json").write_text(json.dumps(details, indent=2) + "\n")


def clean_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def execute_child(source: Path, raw_case: dict, timing: dict, case_index: int) -> dict:
    with restricted_case_workspace(source, violations, case_index=case_index) as (candidate, runtime, env, uid):
        challenge = runtime / "challenge.json"
        result_path = runtime / "result.json"
        challenge.write_text(json.dumps({"case": raw_case, "timing": timing}))
        challenge.chmod(0o444)
        completed = run_restricted(
            ["python", str(RUNNER / "child_eval.py"), str(candidate), str(challenge), str(result_path)],
            cwd=candidate,
            env=env,
            uid=uid,
            timeout_seconds=900,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"child failed: {completed.stdout}\n{completed.stderr}")
        if challenge.exists():
            raise RuntimeError("sealed challenge was not removed before submission import")
        result = json.loads(result_path.read_text())
        arrays = runtime / "arrays"
        for call in result.get("calls", []):
            name = call.get("array")
            if name is None:
                call["output"] = None
            else:
                if not isinstance(name, str) or Path(name).name != name:
                    raise RuntimeError("unsafe child output filename")
                path = arrays / name
                if path.is_symlink() or not path.is_file():
                    raise RuntimeError("missing or unsafe child output")
                call["output"] = np.load(path, allow_pickle=False)
        result["parent_wall_time_ms"] = completed.wall_time_ms
        return result


@torch.no_grad()
def expected_call(case, phase: str, index: int, metadata) -> torch.Tensor:
    args = make_inputs(case, seed=phase_seed(case, phase, index), metadata=metadata)
    return reference_decode(*args).detach().cpu()


def evaluate_submission(
    source: Path,
    raw_case: dict,
    timing: dict,
    process_index: int,
) -> dict:
    case = case_from_dict(raw_case)
    metadata = make_metadata(case)
    child = execute_child(source, raw_case, timing, process_index)
    if child.get("case_id") != case.case_id:
        raise RuntimeError(f"{case.case_id}: wrong child case id")
    timer = child.get("timer", {})
    memory = child.get("memory", {})
    timer_ok = bool(
        timer.get("created_before_submission_import") is True
        and timer.get("fresh_values_per_call") is True
        and timer.get("fresh_output_storage_per_call") is True
        and timer.get("caller_owned_output_ring") is True
        and timer.get("output_ring_preallocated_untimed") is True
        and int(timer.get("output_ring_entries", -1))
        == 1 + int(timing["warmup"])
        + int(timing["repeats"]) * int(timing["timed_inner_calls"])
        and int(timer.get("output_ring_unique_storage_count", -1))
        == int(timer.get("output_ring_entries", -2))
        and timer.get("output_ring_all_storages_unique") is True
        and timer.get("solver_output_copy_into_ring_timed") is True
        and timer.get("submission_temporary_output_retained") is False
        and timer.get("streamed_fresh_input_groups") is True
        and timer.get("resident_calls_per_group") == int(timing["timed_inner_calls"])
        and timer.get("cpu_exact_mutation_snapshots") is True
        and timer.get("cuda_snapshot_bytes") == 0
        and timer.get("input_setup_inside_timing") is False
        and timer.get("validation_inside_timing") is False
        and timer.get("contiguous_calls_inside_each_timed_event") is True
        and timer.get("synchronize_after_each_timed_event") is True
        and timer.get("warmup_batch_size") == int(timing["warmup"])
        and timer.get("warmup_group_count")
        == int(timing["warmup"]) // int(timing["timed_inner_calls"])
        and timer.get("fixed_steady_state_warmup") is True
        and timer.get("adaptive_warmup_or_posthoc_selection") is False
        and timer.get("timed_batch_size") == int(timing["repeats"])
        and timer.get("timed_event_count") == int(timing["repeats"])
        and timer.get("timed_calls_per_event")
        == int(timing["timed_inner_calls"])
        and timer.get("timed_call_count")
        == int(timing["repeats"]) * int(timing["timed_inner_calls"])
        and timer.get("event_latency_divided_by_timed_calls") is True
        and timer.get("clock_primer_rounds") == 64
        and timer.get("clock_primer_immediately_before_warmup") is True
        and timer.get("symmetric_trim_each_side") == int(timing["symmetric_trim_each_side"])
        and timer.get("stability_retained_count") == 13
        and timer.get("dispersion_hard_gate") is True
        and timer.get("required_cpu_affinity") == REQUIRED_CPU_AFFINITY
        and timer.get("cpu_affinity_before_submission_import") == REQUIRED_CPU_AFFINITY
        and timer.get("cpu_affinity_after_all_candidate_calls") == REQUIRED_CPU_AFFINITY
        and timer.get("sched_affinity_verified") is True
        and timer.get("stable_page_table_per_case") is True
        and timer.get("sealed_challenge_unlinked_before_submission_import") is True
        and memory.get("passed") is True
        and int(memory.get("max_peak_allocated_bytes", -1))
        == int(timing["max_peak_allocated_bytes"])
        and int(memory.get("max_peak_reserved_bytes", -1))
        == int(timing["max_peak_reserved_bytes"])
        and 0 < int(memory.get("peak_allocated_bytes", -1))
        <= int(timing["max_peak_allocated_bytes"])
        and 0 < int(memory.get("peak_reserved_bytes", -1))
        <= int(timing["max_peak_reserved_bytes"])
    )
    expected_schedule = [("correctness", 0)]
    expected_schedule.extend(("warmup", index) for index in range(int(timing["warmup"])))
    expected_schedule.extend(
        ("timed", index)
        for index in range(
            int(timing["repeats"]) * int(timing["timed_inner_calls"])
        )
    )
    raw_calls = child.get("calls", [])
    if len(raw_calls) != len(expected_schedule):
        raise RuntimeError(f"{case.case_id}: incomplete call schedule")
    calls = []
    timed_events = {}
    for call, (expected_phase, expected_index) in zip(raw_calls, expected_schedule):
        phase, index = str(call["phase"]), int(call["index"])
        expected_seed = phase_seed(case, expected_phase, expected_index)
        if (phase, index) != (expected_phase, expected_index) or int(call["seed"]) != expected_seed:
            raise RuntimeError(f"{case.case_id}: invalid call schedule or seed")
        elapsed, wall_ms = call.get("elapsed_ms"), call.get("wall_ms")
        if phase == "timed":
            elapsed, wall_ms = float(elapsed), float(wall_ms)
            if not (math.isfinite(elapsed) and elapsed > 0 and math.isfinite(wall_ms) and wall_ms > 0):
                raise RuntimeError(f"{case.case_id}: invalid timed measurement")
            event_index = int(call.get("timed_event_index", -1))
            if not 0 <= event_index < int(timing["repeats"]):
                raise RuntimeError(f"{case.case_id}: invalid timed event index")
            timed_events.setdefault(event_index, []).append((elapsed, wall_ms))
        elif elapsed is not None or wall_ms is not None:
            raise RuntimeError(f"{case.case_id}: untimed call reported a measurement")
        elif call.get("timed_event_index") is not None:
            raise RuntimeError(f"{case.case_id}: untimed call has an event index")
        expected = expected_call(case, phase, index, metadata)
        output = call.pop("output")
        accuracy = (
            {"passed": False, "reason": "missing output"}
            if output is None
            else correctness_report(torch.from_numpy(output), expected, case)
        )
        passed = bool(
            call["immutability"]["passed"]
            and call["structure"]["passed"]
            and call.get("submission_structure", {}).get("passed") is True
            and accuracy["passed"]
        )
        calls.append({"phase": phase, "index": index, "seed": expected_seed,
                      "passed": passed, "accuracy": accuracy})
    repeats = [float(value) for value in child.get("candidate_repeat_ms", [])]
    if len(repeats) != int(timing["repeats"]):
        raise RuntimeError(f"{case.case_id}: wrong retained event count")
    if sorted(timed_events) != list(range(int(timing["repeats"]))):
        raise RuntimeError(f"{case.case_id}: incomplete timed event schedule")
    for event_index, samples in timed_events.items():
        if (
            len(samples) != int(timing["timed_inner_calls"])
            or any(sample != samples[0] for sample in samples[1:])
            or repeats[event_index] != samples[0][0]
        ):
            raise RuntimeError(
                f"{case.case_id}: repeat vector does not match grouped calls"
            )
    timing_check = timing_report(
        repeats,
        float(child["parent_wall_time_ms"]),
        float(timing["max_min_ratio"]),
        int(timing["symmetric_trim_each_side"]),
    )
    correct = bool(timer_ok and calls and all(row["passed"] for row in calls) and timing_check["passed"])
    raw_latency = float(timing_check["median_ms"])
    clean_cuda()
    return {
        "correct": correct,
        "raw_latency_ms": raw_latency,
        "repeat_cuda_ms": repeats,
        "repeat_full_ratio": timing_check["full_max_min_ratio"],
        "repeat_trimmed_ratio": timing_check["trimmed_max_min_ratio"],
        "dispersion_diagnostic_limit": timing_check["diagnostic_max_min_ratio"],
        "dispersion_diagnostic_within_limit": timing_check["diagnostic_within_limit"],
        "dispersion_hard_gate": timing_check["dispersion_hard_gate"],
        "robust_interval": timing_check["robust_interval"],
        "outlier_rule": timing_check["outlier_rule"],
        "wall_time_possible": timing_check["wall_time_possible"],
        "timing_hard_gate_reasons": timing_check["hard_gate_reasons"],
        "stability": timing_check["stability"],
        "call_checks": calls,
    }


def evaluate_case(raw_case: dict, timing: dict, case_index: int) -> dict:
    # A same-verifier geometric bracket controls monotonic clock/thermal drift:
    # immutable baseline before -> candidate -> immutable baseline after.
    process_base = case_index * 3
    baseline_before = evaluate_submission(
        PRODUCTION_BASELINE, raw_case, timing, process_base
    )
    candidate = evaluate_submission(
        SUBMISSION, raw_case, timing, process_base + 1
    )
    baseline_after = evaluate_submission(
        PRODUCTION_BASELINE, raw_case, timing, process_base + 2
    )
    roles_correct = bool(
        baseline_before["correct"]
        and candidate["correct"]
        and baseline_after["correct"]
    )
    before_ms = float(baseline_before["raw_latency_ms"])
    after_ms = float(baseline_after["raw_latency_ms"])
    bracket_drift_ratio = (
        max(before_ms, after_ms) / min(before_ms, after_ms)
        if all(math.isfinite(value) and value > 0.0 for value in (before_ms, after_ms))
        else math.inf
    )
    bracket_drift_ceiling = float(timing["bracket_max_min_ratio"])
    bracket_stable = bool(bracket_drift_ratio <= bracket_drift_ceiling)
    correct = bool(roles_correct and bracket_stable)
    baseline_ms = 0.0
    raw_speedup = 0.0
    if correct:
        baseline_ms, raw_speedup = bracketed_raw_speedup(
            baseline_before["raw_latency_ms"],
            candidate["raw_latency_ms"],
            baseline_after["raw_latency_ms"],
        )
    return {
        "id": str(raw_case["id"]),
        "correct": correct,
        "baseline_before_ms": baseline_before["raw_latency_ms"],
        "baseline_after_ms": baseline_after["raw_latency_ms"],
        "baseline_bracket_ms": baseline_ms,
        "candidate_ms": candidate["raw_latency_ms"],
        "raw_speedup": raw_speedup,
        "baseline_bracket_drift_ratio": bracket_drift_ratio,
        "baseline_bracket_drift_ceiling": bracket_drift_ceiling,
        "baseline_bracket_stable": bracket_stable,
        "baseline_before": baseline_before,
        "candidate": candidate,
        "baseline_after": baseline_after,
    }


def main() -> int:
    try:
        actual_cpu_affinity = sorted(os.sched_getaffinity(0))
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device visible")
        actual_gpu = torch.cuda.get_device_name(0)
        # # relaxed-for-dev: card identity not enforced

        panel_path = TESTS / "heldout" / "test_spec.json"
        calibration_path = TESTS / "calibration" / "test_spec.json"
        visible_path = TESTS / "visible_spec.json"
        actual_hashes = {
            "visible": sha256(visible_path),
            "calibration": sha256(calibration_path),
            "heldout": sha256(panel_path),
        }
        for name, actual in actual_hashes.items():
            if EXPECTED_HASHES[name] != actual:
                raise RuntimeError(f"{name} hash mismatch")
        actual_baseline_hash = sha256(PRODUCTION_BASELINE / "solver.py")
        if actual_baseline_hash != EXPECTED_BASELINE_SHA256:
            raise RuntimeError("verifier-owned production baseline hash mismatch")

        panel = json.loads(panel_path.read_text())
        benchmark_context = json.loads(
            (TESTS / "heldout" / "benchmark_context.json").read_text()
        )
        if not math.isclose(
            float(benchmark_context["human_sota_raw_speedup"]),
            EXPECTED_HUMAN_SOTA,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("heldout Human-SOTA anchor mismatch")
        if not math.isclose(
            float(benchmark_context["upper_bound_raw_speedup"]),
            EXPECTED_UPPER_BOUND,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("heldout upper-bound anchor mismatch")
        context_by_id = {
            row["id"]: row for row in benchmark_context["cases"]
        }
        timing = panel.get("timing", {})
        if (
            panel.get("protocol")
            != "paged_ragged_gqa_decode_v25_d256_extreme_gqa_streamed_exact_snapshot"
            or int(timing.get("warmup", -1)) != 128
            or int(timing.get("repeats", -1)) != 21
            or int(timing.get("timed_inner_calls", -1)) != 8
            or float(timing.get("max_min_ratio", -1.0)) != 1.20
            or int(timing.get("symmetric_trim_each_side", -1)) != 4
            or float(timing.get("bracket_max_min_ratio", -1.0)) != 1.20
            or timing.get("dispersion_hard_gate") is not True
            or timing.get("streamed_fresh_input_groups") is not True
            or timing.get("cpu_exact_mutation_snapshots") is not True
            or int(timing.get("resident_calls_per_group", -1)) != 8
            or int(timing.get("max_peak_allocated_bytes", -1)) != 36 * 1024**3
            or int(timing.get("max_peak_reserved_bytes", -1)) != 40 * 1024**3
        ):
            raise RuntimeError("heldout timing protocol mismatch")
        rows = [
            evaluate_case(raw_case, timing, case_index)
            for case_index, raw_case in enumerate(panel["cases"])
        ]
        if {row["id"] for row in rows} != set(context_by_id):
            raise RuntimeError("heldout benchmark-context case IDs mismatch")
        for row in rows:
            anchor = context_by_id[row["id"]]
            row["baseline_raw_speedup"] = 1.0
            row["human_sota_raw_speedup"] = float(
                anchor["human_sota_raw_speedup"]
            )
            row["upper_bound_raw_speedup"] = float(
                anchor["upper_bound_raw_speedup"]
            )
        all_correct = all(row["correct"] for row in rows)
        for row in rows:
            row["reward"] = reward_of(
                float(row["raw_speedup"]),
                float(row["human_sota_raw_speedup"]),
                float(row["upper_bound_raw_speedup"]),
            )
        raw_speedup = (
            geometric_mean_speedup([row["raw_speedup"] for row in rows])
            if all_correct
            else 0.0
        )
        reward = (
            math.fsum(row["reward"] for row in rows) / len(rows)
            if all_correct
            else 0.0
        )
        # reward.json holds finite numbers ONLY -- harbor drops the whole run
        # on a string, a bool, a null or a nested value.
        reward_doc = {
            "reward": reward,
            "raw_speedup": raw_speedup,
            "all_cases_passed": int(all_correct),
            "case_count": len(rows),
            "grader_failed": 0,
        }
        details = {
            "schema_version": 2,
            "status": "scored",
            "reward": reward,
            "anchors": ANCHOR_SPEC["anchors"],
            "anchor_rewards": ANCHOR_SPEC["rewards"],
            "segment_shape": ANCHOR_SPEC["segment_shape"],
            "protocol": "paged_ragged_gqa_decode_v25_d256_extreme_gqa_streamed_exact_snapshot",
            "hardware": actual_gpu,
            "cpu_affinity": actual_cpu_affinity,
            "production_baseline_sha256": actual_baseline_hash,
            "measurement_order": "baseline-before, candidate, baseline-after per case",
            "per_case_speedup": "sqrt(baseline_before_ms * baseline_after_ms) / candidate_ms",
            "aggregation": "geometric mean of per-case raw speedups",
            "baseline_raw_speedup": 1.0,
            "human_sota_raw_speedup": EXPECTED_HUMAN_SOTA,
            "upper_bound_raw_speedup": EXPECTED_UPPER_BOUND,
            "latency_estimator": (
                "ordinary median of 21 retained CUDA-event group means; "
                "each event encloses exactly eight fresh calls"
            ),
            "dispersion_gate": (
                "hard fail when either fixed event-parity phase exceeds 1.20x "
                "after excluding two low and two high samples within that phase; "
                "all 21 samples still score"
            ),
            "baseline_bracket_gate": "hard fail when before/after medians differ by more than 1.20x",
            "cases": rows,
        }
        write_results(reward_doc, details)
        print(json.dumps(reward_doc))
        return 0
    except Exception as exc:
        reward_doc = {
            "reward": 0.0,
            "raw_speedup": 0.0,
            "all_cases_passed": 0,
            "case_count": 0,
            "grader_failed": 1,
        }
        write_results(
            reward_doc,
            {
                "status": "failed",
                "grader_failed": 1,
                "cases": [],
                "errors": [f"grader failed: {type(exc).__name__}: {exc}"],
            },
        )
        print(json.dumps(reward_doc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
