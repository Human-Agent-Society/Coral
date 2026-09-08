from __future__ import annotations

import gc
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from protocol import (
    TrustedCudaTimer,
    baseline_decode,
    case_from_dict,
    correctness_report,
    make_inputs,
    make_metadata,
    phase_seed,
    reference_decode,
    timing_report,
)
from restricted_runner import restricted_case_workspace, run_restricted
from submission_guard import violations


ROOT = Path(__file__).resolve().parent
METHOD = ROOT / "methods" / "main"
RUNNER = Path("/runner")
EXPECTED_GPU = "PENDING_H100_REMEASURE"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def execute_child(raw_case: dict, timing: dict, case_index: int) -> dict:
    with restricted_case_workspace(METHOD, violations, case_index=case_index) as (candidate, runtime, env, uid):
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
            raise RuntimeError("challenge was not removed before submission import")
        result = json.loads(result_path.read_text())
        arrays = runtime / "arrays"
        for call in result.get("calls", []):
            name = call.get("array")
            call["output"] = None if name is None else np.load(arrays / name, allow_pickle=False)
        result["parent_wall_time_ms"] = completed.wall_time_ms
        return result


@torch.no_grad()
def expected_call(case, phase: str, index: int, metadata) -> torch.Tensor:
    args = make_inputs(case, seed=phase_seed(case, phase, index), metadata=metadata)
    return reference_decode(*args).detach().cpu()


@torch.no_grad()
def measure_baseline(case, timing: dict, metadata) -> dict:
    warmup = int(timing["warmup"])
    repeats = int(timing["repeats"])
    timed_inner_calls = int(timing["timed_inner_calls"])
    if warmup % timed_inner_calls:
        raise RuntimeError("warmup must be divisible by timed_inner_calls")
    timer = TrustedCudaTimer(repeats)
    required_output_ring_entries = 1 + warmup + repeats * timed_inner_calls
    if (
        timing.get("caller_owned_output_ring") is not True
        or timing.get("output_ring_preallocated_untimed") is not True
        or int(timing.get("output_ring_entries", -1))
        != required_output_ring_entries
        or timing.get("solver_output_copy_into_ring_timed") is not True
        or timing.get("submission_temporary_output_retained") is not False
    ):
        raise RuntimeError("starter baseline output-ring contract drift")
    output_ring = [
        torch.empty(
            (case.batch, case.query_heads, case.head_dim),
            device="cuda",
            dtype=case.dtype,
        )
        for _ in range(required_output_ring_entries)
    ]
    ring_pointers = [
        int(output.untyped_storage().data_ptr()) for output in output_ring
    ]
    if len(set(ring_pointers)) != required_output_ring_entries:
        raise RuntimeError("starter baseline output ring aliases storage")
    output_ring_cursor = 0

    def ring_call(args):
        nonlocal output_ring_cursor
        if output_ring_cursor >= len(output_ring):
            raise RuntimeError("starter baseline output ring exhausted")
        submission_output = baseline_decode(*args)
        target = output_ring[output_ring_cursor]
        output_ring_cursor += 1
        target.copy_(submission_output)
        return target, submission_output

    def prepare_group(phase: str, start: int):
        return [
            make_inputs(
                case,
                seed=phase_seed(case, phase, index),
                metadata=metadata,
            )
            for index in range(start, start + timed_inner_calls)
        ]

    correctness_args = make_inputs(
        case,
        seed=phase_seed(case, "correctness", 0),
        metadata=metadata,
    )
    correctness_outputs = ring_call(correctness_args)
    timer.synchronize()
    del correctness_args, correctness_outputs

    batch_wall_start = time.perf_counter()
    for group_start in range(0, warmup, timed_inner_calls):
        group = prepare_group("warmup", group_start)
        if group_start == 0:
            timer.prime_clock()
        outputs = tuple(ring_call(args) for args in group)
        timer.synchronize()
        del group, outputs
    values = []
    for event_index in range(repeats):
        group = prepare_group("timed", event_index * timed_inner_calls)
        outputs, elapsed, _ = timer.measure_one(
            event_index,
            lambda current=group: tuple(ring_call(args) for args in current),
        )
        values.append(elapsed / timed_inner_calls)
        del group, outputs
    parent_wall_ms = (time.perf_counter() - batch_wall_start) * 1.0e3
    if output_ring_cursor != required_output_ring_entries:
        raise RuntimeError("starter baseline output ring was not fully consumed")
    report = timing_report(
        values,
        parent_wall_ms,
        float(timing["max_min_ratio"]),
        int(timing["symmetric_trim_each_side"]),
    )
    if not report["passed"]:
        raise RuntimeError(
            f"{case.case_id}: starter baseline timing hard gate failed: "
            f"{report['hard_gate_reasons']}"
        )
    return {
        "median_ms": float(statistics.median(values)),
        "repeat_cuda_ms": values,
        "repeat_full_ratio": report["full_max_min_ratio"],
        "repeat_trimmed_ratio": report["trimmed_max_min_ratio"],
        "dispersion_diagnostic_limit": report["diagnostic_max_min_ratio"],
        "dispersion_diagnostic_within_limit": report["diagnostic_within_limit"],
        "dispersion_hard_gate": report["dispersion_hard_gate"],
        "robust_interval": report["robust_interval"],
        "outlier_rule": report["outlier_rule"],
        "parent_wall_time_ms": parent_wall_ms,
        "parent_wall_to_gpu_sum_ratio": report["parent_wall_to_gpu_sum_ratio"],
        "wall_time_possible": report["wall_time_possible"],
        "stability": report["stability"],
        "caller_owned_output_ring": True,
        "output_ring_preallocated_untimed": True,
        "output_ring_entries": required_output_ring_entries,
        "output_ring_unique_storage_count": len(set(ring_pointers)),
        "solver_output_copy_into_ring_timed": True,
        "submission_temporary_output_retained": False,
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("[selfcheck] no CUDA device visible")
        return 2
    gpu = torch.cuda.get_device_name(0)
    # # relaxed-for-dev: card identity not enforced
    panel_path = ROOT / "problems" / "visible_spec.json"
    panel = json.loads(panel_path.read_text())
    timing = panel["timing"]
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
        or timing.get("caller_owned_output_ring") is not True
        or timing.get("output_ring_preallocated_untimed") is not True
        or int(timing.get("output_ring_entries", -1)) != 297
        or timing.get("solver_output_copy_into_ring_timed") is not True
        or timing.get("submission_temporary_output_retained") is not False
        or int(timing.get("resident_calls_per_group", -1)) != 8
        or int(timing.get("max_peak_allocated_bytes", -1)) != 36 * 1024**3
        or int(timing.get("max_peak_reserved_bytes", -1)) != 40 * 1024**3
    ):
        raise RuntimeError("visible timing protocol mismatch")
    rows = []
    for case_index, raw_case in enumerate(panel["cases"]):
        case = case_from_dict(raw_case)
        metadata = make_metadata(case)
        baseline_before = measure_baseline(case, timing, metadata)
        clean_cuda()
        child = execute_child(raw_case, timing, case_index)
        if child.get("case_id") != case.case_id:
            raise RuntimeError(f"{case.case_id}: wrong child case id")
        timer_meta = child.get("timer", {})
        memory = child.get("memory", {})
        timer_ok = bool(
            timer_meta.get("created_before_submission_import") is True
            and timer_meta.get("fresh_values_per_call") is True
            and timer_meta.get("fresh_output_storage_per_call") is True
            and timer_meta.get("caller_owned_output_ring") is True
            and timer_meta.get("output_ring_preallocated_untimed") is True
            and int(timer_meta.get("output_ring_entries", -1)) == 297
            and int(timer_meta.get("output_ring_unique_storage_count", -1)) == 297
            and timer_meta.get("output_ring_all_storages_unique") is True
            and timer_meta.get("solver_output_copy_into_ring_timed") is True
            and timer_meta.get("submission_temporary_output_retained") is False
            and timer_meta.get("streamed_fresh_input_groups") is True
            and timer_meta.get("resident_calls_per_group")
            == int(timing["timed_inner_calls"])
            and timer_meta.get("cpu_exact_mutation_snapshots") is True
            and timer_meta.get("cuda_snapshot_bytes") == 0
            and timer_meta.get("input_setup_inside_timing") is False
            and timer_meta.get("validation_inside_timing") is False
            and timer_meta.get("contiguous_calls_inside_each_timed_event") is True
            and timer_meta.get("synchronize_after_each_timed_event") is True
            and timer_meta.get("warmup_batch_size") == int(timing["warmup"])
            and timer_meta.get("warmup_group_count")
            == int(timing["warmup"]) // int(timing["timed_inner_calls"])
            and timer_meta.get("fixed_steady_state_warmup") is True
            and timer_meta.get("adaptive_warmup_or_posthoc_selection") is False
            and timer_meta.get("timed_batch_size") == int(timing["repeats"])
            and timer_meta.get("timed_event_count") == int(timing["repeats"])
            and timer_meta.get("timed_calls_per_event")
            == int(timing["timed_inner_calls"])
            and timer_meta.get("timed_call_count")
            == int(timing["repeats"]) * int(timing["timed_inner_calls"])
            and timer_meta.get("event_latency_divided_by_timed_calls") is True
            and timer_meta.get("clock_primer_rounds") == 64
            and timer_meta.get("clock_primer_immediately_before_warmup") is True
            and timer_meta.get("symmetric_trim_each_side")
            == int(timing["symmetric_trim_each_side"])
            and timer_meta.get("stability_retained_count") == 13
            and timer_meta.get("dispersion_hard_gate") is True
            and timer_meta.get("stable_page_table_per_case") is True
            and timer_meta.get("sealed_challenge_unlinked_before_submission_import") is True
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
        checks = []
        timed_events = {}
        for call in child.get("calls", []):
            expected = expected_call(case, str(call["phase"]), int(call["index"]), metadata)
            output = call.pop("output")
            if output is None:
                accuracy = {"passed": False, "reason": "missing output"}
            else:
                accuracy = correctness_report(torch.from_numpy(output), expected, case)
            passed = bool(
                call["immutability"]["passed"]
                and call["structure"]["passed"]
                and call.get("submission_structure", {}).get("passed") is True
                and accuracy["passed"]
            )
            checks.append({"phase": call["phase"], "index": call["index"], "passed": passed, "accuracy": accuracy})
            if call["phase"] == "timed":
                event_index = int(call["timed_event_index"])
                timed_events.setdefault(event_index, []).append(
                    (float(call["elapsed_ms"]), float(call["wall_ms"]))
                )
            elif call.get("timed_event_index") is not None:
                raise RuntimeError(
                    f"{case.case_id}: untimed call has an event index"
                )
        repeat_ms = [float(x) for x in child.get("candidate_repeat_ms", [])]
        if sorted(timed_events) != list(range(int(timing["repeats"]))):
            raise RuntimeError(f"{case.case_id}: incomplete timed event schedule")
        for event_index, samples in timed_events.items():
            if (
                len(samples) != int(timing["timed_inner_calls"])
                or any(sample != samples[0] for sample in samples[1:])
                or repeat_ms[event_index] != samples[0][0]
            ):
                raise RuntimeError(
                    f"{case.case_id}: invalid group-{event_index} timing audit"
                )
        timing_meta = timing_report(
            repeat_ms,
            float(child["parent_wall_time_ms"]),
            float(timing["max_min_ratio"]),
            int(timing["symmetric_trim_each_side"]),
        )
        if not timer_ok or not checks or not all(row["passed"] for row in checks) or not timing_meta["passed"]:
            raise RuntimeError(f"{case.case_id}: correctness, freshness, or timing gate failed")
        candidate_ms = float(timing_meta["median_ms"])
        clean_cuda()
        baseline_after = measure_baseline(case, timing, metadata)
        baseline_before_ms = float(baseline_before["median_ms"])
        baseline_after_ms = float(baseline_after["median_ms"])
        bracket_drift_ratio = max(baseline_before_ms, baseline_after_ms) / min(
            baseline_before_ms, baseline_after_ms
        )
        bracket_drift_ceiling = float(timing["bracket_max_min_ratio"])
        if bracket_drift_ratio > bracket_drift_ceiling:
            raise RuntimeError(
                f"{case.case_id}: baseline bracket drift "
                f"{bracket_drift_ratio:.6f} exceeds {bracket_drift_ceiling:.6f}"
            )
        baseline_ms = math.sqrt(baseline_before_ms * baseline_after_ms)
        rows.append({
            "id": case.case_id,
            "dtype": case.dtype_name,
            "candidate_median_ms": candidate_ms,
            "starter_before_ms": baseline_before_ms,
            "starter_after_ms": baseline_after_ms,
            "starter_bracket_ms": baseline_ms,
            "speedup_over_starter": baseline_ms / candidate_ms,
            "candidate_repeat_cuda_ms": repeat_ms,
            "repeat_full_ratio": timing_meta["full_max_min_ratio"],
            "repeat_trimmed_ratio": timing_meta["trimmed_max_min_ratio"],
            "dispersion_diagnostic_limit": timing_meta["diagnostic_max_min_ratio"],
            "dispersion_diagnostic_within_limit": timing_meta["diagnostic_within_limit"],
            "dispersion_hard_gate": timing_meta["dispersion_hard_gate"],
            "robust_interval": timing_meta["robust_interval"],
            "outlier_rule": timing_meta["outlier_rule"],
            "baseline_bracket_drift_ratio": bracket_drift_ratio,
            "baseline_bracket_drift_ceiling": bracket_drift_ceiling,
            "candidate_wall_time_possible": timing_meta["wall_time_possible"],
            "starter_before": baseline_before,
            "starter_after": baseline_after,
            "stability": timing_meta["stability"],
        })
        clean_cuda()
    candidate_mean_ms = statistics.mean(row["candidate_median_ms"] for row in rows)
    metric = statistics.geometric_mean(row["speedup_over_starter"] for row in rows)
    print(
        "[selfcheck] correctness=True "
        f"visible_geomean_speedup={metric:.9f} "
        f"higher_is_better=True "
        f"visible_mean_per_case_median_cuda_ms={candidate_mean_ms:.9f} "
        f"cases={len(rows)}"
    )
    print(json.dumps({
        "schema_version": 2,
        "status": "ok",
        "protocol": "paged_ragged_gqa_decode_v25_d256_extreme_gqa_streamed_exact_snapshot",
        "panel_sha256": sha256(panel_path),
        "metric": metric,
        "metric_name": "visible_geomean_bracketed_raw_speedup",
        "higher_is_better": True,
        "baseline": 1.0,
        "visible_geomean_speedup": metric,
        "visible_mean_per_case_median_cuda_ms": candidate_mean_ms,
        "cases": rows,
    }, indent=2))
    if Path("/app/budget.py").exists():
        import subprocess

        subprocess.run([sys.executable, "/app/budget.py"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
