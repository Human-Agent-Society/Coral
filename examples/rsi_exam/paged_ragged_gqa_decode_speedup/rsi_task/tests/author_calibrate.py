from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


PINNED_UPSTREAM_COMMIT = "6104afe3777bcfb16a71fa28eb73b51b7e5e5332"
PINNED_FLASHINFER_VERSION = "0.6.15"
EXPECTED_BASELINE_SHA256 = "25d150e3f6c8a14faea71dbb6a9bda45e7dbf54cd778297bb2fdf0c426f351c3"
EXPECTED_SOTA_SHA256 = "e1119584a208d2e493314c12c627366b9469813052f931b34dcf32d0855fe408"
EXPECTED_GPU_NAME = os.environ.get("ARB_TARGET_GPU", "NVIDIA RTX A6000")
REQUIRED_CPU_AFFINITY = [4]
RUNNER = Path(os.environ.get("PAGED_GQA_RUNNER_ROOT", "/runner"))
sys.path.insert(0, str(RUNNER))

from protocol import (  # noqa: E402
    case_from_dict,
    correctness_report,
    make_inputs,
    make_metadata,
    phase_seed,
    reference_decode,
    timing_report,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def refuse_sealed_output(path: Path) -> None:
    resolved = path.resolve()
    parts = set(resolved.parts)
    if resolved.name == "anchors.json" and {"tests", "heldout"}.issubset(parts):
        raise ValueError("calibration tools never write the sealed heldout anchor file")
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {resolved}")


def run_child(submission: Path, raw_case: dict, timing: dict) -> tuple[dict, Path, float, tempfile.TemporaryDirectory]:
    temporary = tempfile.TemporaryDirectory(prefix="paged-gqa-author-")
    runtime = Path(temporary.name)
    challenge = runtime / "challenge.json"
    result_path = runtime / "result.json"
    challenge.write_text(json.dumps({"case": raw_case, "timing": timing}))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(RUNNER)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    started = time.monotonic()
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER / "child_eval.py"),
            str(submission),
            str(challenge),
            str(result_path),
        ],
        cwd=submission,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=1800,
        check=False,
    )
    wall_ms = (time.monotonic() - started) * 1.0e3
    if completed.returncode != 0:
        temporary.cleanup()
        raise RuntimeError(
            f"trusted calibration child failed for {raw_case['id']}:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return json.loads(result_path.read_text()), runtime, wall_ms, temporary


@torch.no_grad()
def evaluate_role(submission: Path, raw_case: dict, timing: dict) -> dict:
    case = case_from_dict(raw_case)
    child, runtime, parent_wall_ms, temporary = run_child(submission, raw_case, timing)
    try:
        timer = child.get("timer", {})
        memory = child.get("memory", {})
        timer_protocol_ok = bool(
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
            and timer.get("resident_calls_per_group")
            == int(timing["timed_inner_calls"])
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
        expected_order = (
            [("correctness", 0)]
            + [("warmup", index) for index in range(int(timing["warmup"]))]
            + [
                ("timed", index)
                for index in range(
                    int(timing["repeats"]) * int(timing["timed_inner_calls"])
                )
            ]
        )
        calls = child.get("calls", [])
        actual_order = [(str(row.get("phase")), int(row.get("index", -1))) for row in calls]
        if actual_order != expected_order:
            raise RuntimeError(f"{case.case_id}: child call schedule mismatch")

        metadata = make_metadata(case)
        checks = []
        timed_events = {}
        for row in calls:
            array_name = row.get("array")
            if not array_name:
                checks.append(False)
                continue
            array_path = runtime / "arrays" / str(array_name)
            output = torch.from_numpy(np.load(array_path, allow_pickle=False))
            args = make_inputs(
                case,
                seed=phase_seed(case, str(row["phase"]), int(row["index"])),
                metadata=metadata,
            )
            expected = reference_decode(*args).detach().cpu()
            accuracy = correctness_report(output, expected, case)
            checks.append(bool(
                row.get("immutability", {}).get("passed")
                and row.get("structure", {}).get("passed")
                and row.get("submission_structure", {}).get("passed")
                and accuracy["passed"]
            ))
            if str(row["phase"]) == "timed":
                event_index = int(row.get("timed_event_index", -1))
                if not 0 <= event_index < int(timing["repeats"]):
                    raise RuntimeError(f"{case.case_id}: invalid timed event index")
                timed_events.setdefault(event_index, []).append(
                    (float(row["elapsed_ms"]), float(row["wall_ms"]))
                )
            elif row.get("timed_event_index") is not None:
                raise RuntimeError(f"{case.case_id}: untimed call has an event index")
            del args, expected, output

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
        stability = timing_report(
            repeats,
            parent_wall_ms,
            float(timing["max_min_ratio"]),
            int(timing["symmetric_trim_each_side"]),
        )
        all_correct = bool(timer_protocol_ok and checks and all(checks) and stability["passed"])
        return {
            "all_correct": all_correct,
            "timer_protocol_ok": timer_protocol_ok,
            "full_correctness_checks": len(checks),
            "correctness_passed_checks": sum(checks),
            "median_cuda_ms": float(stability["median_ms"]),
            "repeat_cuda_ms": repeats,
            "full_max_min_ratio": float(stability["full_max_min_ratio"]),
            "trimmed_max_min_ratio": float(stability["trimmed_max_min_ratio"]),
            "max_min_ratio": float(stability["trimmed_max_min_ratio"]),
            "stability": stability["stability"],
            "robust_interval": stability["robust_interval"],
            "outlier_rule": stability["outlier_rule"],
            "parent_wall_time_ms": parent_wall_ms,
            "memory": memory,
        }
    finally:
        temporary.cleanup()
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure one A6000 anchor reproduction.")
    parser.add_argument("--panel", required=True)
    parser.add_argument("--baseline-submission", required=True)
    parser.add_argument("--sota-submission", required=True)
    parser.add_argument("--reproduction-id", required=True)
    parser.add_argument(
        "--role-order",
        choices=("baseline-first", "sota-first"),
        default="baseline-first",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for author calibration")
    actual_cpu_affinity = sorted(os.sched_getaffinity(0))
    if actual_cpu_affinity != REQUIRED_CPU_AFFINITY:
        raise RuntimeError(
            f"revision v25 requires sched affinity {REQUIRED_CPU_AFFINITY}, got {actual_cpu_affinity}"
        )
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name != EXPECTED_GPU_NAME:
        raise RuntimeError(f"expected {EXPECTED_GPU_NAME!r}, got {gpu_name!r}")
    installed_commit = os.environ.get("FLASHINFER_SOURCE_COMMIT", "")
    if installed_commit != PINNED_UPSTREAM_COMMIT:
        raise RuntimeError("the calibration image does not attest the pinned upstream commit")
    installed_version = importlib.metadata.version("flashinfer-python")
    if installed_version != PINNED_FLASHINFER_VERSION:
        raise RuntimeError(
            f"expected flashinfer-python {PINNED_FLASHINFER_VERSION}, got {installed_version}"
        )

    panel_path = Path(args.panel).resolve()
    baseline = Path(args.baseline_submission).resolve()
    sota = Path(args.sota_submission).resolve()
    output = Path(args.output)
    refuse_sealed_output(output)
    upstream = json.loads((sota / "UPSTREAM.json").read_text())
    if upstream.get("commit") != PINNED_UPSTREAM_COMMIT:
        raise RuntimeError("SOTA adapter provenance does not match the pinned commit")
    baseline_hash = sha256(baseline / "solver.py")
    sota_hash = sha256(sota / "solver.py")
    if baseline_hash != EXPECTED_BASELINE_SHA256 or sota_hash != EXPECTED_SOTA_SHA256:
        raise RuntimeError("calibration role source hashes do not match the reviewed revision-v25 roles")

    panel = json.loads(panel_path.read_text())
    timing = panel["timing"]
    if (
        panel.get("protocol")
        != "paged_ragged_gqa_decode_v25_d256_extreme_gqa_streamed_exact_snapshot"
        or int(timing["warmup"]) != 128
        or int(timing["repeats"]) != 21
        or int(timing["timed_inner_calls"]) != 8
        or float(timing["max_min_ratio"]) != 1.20
        or int(timing["symmetric_trim_each_side"]) != 4
        or float(timing["bracket_max_min_ratio"]) != 1.20
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
        raise RuntimeError(
            "revision-v25 calibration requires 128 warmups, 21 groups of eight timed calls, "
            "a fixed two-phase balanced 2+2-per-phase robust interval, and a hard 1.20 "
            "within-phase dispersion ceiling"
        )
    cases = {}
    role_paths = {"baseline": baseline, "sota": sota}
    role_order = (
        ("baseline", "sota")
        if args.role_order == "baseline-first"
        else ("sota", "baseline")
    )
    for raw_case in panel["cases"]:
        roles = {
            role: evaluate_role(role_paths[role], raw_case, timing)
            for role in role_order
        }
        if not all(row["all_correct"] for row in roles.values()):
            failure = {
                "case_id": str(raw_case["id"]),
                "role_order": list(role_order),
                "roles": {role: json_safe(row) for role, row in roles.items()},
            }
            raise RuntimeError(
                "calibration role failed correctness or stability: "
                + json.dumps(failure, sort_keys=True, allow_nan=False)
            )
        baseline_ms = roles["baseline"]["median_cuda_ms"]
        sota_ms = roles["sota"]["median_cuda_ms"]
        if not baseline_ms > sota_ms > 0.0:
            raise RuntimeError(f"{raw_case['id']}: empirical order is not baseline > SOTA > 0")
        cases[str(raw_case["id"])] = {"roles": roles}

    properties = torch.cuda.get_device_properties(0)
    record = {
        "schema_version": 1,
        "status": "MEASURED_A6000_REPRODUCTION",
        "task_version": "paged-ragged-gqa-decode-d256-extreme-gqa-v25",
        "reproduction_id": str(args.reproduction_id),
        "panel_sha256": sha256(panel_path),
        "panel_protocol": panel["protocol"],
        "timing": timing,
        "hardware": {
            "name": gpu_name,
            "uuid": str(getattr(properties, "uuid", "unavailable")),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": int(properties.total_memory),
            "cpu_affinity": actual_cpu_affinity,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "flashinfer_python": installed_version,
            "upstream_repository": upstream["repository"],
            "upstream_commit": PINNED_UPSTREAM_COMMIT,
        },
        "submissions": {
            "baseline_sha256": baseline_hash,
            "sota_sha256": sota_hash,
        },
        "protocol": {
            "timer": "21 precreated CUDA event pairs, each over one streamed resident group of eight contiguous calls",
            "fresh_values_per_call": True,
            "fresh_output_storage_per_call": True,
            "caller_owned_output_ring": True,
            "output_ring_preallocated_untimed": True,
            "output_ring_entries": 297,
            "output_ring_all_storages_unique": True,
            "solver_output_copy_into_ring_timed": True,
            "submission_temporary_output_retained": False,
            "timed_region_definition": "submission solver call plus copy into the next untimed-preallocated unique caller-owned ring target",
            "streamed_fresh_input_groups": True,
            "resident_calls_per_group": 8,
            "cpu_exact_mutation_snapshots": True,
            "cuda_snapshot_bytes": 0,
            "input_setup_inside_timing": False,
            "validation_inside_timing": False,
            "contiguous_calls_inside_each_timed_event": True,
            "synchronize_after_each_timed_event": True,
            "warmup_batch_size": 128,
            "warmup_group_count": 16,
            "fixed_steady_state_warmup": True,
            "adaptive_warmup_or_posthoc_selection": False,
            "timed_event_count": 21,
            "timed_calls_per_event": 8,
            "timed_call_count": 168,
            "event_latency_divided_by_timed_calls": True,
            "clock_primer_rounds": 64,
            "clock_primer_immediately_before_warmup": True,
            "latency_estimator": "ordinary median of 21 retained CUDA-event group means, each covering eight fresh calls",
            "stability_estimator": "alternating event-parity center 13 after excluding two low and two high samples within each phase",
            "symmetric_trim_each_side": 4,
            "stability_retained_count": 13,
            "stability_phase_period": 2,
            "stability_phase_trim_each_side": 2,
            "dispersion_hard_gate": True,
            "stability_max_min_ratio_ceiling": 1.20,
            "outlier_rule": "two low and two high samples per fixed event-parity phase affect only stability; all 21 samples remain in the scoring median",
            "raw_metric_uses_all_repeats": True,
            "required_cpu_affinity": REQUIRED_CPU_AFFINITY,
            "sched_affinity_verified": True,
            "stable_page_table_per_case": True,
            "correctness": "independent eager FP32-softmax reference on every call",
            "raw_metric": "fresh-process median CUDA milliseconds; lower is better",
            "role_order": list(role_order),
        },
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "panel_sha256": record["panel_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
