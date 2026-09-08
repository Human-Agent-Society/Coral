from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
import statistics
from pathlib import Path


PINNED_UPSTREAM_COMMIT = "6104afe3777bcfb16a71fa28eb73b51b7e5e5332"
PINNED_FLASHINFER_VERSION = "0.6.15"
EXPECTED_SUBMISSIONS = {
    "baseline_sha256": "25d150e3f6c8a14faea71dbb6a9bda45e7dbf54cd778297bb2fdf0c426f351c3",
    "sota_sha256": "e1119584a208d2e493314c12c627366b9469813052f931b34dcf32d0855fe408",
}
EXPECTED_GPU_NAME = os.environ.get("ARB_TARGET_GPU", "NVIDIA RTX A6000")
REQUIRED_CPU_AFFINITY = [4]
QUALIFIED_SOURCE_STATUS = "MEASURED_A6000_REPRODUCTION"
CANDIDATE_STATUS = "MEASURED_A6000_HUMAN_SOTA_SUMMARY_NOT_PROMOTED"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def positive_finite(value: object, label: str) -> float:
    number = float(value)
    require(math.isfinite(number) and number > 0.0, f"{label} must be positive and finite")
    return number


def repeat_stability_report(samples: list[float], trim_each_side: int) -> dict:
    require(trim_each_side >= 0 and trim_each_side % 2 == 0, "phase-balanced trim must be even")
    phase_trim_each_side = trim_each_side // 2
    require(
        all(len(samples[phase::2]) > 2 * phase_trim_each_side for phase in range(2)),
        "phase-balanced stability has too few samples",
    )
    ordered = sorted(enumerate(samples), key=lambda item: (item[1], item[0]))
    full_min, full_max = ordered[0], ordered[-1]
    phases = {}
    retained_all = []
    trimmed_low = []
    trimmed_high = []
    for phase in range(2):
        phase_ordered = sorted(
            ((index, samples[index]) for index in range(phase, len(samples), 2)),
            key=lambda item: (item[1], item[0]),
        )
        retained = phase_ordered[
            phase_trim_each_side : len(phase_ordered) - phase_trim_each_side or None
        ]
        retained_min, retained_max = retained[0], retained[-1]
        low = [index for index, _ in phase_ordered[:phase_trim_each_side]]
        high = (
            [index for index, _ in phase_ordered[len(phase_ordered) - phase_trim_each_side :]]
            if phase_trim_each_side
            else []
        )
        phases[str(phase)] = {
            "sample_count": len(phase_ordered),
            "trim_each_side": phase_trim_each_side,
            "retained_count": len(retained),
            "trimmed_low_indices": low,
            "trimmed_high_indices": high,
            "retained_indices": [index for index, _ in retained],
            "retained_min_index": retained_min[0],
            "retained_min_cuda_ms": retained_min[1],
            "retained_max_index": retained_max[0],
            "retained_max_cuda_ms": retained_max[1],
            "trimmed_max_min_ratio": retained_max[1] / retained_min[1],
        }
        retained_all.extend(retained)
        trimmed_low.extend(low)
        trimmed_high.extend(high)
    worst_phase = max(
        phases,
        key=lambda phase: (phases[phase]["trimmed_max_min_ratio"], int(phase)),
    )
    worst = phases[worst_phase]
    return {
        "estimator": "alternating_phase_balanced_symmetric_sorted_trim",
        "sample_count": len(samples),
        "trim_each_side": trim_each_side,
        "phase_period": 2,
        "phase_trim_each_side": phase_trim_each_side,
        "retained_count": len(retained_all),
        "worst_phase": int(worst_phase),
        "phases": phases,
        "trimmed_low_indices": sorted(trimmed_low),
        "trimmed_high_indices": sorted(trimmed_high),
        "retained_indices": sorted(index for index, _ in retained_all),
        "full_min_index": full_min[0],
        "full_min_cuda_ms": full_min[1],
        "full_max_index": full_max[0],
        "full_max_cuda_ms": full_max[1],
        "retained_min_index": worst["retained_min_index"],
        "retained_min_cuda_ms": worst["retained_min_cuda_ms"],
        "retained_max_index": worst["retained_max_index"],
        "retained_max_cuda_ms": worst["retained_max_cuda_ms"],
        "full_max_min_ratio": full_max[1] / full_min[1],
        "trimmed_max_min_ratio": worst["trimmed_max_min_ratio"],
    }


def refuse_sealed_output(path: Path) -> None:
    resolved = path.resolve()
    parts = set(resolved.parts)
    if resolved.name == "anchors.json" and {"tests", "heldout"}.issubset(parts):
        raise ValueError("this tool never writes the sealed heldout anchor file")
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite {resolved}")


def validate_measurement(
    path: Path,
    record: dict,
    *,
    panel: dict,
    panel_hash: str,
    case_ids: list[str],
) -> None:
    require(record.get("schema_version") == 1, f"{path}: wrong schema")
    require(record.get("status") == QUALIFIED_SOURCE_STATUS, f"{path}: wrong source status")
    require(
        record.get("task_version")
        == "paged-ragged-gqa-decode-d256-extreme-gqa-v25",
        f"{path}: wrong task version",
    )
    require(record.get("panel_sha256") == panel_hash, f"{path}: panel hash mismatch")
    require(record.get("panel_protocol") == panel["protocol"], f"{path}: protocol mismatch")
    require(record.get("timing") == panel["timing"], f"{path}: timing mismatch")
    require(record.get("hardware", {}).get("name") == EXPECTED_GPU_NAME, f"{path}: wrong GPU")
    require(
        record.get("hardware", {}).get("cpu_affinity") == REQUIRED_CPU_AFFINITY,
        f"{path}: wrong CPU affinity",
    )
    gpu_uuid = record.get("hardware", {}).get("uuid")
    require(
        isinstance(gpu_uuid, str) and gpu_uuid not in {"", "unavailable"},
        f"{path}: missing stable GPU UUID",
    )
    require(
        record.get("software", {}).get("upstream_commit") == PINNED_UPSTREAM_COMMIT,
        f"{path}: wrong upstream commit",
    )
    require(
        record.get("software", {}).get("flashinfer_python") == PINNED_FLASHINFER_VERSION,
        f"{path}: wrong flashinfer-python version",
    )
    require(record.get("submissions") == EXPECTED_SUBMISSIONS, f"{path}: role source hash mismatch")
    require(list(record.get("cases", {}).keys()) == case_ids, f"{path}: case order mismatch")
    order = record.get("protocol", {}).get("role_order")
    require(order in (["baseline", "sota"], ["sota", "baseline"]), f"{path}: invalid role order")
    require(
        record.get("protocol", {}).get("fresh_output_storage_per_call") is True,
        f"{path}: fresh-output enforcement is missing",
    )
    require(
        record.get("protocol", {}).get("caller_owned_output_ring") is True
        and record.get("protocol", {}).get("output_ring_preallocated_untimed")
        is True
        and record.get("protocol", {}).get("output_ring_entries") == 297
        and record.get("protocol", {}).get("output_ring_all_storages_unique")
        is True
        and record.get("protocol", {}).get("solver_output_copy_into_ring_timed")
        is True
        and record.get("protocol", {}).get("submission_temporary_output_retained")
        is False,
        f"{path}: caller-owned output-ring enforcement is missing",
    )
    require(
        record.get("protocol", {}).get("fresh_values_per_call") is True,
        f"{path}: fresh-input enforcement is missing",
    )
    require(
        record.get("protocol", {}).get("streamed_fresh_input_groups") is True
        and record.get("protocol", {}).get("resident_calls_per_group") == 8
        and record.get("protocol", {}).get("cpu_exact_mutation_snapshots") is True
        and record.get("protocol", {}).get("cuda_snapshot_bytes") == 0
        and record.get("protocol", {}).get("input_setup_inside_timing") is False
        and record.get("protocol", {}).get("validation_inside_timing") is False
        and record.get("protocol", {}).get("contiguous_calls_inside_each_timed_event") is True
        and record.get("protocol", {}).get("synchronize_after_each_timed_event") is True
        and record.get("protocol", {}).get("warmup_batch_size") == 128
        and record.get("protocol", {}).get("warmup_group_count") == 16
        and record.get("protocol", {}).get("fixed_steady_state_warmup") is True
        and record.get("protocol", {}).get("adaptive_warmup_or_posthoc_selection")
        is False
        and record.get("protocol", {}).get("timed_event_count") == 21
        and record.get("protocol", {}).get("timed_calls_per_event") == 8
        and record.get("protocol", {}).get("timed_call_count") == 168
        and record.get("protocol", {}).get("event_latency_divided_by_timed_calls")
        is True
        and record.get("protocol", {}).get("clock_primer_rounds") == 64
        and record.get("protocol", {}).get("clock_primer_immediately_before_warmup") is True,
        f"{path}: streamed exact-snapshot timing enforcement is missing",
    )
    require(
        record.get("protocol", {}).get("stability_estimator")
        == "alternating event-parity center 13 after excluding two low and two high samples within each phase"
        and record.get("protocol", {}).get("symmetric_trim_each_side") == 4
        and record.get("protocol", {}).get("stability_retained_count") == 13
        and record.get("protocol", {}).get("stability_phase_period") == 2
        and record.get("protocol", {}).get("stability_phase_trim_each_side") == 2
        and record.get("protocol", {}).get("dispersion_hard_gate") is True
        and record.get("protocol", {}).get("stability_max_min_ratio_ceiling")
        == 1.20
        and record.get("protocol", {}).get("raw_metric_uses_all_repeats") is True
        and record.get("protocol", {}).get("required_cpu_affinity") == REQUIRED_CPU_AFFINITY
        and record.get("protocol", {}).get("sched_affinity_verified") is True,
        f"{path}: v25 robust-stability or CPU-affinity enforcement is missing",
    )

    repeats_expected = int(panel["timing"]["repeats"])
    ratio_limit = float(panel["timing"]["max_min_ratio"])
    trim_each_side = int(panel["timing"]["symmetric_trim_each_side"])
    for case_id in case_ids:
        roles = record["cases"][case_id].get("roles", {})
        require(set(roles) == {"baseline", "sota"}, f"{path}: {case_id} role mismatch")
        measured = {}
        for role in ("baseline", "sota"):
            row = roles[role]
            require(row.get("all_correct") is True, f"{path}: {case_id}/{role} is invalid")
            repeats = [positive_finite(x, f"{path}: {case_id}/{role} repeat") for x in row["repeat_cuda_ms"]]
            require(len(repeats) == repeats_expected, f"{path}: {case_id}/{role} repeat count")
            stability = repeat_stability_report(repeats, trim_each_side)
            require(
                stability["trimmed_max_min_ratio"] <= ratio_limit,
                f"{path}: {case_id}/{role} unstable",
            )
            require(
                row.get("stability") == stability,
                f"{path}: {case_id}/{role} stability audit mismatch",
            )
            require(
                math.isclose(
                    float(row.get("full_max_min_ratio")),
                    float(stability["full_max_min_ratio"]),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
                and math.isclose(
                    float(row.get("trimmed_max_min_ratio")),
                    float(stability["trimmed_max_min_ratio"]),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                ),
                f"{path}: {case_id}/{role} stability ratio mismatch",
            )
            median = positive_finite(row["median_cuda_ms"], f"{path}: {case_id}/{role} median")
            require(
                math.isclose(median, statistics.median(repeats), rel_tol=1.0e-9, abs_tol=1.0e-9),
                f"{path}: {case_id}/{role} median mismatch",
            )
            measured[role] = median
        require(measured["baseline"] > measured["sota"], f"{path}: {case_id} anchor order")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate reviewed A6000 reproductions into a non-promoted anchor candidate."
    )
    parser.add_argument("--panel", required=True)
    parser.add_argument("--measurement", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-reproductions", type=int, default=3)
    parser.add_argument("--max-reproduction-ratio", type=float, default=1.15)
    args = parser.parse_args()

    require(args.minimum_reproductions >= 3, "at least three reproductions are mandatory")
    require(args.max_reproduction_ratio >= 1.0, "invalid reproduction ratio")
    output = Path(args.output)
    refuse_sealed_output(output)

    panel_path = Path(args.panel)
    panel = json.loads(panel_path.read_text())
    require(
        panel.get("protocol")
        == "paged_ragged_gqa_decode_v25_d256_extreme_gqa_streamed_exact_snapshot",
        "v25 panel protocol mismatch",
    )
    require(int(panel["timing"]["warmup"]) == 128, "v25 panel must use exactly 128 warmups")
    require(int(panel["timing"]["repeats"]) == 21, "v25 panel must use exactly 21 event groups")
    require(
        int(panel["timing"]["timed_inner_calls"]) == 8,
        "v25 panel must use exactly eight fresh calls per retained event",
    )
    require(
        float(panel["timing"]["max_min_ratio"]) == 1.20,
        "v25 panel must use the 1.20 stability ceiling",
    )
    require(
        int(panel["timing"]["symmetric_trim_each_side"]) == 4,
        "v25 panel must exclude two samples per tail within each fixed event-parity phase",
    )
    require(
        panel["timing"].get("dispersion_hard_gate") is True,
        "v25 panel must enable the dispersion hard gate",
    )
    require(
        panel["timing"].get("streamed_fresh_input_groups") is True
        and panel["timing"].get("cpu_exact_mutation_snapshots") is True
        and int(panel["timing"].get("resident_calls_per_group", -1)) == 8
        and int(panel["timing"].get("max_peak_allocated_bytes", -1)) == 36 * 1024**3
        and int(panel["timing"].get("max_peak_reserved_bytes", -1)) == 40 * 1024**3,
        "v25 panel memory-safety contract mismatch",
    )
    panel_hash = sha256(panel_path)
    case_ids = [str(row["id"]) for row in panel["cases"]]
    measurement_paths = [Path(item) for item in args.measurement]
    require(
        len(measurement_paths) >= args.minimum_reproductions,
        f"need at least {args.minimum_reproductions} independent reproductions",
    )

    records = []
    reproduction_ids = set()
    role_orders = set()
    solver_hashes = None
    gpu_uuid = None
    for path in measurement_paths:
        record = json.loads(path.read_text())
        validate_measurement(path, record, panel=panel, panel_hash=panel_hash, case_ids=case_ids)
        reproduction_id = str(record.get("reproduction_id", ""))
        require(reproduction_id and reproduction_id not in reproduction_ids, f"{path}: duplicate reproduction id")
        reproduction_ids.add(reproduction_id)
        role_orders.add(tuple(record["protocol"]["role_order"]))
        current_hashes = record.get("submissions")
        if solver_hashes is None:
            solver_hashes = current_hashes
        require(current_hashes == solver_hashes, f"{path}: solver changed between reproductions")
        current_uuid = record["hardware"].get("uuid")
        if gpu_uuid is None:
            gpu_uuid = current_uuid
        require(current_uuid == gpu_uuid, f"{path}: GPU changed between reproductions")
        records.append(record)

    require(
        role_orders == {("baseline", "sota"), ("sota", "baseline")},
        "reproductions must include both baseline-first and SOTA-first orders",
    )

    cases = {}
    for case_id in case_ids:
        values = {
            role: [float(record["cases"][case_id]["roles"][role]["median_cuda_ms"]) for record in records]
            for role in ("baseline", "sota")
        }
        for role, samples in values.items():
            require(
                max(samples) / min(samples) <= args.max_reproduction_ratio,
                f"{case_id}/{role}: reproduction drift exceeds limit",
            )
        baseline = statistics.median(values["baseline"])
        sota = statistics.median(values["sota"])
        require(baseline > sota > 0.0, f"{case_id}: aggregate anchor order is invalid")
        repeat_noise = max(
            max(abs(left - right) for left in samples for right in samples)
            / statistics.median(samples)
            for samples in values.values()
        )
        normalized_gap = (baseline - sota) / baseline
        required_margin = max(0.01, 3.0 * repeat_noise)
        require(
            normalized_gap > required_margin,
            f"{case_id}: expert gap {normalized_gap:.6f} does not clear "
            f"max(0.01, 3*repeat_noise)={required_margin:.6f}",
        )
        cases[case_id] = {
            "baseline": baseline,
            "human_sota": sota,
            "raw_speedup": baseline / sota,
            "normalized_baseline_to_sota_gap": normalized_gap,
            "normalized_repeat_noise": repeat_noise,
            "required_margin": required_margin,
            "reproduction_medians": values,
        }

    candidate = {
        "schema_version": 1,
        "status": CANDIDATE_STATUS,
        "task_version": "paged-ragged-gqa-decode-d256-extreme-gqa-v25",
        "panel_sha256": panel_hash,
        "hardware": {"name": EXPECTED_GPU_NAME, "uuid": gpu_uuid},
        "upstream": {
            "repository": "https://github.com/flashinfer-ai/flashinfer",
            "commit": PINNED_UPSTREAM_COMMIT,
        },
        "scoring_contract": "raw speedup only; this calibration summary is not loaded by grade.py",
        "stability": {
            "estimator": "alternating_phase_balanced_symmetric_sorted_trim",
            "sample_count": 21,
            "calls_per_sample": 8,
            "timed_call_count": 168,
            "event_latency_divided_by_calls_per_sample": True,
            "trim_each_side": 4,
            "phase_period": 2,
            "phase_trim_each_side": 2,
            "retained_count": 13,
            "max_min_ratio_ceiling": 1.20,
            "dispersion_hard_gate": True,
            "raw_metric_uses_all_repeats": True,
        },
        "global_summary": {
            "human_sota_geomean_raw_speedup": statistics.geometric_mean(
                row["raw_speedup"] for row in cases.values()
            ),
            "aggregation": "geometric mean of per-case baseline median / Human-SOTA median",
        },
        "cases": cases,
        "evidence": [
            {
                "reproduction_id": record["reproduction_id"],
                "sha256": sha256(path),
                "role_order": record["protocol"]["role_order"],
            }
            for path, record in zip(measurement_paths, records)
        ],
        "promotion": {
            "automatic": False,
            "required_review": "follow solution/calibration_protocol.md and perform an explicit source promotion",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(candidate, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), "status": CANDIDATE_STATUS}, indent=2))


if __name__ == "__main__":
    main()
