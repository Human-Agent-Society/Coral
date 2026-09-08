from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
BUILDER = HERE / "build_anchor_manifest.py"
PANEL = HERE / "calibration" / "test_spec.json"


def flat_stability(value: float, count: int) -> dict:
    return {
        "estimator": "alternating_phase_balanced_symmetric_sorted_trim",
        "sample_count": count,
        "trim_each_side": 4,
        "phase_period": 2,
        "phase_trim_each_side": 2,
        "retained_count": count - 8,
        "worst_phase": 1,
        "phases": {
            "0": {
                "sample_count": 11,
                "trim_each_side": 2,
                "retained_count": 7,
                "trimmed_low_indices": [0, 2],
                "trimmed_high_indices": [18, 20],
                "retained_indices": [4, 6, 8, 10, 12, 14, 16],
                "retained_min_index": 4,
                "retained_min_cuda_ms": value,
                "retained_max_index": 16,
                "retained_max_cuda_ms": value,
                "trimmed_max_min_ratio": 1.0,
            },
            "1": {
                "sample_count": 10,
                "trim_each_side": 2,
                "retained_count": 6,
                "trimmed_low_indices": [1, 3],
                "trimmed_high_indices": [17, 19],
                "retained_indices": [5, 7, 9, 11, 13, 15],
                "retained_min_index": 5,
                "retained_min_cuda_ms": value,
                "retained_max_index": 15,
                "retained_max_cuda_ms": value,
                "trimmed_max_min_ratio": 1.0,
            },
        },
        "trimmed_low_indices": [0, 1, 2, 3],
        "trimmed_high_indices": [17, 18, 19, 20],
        "retained_indices": list(range(4, count - 4)),
        "full_min_index": 0,
        "full_min_cuda_ms": value,
        "full_max_index": count - 1,
        "full_max_cuda_ms": value,
        "retained_min_index": 5,
        "retained_min_cuda_ms": value,
        "retained_max_index": 15,
        "retained_max_cuda_ms": value,
        "full_max_min_ratio": 1.0,
        "trimmed_max_min_ratio": 1.0,
    }


def measurement(panel: dict, panel_hash: str, run: int, order: list[str]) -> dict:
    cases = {}
    for index, case in enumerate(panel["cases"], start=1):
        factor = (1.0, 1.01, 0.99)[run]
        baseline = (10.0 + index) * factor
        sota = (4.0 + index / 10.0) * factor
        cases[case["id"]] = {
            "roles": {
                "baseline": {
                    "all_correct": True,
                    "median_cuda_ms": baseline,
                    "repeat_cuda_ms": [baseline] * panel["timing"]["repeats"],
                    "full_max_min_ratio": 1.0,
                    "trimmed_max_min_ratio": 1.0,
                    "stability": flat_stability(baseline, panel["timing"]["repeats"]),
                },
                "sota": {
                    "all_correct": True,
                    "median_cuda_ms": sota,
                    "repeat_cuda_ms": [sota] * panel["timing"]["repeats"],
                    "full_max_min_ratio": 1.0,
                    "trimmed_max_min_ratio": 1.0,
                    "stability": flat_stability(sota, panel["timing"]["repeats"]),
                },
            }
        }
    return {
        "schema_version": 1,
        "status": "MEASURED_A6000_REPRODUCTION",
        "task_version": "paged-ragged-gqa-decode-d256-extreme-gqa-v25",
        "reproduction_id": f"synthetic-static-test-r{run + 1}",
        "panel_sha256": panel_hash,
        "panel_protocol": panel["protocol"],
        "timing": panel["timing"],
        "hardware": {
            "name": "NVIDIA RTX A6000",
            "uuid": "synthetic-test-only",
            "cpu_affinity": [4],
        },
        "software": {
            "upstream_commit": "6104afe3777bcfb16a71fa28eb73b51b7e5e5332",
            "flashinfer_python": "0.6.15",
        },
        "submissions": {
            "baseline_sha256": "25d150e3f6c8a14faea71dbb6a9bda45e7dbf54cd778297bb2fdf0c426f351c3",
            "sota_sha256": "e1119584a208d2e493314c12c627366b9469813052f931b34dcf32d0855fe408",
        },
        "protocol": {
            "role_order": order,
            "fresh_values_per_call": True,
            "fresh_output_storage_per_call": True,
            "caller_owned_output_ring": True,
            "output_ring_preallocated_untimed": True,
            "output_ring_entries": 297,
            "output_ring_all_storages_unique": True,
            "solver_output_copy_into_ring_timed": True,
            "submission_temporary_output_retained": False,
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
            "required_cpu_affinity": [4],
            "sched_affinity_verified": True,
        },
        "cases": cases,
    }


def main() -> None:
    import hashlib

    panel = json.loads(PANEL.read_text())
    panel_hash = hashlib.sha256(PANEL.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = []
        orders = [["baseline", "sota"], ["sota", "baseline"], ["baseline", "sota"]]
        for run, order in enumerate(orders):
            path = root / f"r{run + 1}.json"
            path.write_text(json.dumps(measurement(panel, panel_hash, run, order)))
            paths.append(path)
        output = root / "candidate.json"
        command = [sys.executable, str(BUILDER), "--panel", str(PANEL)]
        for path in paths:
            command.extend(["--measurement", str(path)])
        command.extend(["--output", str(output)])
        subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
        candidate = json.loads(output.read_text())
        assert candidate["status"] == "MEASURED_A6000_HUMAN_SOTA_SUMMARY_NOT_PROMOTED"
        assert candidate["global_summary"]["human_sota_geomean_raw_speedup"] > 1.0
        assert all("upper_bound" not in row for row in candidate["cases"].values())
        assert subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode != 0

        sealed = root / "tests" / "heldout" / "anchors.json"
        sealed.parent.mkdir(parents=True)
        sealed_command = command[:-1] + [str(sealed)]
        assert subprocess.run(sealed_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode != 0
    print("test_calibration_builder: PASS")


if __name__ == "__main__":
    main()
