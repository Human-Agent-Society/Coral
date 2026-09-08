"""Fresh visible-only pressure using the frozen v25 output-ring evaluator.

This is an author-side pressure harness, not the benchmark reward verifier.  It
binds the strongest captured agent source and measures every visible case as

    immutable baseline before -> captured candidate -> immutable baseline after

Each role is evaluated in a disposable process.  The candidate is never given
the held-out panel, and this module refuses every panel except the published
visible v25 panel.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


BASE = Path("/calibration/author_calibrate_base.py")
WORKER_FLAG = "--bracket-role-worker"
PINNED_UPSTREAM_COMMIT = "6104afe3777bcfb16a71fa28eb73b51b7e5e5332"
PINNED_FLASHINFER_VERSION = "0.6.15"
EXPECTED_GPU_NAME = os.environ.get("ARB_TARGET_GPU", "NVIDIA RTX A6000")
REQUIRED_CPU_AFFINITY = [4]
EXPECTED_VISIBLE_SHA256 = (
    "ff07368f227e7e6cc26a1c816ebd83c8a16b528b9a2cf173d6d0c31d848db087"
)
EXPECTED_BASELINE_SHA256 = (
    "25d150e3f6c8a14faea71dbb6a9bda45e7dbf54cd778297bb2fdf0c426f351c3"
)
EXPECTED_CANDIDATE_SHA256 = (
    "08c107570f1d860114a0fc9e9d6f58a7ecdf431314e6f67c8bc6094860faff5c"
)
EXPECTED_PROTOCOL = (
    "paged_ragged_gqa_decode_v25_d256_extreme_gqa_streamed_exact_snapshot"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_base():
    spec = importlib.util.spec_from_file_location("paged_v25_r12_pressure_base", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen v25 author evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def json_safe(module, value: object) -> object:
    return module.json_safe(value)


def worker_main(arguments: list[str]) -> None:
    if len(arguments) != 3:
        raise RuntimeError("bracket role worker requires payload, output, submission")
    payload_path, output_path, submission = map(Path, arguments)
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    module = load_base()
    result = module.evaluate_role(submission, payload["raw_case"], payload["timing"])
    output_path.write_text(
        json.dumps(json_safe(module, result), separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def isolated_evaluate_role(submission: Path, raw_case: dict, timing: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="paged-v25-r12-pressure-role-") as directory:
        root = Path(directory)
        payload = root / "payload.json"
        output = root / "result.json"
        payload.write_text(
            json.dumps({"raw_case": raw_case, "timing": timing}),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                WORKER_FLAG,
                str(payload),
                str(output),
                str(submission),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3600,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "disposable bracket role worker failed:\n"
                + completed.stdout
                + "\n"
                + completed.stderr
            )
        return json.loads(output.read_text(encoding="utf-8"))


def validate_timing(panel: dict) -> dict:
    timing = panel.get("timing", {})
    if (
        panel.get("protocol") != EXPECTED_PROTOCOL
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
        raise RuntimeError("visible pressure timing contract drift")
    return timing


def geometric_mean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise RuntimeError("cannot aggregate invalid raw speedups")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Paged v25 R12 bracketed visible pressure")
    parser.add_argument("--panel", required=True)
    parser.add_argument("--baseline-submission", required=True)
    parser.add_argument("--sota-submission", required=True)
    parser.add_argument("--reproduction-id", required=True)
    parser.add_argument("--role-order", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    panel_path = Path(args.panel).resolve()
    baseline = Path(args.baseline_submission).resolve()
    candidate = Path(args.sota_submission).resolve()
    output = Path(args.output)
    if panel_path != Path("/panels/visible.json"):
        raise RuntimeError("pressure is sealed to /panels/visible.json only")
    if args.role_order != "baseline-first":
        raise RuntimeError("pressure requires baseline-before/candidate/baseline-after")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite pressure evidence: {output}")
    if sorted(os.sched_getaffinity(0)) != REQUIRED_CPU_AFFINITY:
        raise RuntimeError("pressure requires CPU affinity [4]")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != EXPECTED_GPU_NAME:
        raise RuntimeError("pressure requires one NVIDIA RTX A6000")
    if os.environ.get("FLASHINFER_SOURCE_COMMIT", "") != PINNED_UPSTREAM_COMMIT:
        raise RuntimeError("immutable image upstream commit drift")
    if importlib.metadata.version("flashinfer-python") != PINNED_FLASHINFER_VERSION:
        raise RuntimeError("immutable image FlashInfer version drift")
    if sha256(panel_path) != EXPECTED_VISIBLE_SHA256:
        raise RuntimeError("visible panel hash drift")
    if sha256(baseline / "solver.py") != EXPECTED_BASELINE_SHA256:
        raise RuntimeError("immutable production baseline hash drift")
    if sha256(candidate / "solver.py") != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError("frozen strongest-captured candidate hash drift")
    upstream = json.loads((candidate / "UPSTREAM.json").read_text(encoding="utf-8"))
    if (
        upstream.get("candidate_sha256") != EXPECTED_CANDIDATE_SHA256
        or upstream.get("heldout_allowed") is not False
        or upstream.get("human_sota_claim") is not False
    ):
        raise RuntimeError("candidate pressure provenance drift")

    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    timing = validate_timing(panel)
    cases = {}
    speedups = []
    for raw_case in panel["cases"]:
        baseline_before = isolated_evaluate_role(baseline, raw_case, timing)
        measured_candidate = isolated_evaluate_role(candidate, raw_case, timing)
        baseline_after = isolated_evaluate_role(baseline, raw_case, timing)
        roles = (baseline_before, measured_candidate, baseline_after)
        if not all(role.get("all_correct") is True for role in roles):
            raise RuntimeError(f"{raw_case['id']}: correctness or dispersion failure")
        before_ms = float(baseline_before["median_cuda_ms"])
        candidate_ms = float(measured_candidate["median_cuda_ms"])
        after_ms = float(baseline_after["median_cuda_ms"])
        if not all(math.isfinite(value) and value > 0.0 for value in (before_ms, candidate_ms, after_ms)):
            raise RuntimeError(f"{raw_case['id']}: non-finite latency")
        bracket_drift = max(before_ms, after_ms) / min(before_ms, after_ms)
        if bracket_drift > float(timing["bracket_max_min_ratio"]):
            raise RuntimeError(f"{raw_case['id']}: baseline bracket drift {bracket_drift}")
        baseline_bracket_ms = math.sqrt(before_ms * after_ms)
        raw_speedup = baseline_bracket_ms / candidate_ms
        speedups.append(raw_speedup)
        cases[str(raw_case["id"])] = {
            "measurement_order": ["baseline_before", "candidate", "baseline_after"],
            "baseline_before_ms": before_ms,
            "candidate_ms": candidate_ms,
            "baseline_after_ms": after_ms,
            "baseline_bracket_ms": baseline_bracket_ms,
            "baseline_bracket_drift_ratio": bracket_drift,
            "baseline_bracket_drift_ceiling": float(timing["bracket_max_min_ratio"]),
            "raw_speedup": raw_speedup,
            "roles": {
                "baseline_before": baseline_before,
                "candidate": measured_candidate,
                "baseline_after": baseline_after,
            },
        }

    properties = torch.cuda.get_device_properties(0)
    record = {
        "schema_version": 1,
        "status": "MEASURED_A6000_BRACKETED_VISIBLE_PRESSURE_REPRODUCTION",
        "task_version": "paged-ragged-gqa-decode-d256-extreme-gqa-v25",
        "protocol": EXPECTED_PROTOCOL,
        "reproduction_id": str(args.reproduction_id),
        "panel": "visible",
        "panel_sha256": EXPECTED_VISIBLE_SHA256,
        "heldout_candidate_invocations": 0,
        "heldout_read_or_evaluated": False,
        "candidate_human_sota_claim": False,
        "candidate_sha256": EXPECTED_CANDIDATE_SHA256,
        "baseline_sha256": EXPECTED_BASELINE_SHA256,
        "metric": "geomean_per_case_bracketed_raw_speedup",
        "aggregate_raw_speedup": geometric_mean(speedups),
        "measurement_order": "baseline-before, candidate, baseline-after per case",
        "per_case_speedup": "sqrt(baseline_before_ms * baseline_after_ms) / candidate_ms",
        "timing": timing,
        "hardware": {
            "name": EXPECTED_GPU_NAME,
            "uuid": str(getattr(properties, "uuid", "unavailable")),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": int(properties.total_memory),
            "cpu_affinity": REQUIRED_CPU_AFFINITY,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "flashinfer_python": PINNED_FLASHINFER_VERSION,
            "image_upstream_commit": PINNED_UPSTREAM_COMMIT,
        },
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "aggregate_raw_speedup": record["aggregate_raw_speedup"],
                "heldout_candidate_invocations": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == WORKER_FLAG:
        worker_main(sys.argv[2:])
    else:
        main()
