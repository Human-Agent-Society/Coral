from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path("/app")
TASK_VERSION = "flashattention-noncausal-global-d160-extreme-gqa-full-vjp-v21"
CALIBRATION_STATUS = "pending_h100_remeasure"


def cpu_exact_snapshot(args):
    """Keep the exact immutability oracle off the capacity-critical GPU."""
    return tuple(value.detach().to(device="cpu", copy=True) for value in args)


def cpu_exact_immutability(args, snapshots):
    changed = []
    for index, (actual, saved) in enumerate(zip(args, snapshots)):
        current = actual.detach().to(device="cpu", copy=True)
        if not current.equal(saved):
            changed.append(index)
        del current
    return {"passed": not changed, "changed_indices": changed}


def load():
    import protocol

    path = ROOT / "methods" / "main" / "solver.py"
    spec = importlib.util.spec_from_file_location("candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, protocol.ENTRYPOINT)


def arrays(values):
    return [value.detach().float().cpu().numpy() for value in values]


def challenge_seed(case_id: str, panel_sha: str) -> int:
    payload = f"public-selfcheck|{panel_sha}|{case_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFF_FFFF


def reference_arrays(case, seed):
    import torch
    import protocol

    args = protocol.make_inputs(case, seed)
    values = protocol.run_reference(args, case)
    torch.cuda.synchronize()
    result = arrays(values)
    del args, values
    gc.collect()
    torch.cuda.empty_cache()
    return result


def reference_signature(case, seed, signature_seed):
    import torch
    import protocol

    args = protocol.make_inputs(case, seed)
    values = protocol.run_reference(args, case)
    torch.cuda.synchronize()
    result = protocol.signature(values, signature_seed)
    del args, values
    gc.collect()
    torch.cuda.empty_cache()
    return result


def candidate_call(fn, case, seed, signature_seed, *, timed):
    import torch
    import protocol

    args = protocol.make_inputs(case, seed)
    snapshots = cpu_exact_snapshot(args)
    if timed:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        values = protocol.run_candidate(fn, args, case)
        end.record()
        end.synchronize()
        elapsed_ms = float(start.elapsed_time(end))
    else:
        values = protocol.run_candidate(fn, args, case)
        torch.cuda.synchronize()
        elapsed_ms = None
    result = {
        "elapsed_ms": elapsed_ms,
        "signature": protocol.signature(values, signature_seed),
        "immutability": cpu_exact_immutability(args, snapshots),
        "structure": protocol.structure(values, args, case),
    }
    del args, snapshots, values
    gc.collect()
    torch.cuda.empty_cache()
    return result


def worker(case_id: str) -> None:
    """Evaluate one public case in one fresh CUDA child process.

    The candidate remains loaded across one full-array call, five warm-up
    seeds, and twenty-one independently generated timed seeds.  This is the
    same multi-call state and fresh-input contract used by the sealed verifier.
    """

    import torch
    import protocol

    fn = load()
    panel_path = ROOT / "problems" / "visible_cases.json"
    panel_bytes = panel_path.read_bytes()
    panel_sha = hashlib.sha256(panel_bytes).hexdigest()
    panel = json.loads(panel_bytes)
    baseline_payload = json.loads(
        (ROOT / "problems" / "visible_baseline_manifest.json").read_text()
    )
    if baseline_payload.get("status") != CALIBRATION_STATUS:
        raise RuntimeError("v21 visible baseline is not freshly calibrated")
    if baseline_payload.get("task_version") != TASK_VERSION:
        raise RuntimeError("v21 visible baseline task-version mismatch")
    if baseline_payload.get("panel_sha256") != panel_sha:
        raise RuntimeError("v21 visible baseline does not bind public panel")
    baseline = baseline_payload["cases"]
    case = next(
        (item for item in panel["cases"] if item["id"] == case_id),
        None,
    )
    if case is None:
        raise RuntimeError(f"unknown visible case: {case_id}")
    cseed = challenge_seed(case_id, panel_sha)

    correctness_seed = protocol.phase_seed(case, "correctness", 0)
    candidate_inputs = protocol.make_inputs(case, correctness_seed)
    snapshots = cpu_exact_snapshot(candidate_inputs)
    actual = protocol.run_candidate(fn, candidate_inputs, case)
    torch.cuda.synchronize()
    actual_arrays = arrays(actual)
    immutable = cpu_exact_immutability(candidate_inputs, snapshots)
    structured = protocol.structure(actual, candidate_inputs, case)
    del candidate_inputs, snapshots, actual
    gc.collect()
    torch.cuda.empty_cache()
    comparisons = protocol.compare(
        actual_arrays,
        reference_arrays(case, correctness_seed),
        case,
    )
    del actual_arrays
    if not (
        immutable["passed"]
        and structured["passed"]
        and len(comparisons) == 4
        and all(row["passed"] for row in comparisons)
    ):
        print(json.dumps({
            "id": case_id,
            "passed": False,
            "stage": "full_array_correctness",
            "immutability": immutable,
            "structure": structured,
            "comparisons": comparisons,
        }), flush=True)
        raise SystemExit(1)

    calls = []
    for phase, count, offset in (
        ("warmup", protocol.WARMUPS, 10_000),
        ("timed", protocol.REPEATS, 20_000),
    ):
        for index in range(count):
            seed = protocol.phase_seed(case, phase, index)
            signature_seed = cseed + offset + index
            row = candidate_call(
                fn,
                case,
                seed,
                signature_seed,
                timed=(phase == "timed"),
            )
            expected = reference_signature(case, seed, signature_seed)
            passed = (
                row["immutability"]["passed"]
                and row["structure"]["passed"]
                and protocol.signature_close(row["signature"], expected)
            )
            calls.append({
                "phase": phase,
                "index": index,
                "seed": seed,
                "passed": bool(passed),
                "elapsed_ms": row["elapsed_ms"],
            })
            if not passed:
                print(json.dumps({
                    "id": case_id,
                    "passed": False,
                    "stage": "fresh_multiseed_calls",
                    "failed_call": calls[-1],
                    "worker_pid": os.getpid(),
                }), flush=True)
                raise SystemExit(1)

    samples = [
        float(row["elapsed_ms"])
        for row in calls
        if row["phase"] == "timed"
    ]
    ordered = sorted(samples)
    trim = (len(ordered) - protocol.DISPERSION_CENTER_COUNT) // 2
    center = ordered[trim:len(ordered) - trim]
    dispersion = max(center) / max(min(center), 1e-9)
    if (
        len(samples) != protocol.REPEATS
        or len(center) != protocol.DISPERSION_CENTER_COUNT
        or dispersion > protocol.MAX_DISPERSION
    ):
        raise RuntimeError(
            f"{case_id}: invalid timing vector or dispersion {dispersion}"
        )
    candidate_ms = statistics.median(samples)
    raw_speedup = float(baseline[case_id]["baseline_ms"]) / candidate_ms
    print(json.dumps({
        "id": case_id,
        "passed": True,
        "candidate_ms": candidate_ms,
        "raw_speedup": raw_speedup,
        "center_dispersion": dispersion,
        "full_array_correctness_calls": 1,
        "fresh_signature_calls": len(calls),
        "worker_pid": os.getpid(),
        "process_isolation": "fresh_child_per_case",
    }), flush=True)


def invoke(case_id: str) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--case-id",
            case_id,
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, end="", flush=True)
        raise RuntimeError(
            f"isolated visible worker failed ({completed.returncode}): {case_id}"
        )
    if not lines:
        raise RuntimeError(f"isolated visible worker emitted no result: {case_id}")
    result = json.loads(lines[-1])
    if (
        result.get("id") != case_id
        or result.get("passed") is not True
        or result.get("fresh_signature_calls") != 26
    ):
        raise RuntimeError(f"isolated visible worker did not pass: {case_id}")
    print(json.dumps(result), flush=True)
    return result


def orchestrate() -> None:
    panel = json.loads((ROOT / "problems" / "visible_cases.json").read_text())
    results = [invoke(case["id"]) for case in panel["cases"]]
    worker_pids = [int(result["worker_pid"]) for result in results]
    if len(set(worker_pids)) != len(worker_pids):
        raise RuntimeError("visible selfcheck workers did not use distinct processes")
    scores = [float(result["raw_speedup"]) for result in results]
    print(json.dumps({
        "all_cases_passed": True,
        "raw_speedup": math.exp(sum(math.log(x) for x in scores) / len(scores)),
        "normalization": None,
        "correctness_contract": "one full-array plus 5 warmup and 21 timed fresh seeds per case",
        "worker_count": len(worker_pids),
        "all_workers_have_distinct_pids": True,
    }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case-id")
    args = parser.parse_args()
    if args.worker:
        if not args.case_id:
            raise RuntimeError("--worker requires --case-id")
        worker(args.case_id)
    elif args.case_id is not None:
        raise RuntimeError("--case-id is only valid with --worker")
    else:
        orchestrate()
