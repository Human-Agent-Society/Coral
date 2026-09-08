from __future__ import annotations

import gc
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/runner")
import protocol


_EVENT = torch.cuda.Event
_SYNC = torch.cuda.synchronize
_NP_SAVE = np.save


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


def load_solver(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_solver", path / "solver.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, protocol.ENTRYPOINT)


def save_values(values, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, value in enumerate(values):
        path = out_dir / f"value_{index}.npy"
        _NP_SAVE(path, value.detach().float().cpu().numpy(), allow_pickle=False)
        paths.append(str(path))
    return paths


def one_call(fn, case, seed):
    args = protocol.make_inputs(case, seed)
    snapshots = cpu_exact_snapshot(args)
    values = protocol.run_candidate(fn, args, case)
    return args, values, cpu_exact_immutability(args, snapshots), protocol.structure(values, args, case)


def main():
    submission, challenge_path, output_path = map(Path, sys.argv[1:4])
    challenge = json.loads(challenge_path.read_text())
    case = challenge["case"]
    challenge_seed = int(challenge["challenge_seed"])
    fn = load_solver(submission)

    args, values, immutable, structure = one_call(fn, case, protocol.phase_seed(case, "correctness", 0))
    correctness_paths = save_values(values, output_path.parent / "correctness")
    correctness_signature = protocol.signature(values, challenge_seed ^ 0xC011EC7)
    del args, values
    gc.collect()
    torch.cuda.empty_cache()

    calls = []
    for index in range(protocol.WARMUPS):
        seed = protocol.phase_seed(case, "warmup", index)
        args, values, imm, struct = one_call(fn, case, seed)
        _SYNC()
        calls.append({
            "phase": "warmup", "index": index, "seed": seed,
            "signature": protocol.signature(values, challenge_seed + 10_000 + index),
            "immutability": imm, "structure": struct,
        })
        del args, values
        gc.collect()

    times = []
    for index in range(protocol.REPEATS):
        seed = protocol.phase_seed(case, "timed", index)
        args = protocol.make_inputs(case, seed)
        snapshots = cpu_exact_snapshot(args)
        start, end = _EVENT(enable_timing=True), _EVENT(enable_timing=True)
        start.record()
        values = protocol.run_candidate(fn, args, case)
        end.record()
        end.synchronize()
        elapsed = float(start.elapsed_time(end))
        times.append(elapsed)
        calls.append({
            "phase": "timed", "index": index, "seed": seed, "elapsed_ms": elapsed,
            "signature": protocol.signature(values, challenge_seed + 20_000 + index),
            "immutability": cpu_exact_immutability(args, snapshots),
            "structure": protocol.structure(values, args, case),
        })
        del args, values
        gc.collect()

    ordered = sorted(times)
    trim = (len(ordered) - protocol.DISPERSION_CENTER_COUNT) // 2
    center = ordered[trim:len(ordered) - trim]
    output_path.write_text(json.dumps({
        "case_id": case["id"],
        "timer": {"prebound": True, "fresh_inputs": True, "setup_outside_timing": True},
        "correctness": {
            "paths": correctness_paths,
            "signature": correctness_signature,
            "immutability": immutable,
            "structure": structure,
        },
        "calls": calls,
        "candidate_repeat_ms": times,
        "candidate_median_ms": statistics.median(times),
        "candidate_center_samples_ms": center,
        "candidate_center_dispersion": max(center) / max(min(center), 1e-9),
    }, indent=2))


if __name__ == "__main__":
    main()
