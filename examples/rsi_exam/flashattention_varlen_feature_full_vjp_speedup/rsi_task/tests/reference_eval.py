from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/runner")
import protocol


_NP_SAVE = np.save


def save_values(values, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, value in enumerate(values):
        path = out_dir / f"value_{index}.npy"
        _NP_SAVE(path, value.detach().float().cpu().numpy(), allow_pickle=False)
        paths.append(str(path))
    return paths


def reference_call(case, seed, signature_seed):
    args = protocol.make_inputs(case, seed)
    values = protocol.run_reference(args, case)
    signature = protocol.signature(values, signature_seed)
    del args, values
    gc.collect()
    torch.cuda.empty_cache()
    return signature


def main():
    challenge_path, output_path = map(Path, sys.argv[1:3])
    challenge = json.loads(challenge_path.read_text())
    case = challenge["case"]
    challenge_seed = int(challenge["challenge_seed"])

    correctness_seed = protocol.phase_seed(case, "correctness", 0)
    args = protocol.make_inputs(case, correctness_seed)
    values = protocol.run_reference(args, case)
    correctness_paths = save_values(
        values,
        output_path.parent / "reference_correctness",
    )
    del args, values
    gc.collect()
    torch.cuda.empty_cache()

    calls = []
    for phase, count, offset in (
        ("warmup", protocol.WARMUPS, 10_000),
        ("timed", protocol.REPEATS, 20_000),
    ):
        for index in range(count):
            seed = protocol.phase_seed(case, phase, index)
            calls.append({
                "phase": phase,
                "index": index,
                "seed": seed,
                "signature": reference_call(
                    case,
                    seed,
                    challenge_seed + offset + index,
                ),
            })

    output_path.write_text(json.dumps({
        "case_id": case["id"],
        "correctness_seed": correctness_seed,
        "correctness_paths": correctness_paths,
        "calls": calls,
        "fresh_reference_process_per_case": True,
    }, indent=2))


if __name__ == "__main__":
    main()
