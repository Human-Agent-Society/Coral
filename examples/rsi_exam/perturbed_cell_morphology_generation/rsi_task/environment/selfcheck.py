#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

import copy_check

from fid_metric import overall_fid


def load_solver():
    path = Path("/app/methods/main/solver.py")
    if not path.exists():
        path = Path(__file__).parent / "methods" / "main" / "solver.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("submission_solver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_report(generated) -> dict:
    """Same rule the verifier applies: outputs must not duplicate a treated image you were given.

    Control images are excluded on purpose -- echoing your own input is the starter's
    behaviour and already scores zero, so it needs no guardrail.
    """
    pool = []
    with np.load("/app/data/train.npz", allow_pickle=False) as data:
        pool.append(data["treated"])
    with np.load("/app/data/validation_targets.npz", allow_pickle=False) as data:
        pool.append(data["target"])
    violated, details = copy_check.violates(generated, np.concatenate(pool))
    details["copy_violation"] = violated
    details["note"] = (
        "VIOLATION -- the verifier scores this submission 0. Predictions must be generated, "
        "not retrieved from the data you were given."
        if violated else "ok -- outputs are not copies of the data you were given"
    )
    return details

def main() -> None:
    root = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default=str(root / "data" / "validation_inputs.npz"))
    parser.add_argument("--targets", default=str(root / "data" / "validation_targets.npz"))
    parser.add_argument("--checkpoint", default="/app/submission")
    parser.add_argument("--output", default="/tmp/visible_predictions.npz")
    args = parser.parse_args()
    load_solver().predict(args.inputs, args.checkpoint, args.output)
    with np.load(args.output, allow_pickle=False) as prediction_file, np.load(
        args.targets, allow_pickle=False
    ) as target_file, np.load(args.inputs, allow_pickle=False) as input_file:
        if not np.array_equal(prediction_file["sample_id"], input_file["sample_id"]):
            raise ValueError("sample_id order mismatch")
        generated = prediction_file["prediction"].copy()
        fid = overall_fid(target_file["target"], prediction_file["prediction"])
    print(json.dumps({"overall_fid": fid, "lower_is_better": True,
                      "copy_check": copy_report(generated)}, indent=2))
    if Path("/app/budget.py").exists():
        import subprocess
        subprocess.run([sys.executable, "/app/budget.py"], check=False)


if __name__ == "__main__":
    main()
