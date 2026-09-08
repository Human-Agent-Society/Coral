"""SOLVE phase for pbmc_batch_correction (answer-key-isolated grading).

Runs the SUBMITTED embed() on a feature-only boundary the grader wrote (binary ATAC peaks + batch
ids + peak names, NO cell-type labels). The submission gets plain numpy / scipy arrays and the
grader has removed the test file from disk, so it cannot read the `annot` labels the judge scores
against. Writes the cell embedding for the trusted grader.

    python solve.py --boundary <dir> --submission <methods/main dir> --out <embed.npy>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
from scipy import sparse


def _load_solver(submission: str):
    sub = os.path.abspath(submission)
    if sub not in sys.path:
        sys.path.insert(0, sub)
    spec = importlib.util.spec_from_file_location(
        "submission_solver", os.path.join(sub, "solver.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "embed"):
        raise AttributeError("submission's solver.py must define embed(...)")
    return mod.embed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    b = args.boundary
    atac = sparse.load_npz(os.path.join(b, "atac.npz")).tocsr()
    batch_ids = np.load(os.path.join(b, "aux.npz"))["batch_ids"]
    names = json.load(open(os.path.join(b, "names.json")))

    embed = _load_solver(args.submission)
    Z = np.asarray(embed(atac, batch_ids, names["peak_names"]), dtype=np.float64)
    np.save(args.out, Z)


if __name__ == "__main__":
    main()
