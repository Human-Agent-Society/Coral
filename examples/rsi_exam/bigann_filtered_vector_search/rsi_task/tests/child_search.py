"""CHILD (untrusted side): imports the submitted Solver, builds it on the public library, then --
only after signalling BUILD_DONE and receiving a query-bundle path over stdin -- answers the whole
sealed workload in ONE batch call. By construction the hidden queries do not exist on disk (and are
never read by this process) until the parent hands over their path post-build.

Usage: child_search.py <submission_dir> <base_data_dir>
Protocol:
  stdout: "BUILD_DONE\n" once Solver.build() returns. The parent narrows this process tree to a
          single CPU at that moment, so build() may use every core but search may not.
  stdin:  one JSON line {"query_path": ..., "out_path": ...}. query_path is an .npz holding
          "vectors" uint8 [n, 192] and the queries' own tag CSR, "tag_indptr" int64 [n+1] and
          "tag_indices" int32.
  stdout: "SEARCH_DONE\n" once the results are on disk.

Search is a BATCH call, the way the Big-ANN filtered track measures throughput: the whole workload
is handed over at once and the wall-clock of that single call is the timed quantity. What a solver
does inside -- one query at a time, grouped by predicate, distances vectorised across the batch,
reordered for locality -- is part of the problem.

The library is memory-mapped, never materialised: 10M x 192 uint8 is 1.9 GB as bytes but 7.7 GB if
cast to float32, so the contract hands the solver the raw uint8 view and lets it decide what to
widen and when. Tags are CSR for the same reason -- 108M tag entries as 10M Python frozensets
would cost more than the vectors do.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def read_u8bin(path):
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype="int32", count=2)
    return np.memmap(path, dtype="uint8", mode="r", offset=8, shape=(int(n), int(d)))


def read_spmat(path):
    with open(path, "rb") as f:
        nrow, ncol, nnz = np.fromfile(f, dtype="int64", count=3)
    nrow, ncol, nnz = int(nrow), int(ncol), int(nnz)
    indptr = np.memmap(path, dtype="int64", mode="r", offset=24, shape=(nrow + 1,))
    indices = np.memmap(path, dtype="int32", mode="r", offset=24 + 8 * (nrow + 1), shape=(nnz,))
    return indptr, indices, ncol


def main():
    submission_dir, base_dir = sys.argv[1], sys.argv[2]
    sys.path.insert(0, submission_dir)
    import solver  # the submission's Solver class

    base = Path(base_dir)
    base_vectors = read_u8bin(base / "base.10M.u8bin")
    tag_indptr, tag_indices, _ = read_spmat(base / "base.metadata.10M.spmat")

    s = solver.Solver()
    s.build(base_vectors, tag_indptr, tag_indices)
    print("BUILD_DONE", flush=True)

    msg = json.loads(sys.stdin.readline())
    bundle = np.load(msg["query_path"])
    n = len(bundle["vectors"])
    out = s.search_batch(bundle["vectors"], bundle["tag_indptr"], bundle["tag_indices"], k=10)

    out = np.asarray(out, dtype=np.int64)
    if out.ndim != 2 or out.shape[0] != n:
        raise ValueError(f"search_batch must return [n_queries, k]; got shape {out.shape}")
    Path(msg["out_path"]).write_text(json.dumps(out.tolist()))
    print("SEARCH_DONE", flush=True)


if __name__ == "__main__":
    main()
