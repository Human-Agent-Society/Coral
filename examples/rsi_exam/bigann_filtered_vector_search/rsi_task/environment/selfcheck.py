"""Honest self-evaluation on the visible query workload.

Reproduces the grader's protocol exactly:

  * build() runs with every core the container has, and is untimed;
  * every thread of this process is then pinned to ONE CPU;
  * search_batch() is called once with all 500 visible queries and that call is what gets timed;
  * recall@10 is recomputed against the shipped visible ground truth.

Prints the raw metrics only: recall@10 and QPS = n_queries / batch_seconds.

    python /app/selfcheck.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP / "methods" / "main"))
DATA = APP / "data"
RECALL_TARGET = 0.90
K = 10

from ann_utils import read_spmat, read_u8bin  # noqa: E402


def pin_self_to_one_core() -> int:
    """Narrow every thread of this process to a single CPU, the way the grader does to its child.

    Setting affinity on the process only moves the calling thread; library thread pools already
    running would keep the wide mask. Threads created afterwards inherit from their creator, so
    walking /proc/self/task once the pools exist is what actually contains everything.
    """
    cpu = sorted(os.sched_getaffinity(0))[0]
    pinned = 0
    for _ in range(3):
        for tid in os.listdir("/proc/self/task"):
            try:
                os.sched_setaffinity(int(tid), {cpu})
                pinned += 1
            except (ProcessLookupError, PermissionError, OSError):
                pass
    return pinned


def main() -> None:
    base_vectors = read_u8bin(DATA / "base.10M.u8bin")
    tag_indptr, tag_indices, _ = read_spmat(DATA / "base.metadata.10M.spmat")

    qv = np.load(DATA / "query_vectors.npy")
    qip = np.load(DATA / "query_tag_indptr.npy")
    qix = np.load(DATA / "query_tag_indices.npy")
    gt = np.load(DATA / "ground_truth.npy")

    import solver  # /app/methods/main/solver.py

    s = solver.Solver()
    t0 = time.time()
    s.build(base_vectors, tag_indptr, tag_indices)          # untimed, all cores
    build_seconds = time.time() - t0

    pin_self_to_one_core()                                   # timed section: one core

    n = len(qv)
    t0 = time.time()
    results = s.search_batch(qv, qip, qix, k=K)
    search_seconds = time.time() - t0

    results = np.asarray(results, dtype=np.int64)
    if results.ndim != 2 or results.shape[0] != n:
        print(f"[selfcheck] ERROR: search_batch must return [n_queries, k]; got {results.shape}")
        return

    recalls = []
    for i in range(n):
        truth = set(int(x) for x in gt[i] if x >= 0)
        if not truth:
            recalls.append(1.0)
            continue
        got = set(int(x) for x in results[i][:K])
        recalls.append(len(got & truth) / min(K, len(truth)))
    mean_recall = float(np.mean(recalls))
    qps = n / max(search_seconds, 1e-9)

    print(f"[selfcheck] build_seconds={build_seconds:.2f}  batch_seconds={search_seconds:.4f}  "
          f"n_queries={n}")
    print(f"[selfcheck] recall@10 = {mean_recall:.4f}  (gate: >= {RECALL_TARGET})")
    print(f"[selfcheck] QPS = {qps:.4f}")
    if mean_recall < RECALL_TARGET:
        print("[selfcheck] WARNING: below the recall gate -- grading would score 0.")

    # Budget reminder (docs/budget-hookup.md). Silent unless the overlay mounted it,
    # so a standalone run of this task is unaffected.
    if Path("/app/budget.py").exists():
        import subprocess
        subprocess.run([sys.executable, "/app/budget.py"], check=False)


if __name__ == "__main__":
    main()
