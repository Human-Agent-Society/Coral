"""TWO-PHASE Harbor verifier (PARENT / trusted side) for bigann_filtered_vector_search.

The PARENT (this file, which never imports the submission) loads the sealed ground truth into
memory, MOVES the hidden query files out of the heldout directory into a parent-private temp dir,
and deletes the ground-truth file from disk -- all BEFORE spawning the child. The child
(child_search.py, untrusted) imports the submission, builds it on the public library, and signals
"BUILD_DONE"; only then does the parent write the query bundle into its private dir and hand the
child that path over stdin. The parent times the search loop from the moment it sends the path to
the moment the child exits, recomputes recall@10 against the in-memory truth, and maps QPS through
the 4-anchor band declared in `[verifier.env]`.

Search is ONE batch call over the whole sealed workload, the way the Big-ANN filtered track measures
throughput. build() runs with every core available; the timed search is pinned to a single CPU.

Reward band (repo-wide placement, see skills/task-review-checklist E-0):
    BASELINE -> 0.0   also the gate: at or below it, reward 0
    SOTA -> 0.6
    UPPER -> 1.0
There is deliberately NO reference anchor. BASELINE is the starter the agent is handed
(environment/methods/main/solver.py, a textbook IVF post-filter) -- not a trivial floor. Reward
therefore measures exactly what the agent ADDS to its starting point: an untouched submission
scores 0.0. Pinning baseline lower (say at brute force) and shipping the IVF starter would hand
out ~0.3 for doing nothing, which is what this task did until 2026-08-05.

Both segments work in log(QPS), not QPS (a raw scale with a large dynamic range gets
log-then-linear). QPS spans 185 -> 130,000 here, close to three orders of magnitude.

t = log-progress from baseline to sota, reward = 0.6 * t**gamma. gamma > 1 makes the band
DELIBERATELY non-uniform, and that is a deliberate override of a uniform band. Difficulty on
this task is not uniform in log(QPS): going from 10% to 20% of
the way to sota is a morning's work, going from 90% to 100% means matching the open-source world
champion. A uniform band prices those the same; this one does not.

A mean recall@10 below the gate forces reward 0 regardless of throughput -- returning fast garbage
must never score.
"""
from __future__ import annotations

import json
import math
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# HARDCODED ON PURPOSE. Anything the grader reads from the
# environment can be overridden by whoever starts the container, so every constant that guards
# something must not be readable from there. `-e RECALL_TARGET=0` would have switched the recall
# gate off outright, and a fast solver returning garbage would then have scored. Paths are fixed by
# the image layout, not configuration; /logs/verifier is a Harbor convention, not a setting.
HELDOUT = Path("/tests/heldout")
SUBMISSION_DIR = Path("/app/methods/main")
BASE_DATA_DIR = Path("/app/data")
REWARD_DIR = Path("/logs/verifier")
CHILD = Path(__file__).resolve().parent / "child_search.py"

BUILD_TIMEOUT_SEC = 7200.0
SEARCH_TIMEOUT_SEC = 1800.0
RECALL_TARGET = 0.90
K = 10

# The scoring band comes from a sealed file, NOT from the environment.
# [verifier.env] lands in PID 1's environment, which submitted code reads straight out
# of /proc/1/environ -- and this grader hands the child a copy of its own environment, so anchors
# passed that way would have been in the child's os.environ outright. ANCHORS_PATH is hardcoded for
# the same reason RECALL_TARGET is: a configurable path lets the caller supply their own anchors.
# The file is deleted from disk in main() before the child starts; these floats are read
# at import, so scoring is unaffected.
ANCHORS_PATH = Path("/tests/anchors.json")
if not ANCHORS_PATH.exists():                 # ad hoc runs outside the image, from the repo tree
    ANCHORS_PATH = Path(__file__).resolve().parent / "anchors.json"
_ANCHORS = json.loads(ANCHORS_PATH.read_text())
BASELINE = float(_ANCHORS["baseline"])
SOTA = float(_ANCHORS["sota"])
UPPER = float(_ANCHORS["upper"])


def _reward_of(qps: float) -> float:
    """baseline->0, sota->0.6, upper->1; linear in log(QPS) on both segments."""
    if not (0.0 < BASELINE < SOTA < UPPER):
        raise RuntimeError(f"bad anchors: need 0<baseline<sota<upper, "
                           f"got {BASELINE},{SOTA},{UPPER}")
    if not math.isfinite(qps) or qps <= BASELINE:
        return 0.0
    lb, ls, lu, lq = math.log(BASELINE), math.log(SOTA), math.log(UPPER), math.log(qps)
    if qps <= SOTA:
        return 0.6 * (lq - lb) / (ls - lb)
    return 0.6 + 0.4 * (lq - ls) / (lu - ls)


def _read_line_with_deadline(fileobj, deadline):
    """Read one newline-terminated line from a raw pipe fd, honoring a wall-clock deadline."""
    fd = fileobj.fileno()
    os.set_blocking(fd, False)
    buf = b""
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return None
        r, _, _ = select.select([fd], [], [], min(remaining, 1.0))
        if fd in r:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                chunk = b""
            if chunk == b"":
                return buf.decode("utf-8", "replace") if buf else None
            buf += chunk
            if b"\n" in buf:
                line, _, _ = buf.partition(b"\n")
                return line.decode("utf-8", "replace")


def main() -> None:
    REWARD_DIR.mkdir(parents=True, exist_ok=True)
    out = {"metric": None, "reward": 0.0, "correctness": False, "errors": []}
    details = {}
    priv = None
    proc = None
    try:
        # truth into memory NOW, then quarantine the hidden files before untrusted code runs
        gt = np.load(HELDOUT / "ground_truth.npy")

        priv = Path(tempfile.mkdtemp(prefix="bigann_priv_"))
        os.chmod(priv, 0o700)
        for name in ("query_vectors.npy", "query_tag_indptr.npy", "query_tag_indices.npy"):
            shutil.move(str(HELDOUT / name), str(priv / name))
        (HELDOUT / "ground_truth.npy").unlink()
        if ANCHORS_PATH.exists():        # anchors already read at import
            ANCHORS_PATH.unlink()

        stderr_log = priv / "child_stderr.log"
        child_env = dict(os.environ)
        child_env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                          "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
                          "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})

        # TWO PHASES, TWO CPU BUDGETS.
        #
        # build() is untimed, so there is no reason to restrict it -- it gets every core the
        # container has. The timed search is pinned to ONE core, so throughput measures the method
        # rather than how many cores a submission managed to grab.
        #
        # The env vars above cannot enforce that: they only bind libraries that read them (BLAS,
        # OpenMP, numexpr), and this image ships a compiler, so a submission with its own pthreads
        # would ignore them. CPU affinity is the enforceable version, and it has to be applied to
        # every EXISTING thread -- setting it on the process only moves the calling thread. Threads
        # created afterwards inherit the mask of whichever thread spawned them, so once every task
        # in /proc/<pid>/task is narrowed the whole tree is contained.
        pinned_cpu = sorted(os.sched_getaffinity(0))[0]

        proc = subprocess.Popen(
            [sys.executable, str(CHILD), str(SUBMISSION_DIR), str(BASE_DATA_DIR)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(stderr_log, "wb"),
            start_new_session=True, env=child_env,
        )

        def _narrow_child_to_one_core() -> int:
            """Pin every thread of the child to one CPU. Returns how many were pinned."""
            pinned = 0
            for _ in range(3):          # a couple of passes: threads can appear while we iterate
                try:
                    tids = os.listdir(f"/proc/{proc.pid}/task")
                except FileNotFoundError:
                    break
                for tid in tids:
                    try:
                        os.sched_setaffinity(int(tid), {pinned_cpu})
                        pinned += 1
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
            return pinned

        line = _read_line_with_deadline(proc.stdout, time.time() + BUILD_TIMEOUT_SEC)
        if line is None or line.strip() != "BUILD_DONE":
            raise RuntimeError(f"build handshake failed (got {line!r}); stderr tail: "
                               f"{stderr_log.read_text(errors='replace')[-2000:]}")

        # build is over: narrow to a single core before anything timed happens
        n_pinned = _narrow_child_to_one_core()

        # only now does the hidden query data become readable to the child
        bundle_path = priv / "query_bundle.npz"
        np.savez(bundle_path,
                 vectors=np.load(priv / "query_vectors.npy"),
                 tag_indptr=np.load(priv / "query_tag_indptr.npy"),
                 tag_indices=np.load(priv / "query_tag_indices.npy"))
        out_path = priv / "results.json"
        n = len(gt)

        t0 = time.time()
        proc.stdin.write((json.dumps({"query_path": str(bundle_path),
                                      "out_path": str(out_path)}) + "\n").encode("utf-8"))
        proc.stdin.flush()
        proc.stdin.close()
        try:
            proc.wait(timeout=SEARCH_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            raise RuntimeError("search() timed out")
        finally:
            search_seconds = time.time() - t0

        if proc.returncode != 0:
            raise RuntimeError(f"child failed rc={proc.returncode}; stderr tail: "
                               f"{stderr_log.read_text(errors='replace')[-2000:]}")
        if not out_path.exists():
            raise RuntimeError("child produced no results")
        results = json.loads(out_path.read_text())
        if len(results) != n:
            raise ValueError(f"result count mismatch: got {len(results)}, expected {n}")

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

        reward = 0.0 if mean_recall < RECALL_TARGET else _reward_of(qps)
        reward = round(float(max(0.0, reward)), 6)

        out = {"metric": round(qps, 6), "reward": reward, "correctness": True, "errors": [],
               "details": {"n_queries": n, "recall_at_10": round(mean_recall, 6),
                           "search_seconds": round(search_seconds, 6)}}
        details = {"recall_at_10": round(mean_recall, 6), "qps": round(qps, 6),
                   "search_seconds": round(search_seconds, 6), "n_queries": n,
                   "recall_gate": RECALL_TARGET, "pinned_cpu": pinned_cpu, "threads_pinned": n_pinned,
                   "anchors": {"baseline": BASELINE, "sota": SOTA, "upper": UPPER},
                   "band": "log(QPS); baseline(=starter)->0, 0.6*t**gamma to sota->0.6, linear to upper->1.0"}
    except Exception as exc:  # noqa: BLE001 -- any failure -> reward 0
        out = {"metric": None, "reward": 0.0, "correctness": False,
               "errors": [f"{type(exc).__name__}: {exc}"]}
    finally:
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        if priv is not None:
            shutil.rmtree(priv, ignore_errors=True)

    rewards_json = {"reward": float(out["reward"]),
                    "qps": float(out["metric"]) if out["metric"] is not None else 0.0}
    (REWARD_DIR / "reward.txt").write_text(f"{out['reward']}\n", encoding="utf-8")
    (REWARD_DIR / "reward.json").write_text(json.dumps(rewards_json), encoding="utf-8")
    (REWARD_DIR / "score_details.json").write_text(json.dumps(details), encoding="utf-8")
    (REWARD_DIR / "grade_debug.json").write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    main()
