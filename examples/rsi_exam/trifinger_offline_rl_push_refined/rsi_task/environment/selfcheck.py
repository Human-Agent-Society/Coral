"""Free local dry-run: train your methods/main/solver.py (to a scratch model dir), evaluate it on the
VISIBLE seeds via the same run/replay-eligible loop the verifier uses, and print the per-seed returns
and the mean +/- standard error (higher is better).

The 100 visible seeds are scattered draws from the same episode generator the hidden sealed batch was
drawn from (seed k -> sha256("trifinger-visible-v2-{k}") mod 2^31-1), and fixed anchor policies score
the two sets the same within sampling noise — so an HONESTLY-obtained visible mean is a
faithful proxy. It stops being one the moment you tune against it: a mean improved by repeatedly
selecting cutoffs / architectures / hyperparameters on these same 100 episodes is a training score,
and the sealed batch will not reproduce it. You have the simulator and trifinger_score.seed_episode —
for model selection, evaluate candidates on FRESH seeds of your own choosing that you have not tuned
against, and use this fixed pool as an occasional untouched readout.

    python /app/selfcheck.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = os.environ.get("DATA_DIR", "/app/data/trifinger_dataset")

# Visible-only proxy seeds: 100 scattered, generator-exchangeable with the sealed batch (provenance in
# the module docstring). Disjoint from the sealed held-out seeds, which you never see.
VISIBLE_SEEDS = [
    806160365, 621685217, 969179816, 1100699169, 1651121590, 722515255, 1495643789, 520083202,
    108415304, 1322693921, 503908127, 1739069046, 1598758570, 710363184, 949098118, 1713523063,
    1400822240, 2056686026, 953019398, 495656220, 653435759, 147208380, 110939448, 286728236,
    1396827099, 1686492071, 1841236836, 1641125700, 539654816, 367080247, 1536231611, 753297230,
    970990933, 688389061, 1918216804, 1796658458, 540197446, 871217925, 1186639655, 596407766,
    175192822, 185414052, 1529675285, 1917412631, 1540088195, 1694147235, 1258004417, 78785691,
    233687885, 66618633, 1468342693, 211685980, 135060719, 629637373, 1846912683, 1227842989,
    95011745, 1780201283, 1071394076, 660927811, 1863516575, 744914564, 1014641097, 1221384822,
    2051610237, 143092082, 1494839330, 1705379631, 1685351140, 252946268, 486619540, 1283220877,
    1206997975, 950130089, 617137143, 452945993, 865994704, 616941292, 209532507, 409815737,
    236415203, 1213164843, 1606663256, 1655561495, 1952660110, 2127673497, 1912941186, 343563031,
    698711013, 125665373, 1595271941, 1460418756, 671887820, 1448748864, 1706206198, 1177375239,
    1401163397, 211798364, 2099914179, 1553790463,
]

N_WORKERS = max(1, min(4, os.cpu_count() or 1))


def _load_solver():
    spec = importlib.util.spec_from_file_location("solver", HERE / "methods" / "main" / "solver.py")
    solver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver)
    if not hasattr(solver, "train") or not hasattr(solver, "Policy"):
        raise SystemExit("methods/main/solver.py must define train(...) and class Policy(PolicyBase)")
    return solver


def _worker(chunk):
    # one env + one policy per worker; episodes run sequentially inside the worker, matching the
    # verifier's per-episode semantics (seed_episode + env.reset per episode)
    # HARD assignment, not setdefault, and the value is 4 -- the same constant the graded run
    # uses. setdefault was a latent bug: this image already sets these, so setdefault never
    # fired, and the self-check silently ran at a different thread count than grading. The two
    # sides must be bit-isomorphic or the self-check is not a proxy.
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "4"
    sys.path.insert(0, str(HERE))
    import trifinger_score as ts
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    solver = _load_solver()
    env = ts.make_env(data_dir=DATA_DIR)
    policy = solver.Policy(env.action_space, env.observation_space,
                            env.unwrapped.sim_env.episode_length)
    return [(s, ts.run_policy_episode(env, s, policy)["return"]) for s in chunk]


def main() -> None:
    solver = _load_solver()

    with tempfile.TemporaryDirectory(prefix="selfcheck_trifinger_") as tmp:
        print(f"[selfcheck] training on {DATA_DIR} -> {tmp} ...")
        solver.train(DATA_DIR, tmp, device="cpu")
        os.environ["MODEL_DIR"] = tmp

        chunks = [VISIBLE_SEEDS[i::N_WORKERS] for i in range(N_WORKERS)]
        with get_context("spawn").Pool(processes=N_WORKERS) as pool:
            res = pool.map(_worker, chunks)
    flat = dict(t for c in res for t in c)
    returns = [flat[s] for s in VISIBLE_SEEDS]
    for s in VISIBLE_SEEDS:
        print(f"  seed={s}: return={flat[s]:.3f}")

    mean = float(np.mean(returns))
    sem = float(np.std(returns, ddof=1) / len(returns) ** 0.5)
    print(f"\n[selfcheck] visible mean return = {mean:.3f} +/- {sem:.3f} (SEM) over {len(returns)} "
          "episodes (higher is better; same episode distribution as the sealed batch — honest means "
          "transfer, selection-tuned means do not)")

    # Budget reminder. /app/budget.py is mounted by the harness; when it is not mounted the
    # whole block is skipped, so running this task standalone is unaffected. check=False so a
    # failing budget.py can never take selfcheck down with it. Placed at the end of main() so
    # the reminder prints after this run's score.
    import os as _os, subprocess as _sp, sys as _sys
    if _os.path.exists("/app/budget.py"):
        _sys.stdout.flush()          # without this the reminder prints before the score in a pipe
        _sp.run([_sys.executable, "/app/budget.py"], check=False)


if __name__ == "__main__":
    main()
