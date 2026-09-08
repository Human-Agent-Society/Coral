"""AUTHORING-TIME split-legality gate (task-review 2026-07-17 §5.2). NOT part of the runtime
verifier — run it whenever the visible or sealed episode seeds change, before release.

Rationale: the original visible seeds (1001..1032) turned out to be systematically EASIER than the
sealed held-out seeds for trained policies (a fixed policy read ~600 visible vs 173 sealed, ~15
standard errors), so every visible gain the agent optimized for deepened an overfit the sealed set
then punished. The audit's cheapest strong defense: score one or more FIXED anchor policies on both
seed sets and require the means to agree within sampling noise — a broken split fails loudly here.

Usage (inside either task image, both dirs mounted):
    python split_check.py --solver <solver.py> --model-dir <ckpt> \
        --visible-from <selfcheck.py> [--z 2.5] [--workers 8]

Verdict: |mean_visible - mean_sealed| <= z * sqrt(sem_v^2 + sem_h^2)  ->  PASS / FAIL (exit code).
Run it per anchor (baseline / reference at minimum). z default 2.5 (one check per policy; tighten
only with more policies).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_attr(py_path: str, attr: str):
    spec = importlib.util.spec_from_file_location("m_" + attr, py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, attr)


def _run_chunk(args):
    chunk, solver_path, model_dir = args
    os.environ["MODEL_DIR"] = model_dir
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, "/app")
    import trifinger_score as ts
    spec = importlib.util.spec_from_file_location("solver", solver_path)
    solver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver)
    env = ts.make_env(data_dir=os.environ.get("DATA_DIR", "/app/data/trifinger_dataset"))
    policy = solver.Policy(env.action_space, env.observation_space,
                           env.unwrapped.sim_env.episode_length)
    return [(s, ts.run_policy_episode(env, s, policy)["return"]) for s in chunk]


def _eval(seeds, solver, model_dir, workers):
    w = max(1, min(workers, len(seeds)))
    chunks = [seeds[i::w] for i in range(w)]
    with get_context("spawn").Pool(processes=w) as pool:
        res = pool.map(_run_chunk, [(c, solver, model_dir) for c in chunks])
    rets = [r for c in res for _, r in c]
    import numpy as np
    return float(np.mean(rets)), float(np.std(rets, ddof=1) / len(rets) ** 0.5), len(rets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", required=True)
    ap.add_argument("--model-dir", default="/tmp/nomodel")
    ap.add_argument("--visible-from", default="/app/selfcheck.py",
                    help="file defining VISIBLE_SEEDS (the shipped selfcheck)")
    ap.add_argument("--z", type=float, default=2.5)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    visible = list(_load_attr(a.visible_from, "VISIBLE_SEEDS"))
    sealed = list(_load_attr(str(HERE / "heldout_seeds.py"), "HELDOUT_SEEDS"))

    mv, sv, nv = _eval(visible, a.solver, a.model_dir, a.workers)
    mh, sh, nh = _eval(sealed, a.solver, a.model_dir, a.workers)
    gap = abs(mv - mh)
    tol = a.z * (sv ** 2 + sh ** 2) ** 0.5
    verdict = "PASS" if gap <= tol else "FAIL"
    print(json.dumps({
        "visible": {"mean": mv, "sem": sv, "n": nv},
        "sealed": {"mean": mh, "sem": sh, "n": nh},
        "gap": gap, "tolerance": tol, "z": a.z, "verdict": verdict,
    }, indent=1))
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
