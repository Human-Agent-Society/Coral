#!/usr/bin/env python3
"""Run the public optimizer against the visible leaderboard protocol.

The optimizer receives only noisy observations. The self-check separately
records the latent best-so-far trajectory and applies the same score family as
the sealed evaluator: median over deterministic replicates, normalization
between a uniform-search trace and the exact per-instance optimum, then
per-instance 70% pre-final anytime quality plus 30% final quality followed by
the mean across instances.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

import bbo_harness


ROOT = Path(__file__).resolve().parent
DEFAULT_SOLVER = ROOT / "methods" / "main" / "solver.py"
DATA = json.loads((ROOT / "data" / "visible.json").read_text(encoding="utf-8"))
N_SEEDS = 20
ANYTIME_WEIGHT = 0.70
FINAL_WEIGHT = 0.30
_SEED_STEP = 0x9E3779B97F4A7C15
_SEED_MASK = (1 << 63) - 1
_TOL = 1e-10


class _UniformNormalizationBaseline:
    """The deterministic visible quality-zero policy."""

    batch = 120

    def __init__(self, dim, lower, upper, budget, seed, rng):
        del dim, budget, seed
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.rng = rng

    def ask(self, n):
        return self.lower + (self.upper - self.lower) * self.rng.random(
            (int(n), self.lower.size)
        )

    def tell(self, X, y):
        del X, y


def _load_optimizer(path: Path):
    spec = importlib.util.spec_from_file_location("visible_submitted_solver", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import optimizer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Optimizer"):
        raise AttributeError(f"{path} must define Optimizer")
    return module.Optimizer


def _run_seed(instance: dict[str, Any], *, instance_index: int, seed_index: int) -> int:
    base = int(instance.get("seed", 1000 * instance_index))
    return int((base + _SEED_STEP * (seed_index + 1)) & _SEED_MASK)


def _batch_hint(optimizer: Any, *, remaining: int) -> int:
    try:
        raw = getattr(optimizer, "batch", 1)
        if isinstance(raw, bool):
            raise TypeError("bool is not a batch size")
        if isinstance(raw, (int, np.integer)):
            value = int(raw)
        elif isinstance(raw, float) and np.isfinite(raw) and raw.is_integer():
            value = int(raw)
        else:
            raise TypeError("batch must be an integer")
    except Exception:
        value = 1
    return min(max(value, 1), max(remaining, 1))


def _validate_points(
    points: Any,
    *,
    dim: int,
    lower: np.ndarray,
    upper: np.ndarray,
    requested: int,
    remaining: int,
) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != dim:
        raise ValueError(f"ask must return shape [n, {dim}]")
    if not 1 <= array.shape[0] <= min(requested, remaining):
        raise ValueError("ask returned an invalid batch size")
    if not np.isfinite(array).all():
        raise ValueError("ask returned a non-finite point")
    if np.any(array < lower[None, :]) or np.any(array > upper[None, :]):
        raise ValueError("ask returned an out-of-bounds point")
    return array


def _tell(optimizer: Any, X: np.ndarray, y: np.ndarray) -> None:
    """Match the sealed child's handling of optional metadata=None."""
    tell = optimizer.tell
    signature = inspect.signature(tell)
    try:
        signature.bind(X, y, None)
    except TypeError:
        signature.bind(X, y)
        tell(X, y)
    else:
        tell(X, y, None)


def _run_one(
    optimizer_type: type,
    instance: dict[str, Any],
    *,
    seed: int,
    budget: int,
) -> np.ndarray:
    noisy_objective, lower, upper = bbo_harness.make_objective(instance, seed=seed)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    dim = int(lower.size)
    rng = np.random.default_rng(seed)
    optimizer = optimizer_type(
        dim=dim,
        lower=lower,
        upper=upper,
        budget=budget,
        seed=seed,
        rng=rng,
    )

    best_latent = float("inf")
    trace: list[float] = []
    used = 0
    next_batch = _batch_hint(optimizer, remaining=budget)
    while used < budget:
        requested = min(next_batch, budget - used)
        X = _validate_points(
            optimizer.ask(requested),
            dim=dim,
            lower=lower,
            upper=upper,
            requested=requested,
            remaining=budget - used,
        )
        # The sealed child reports its next hint immediately after ask().
        ask_batch_hint = _batch_hint(optimizer, remaining=budget - used)
        observed = []
        for point in X:
            noisy_value = float(noisy_objective(point))
            latent_value = float(bbo_harness.latent_objective(instance, point))
            if not np.isfinite(noisy_value) or not np.isfinite(latent_value):
                raise ValueError("objective returned a non-finite value")
            observed.append(noisy_value)
            best_latent = min(best_latent, latent_value)
            trace.append(best_latent)
        y = np.asarray(observed, dtype=float)
        used += int(X.shape[0])
        _tell(optimizer, X, y)
        next_batch = min(ask_batch_hint, max(budget - used, 1))

    if len(trace) != budget:
        raise RuntimeError(f"trace has {len(trace)} entries, expected {budget}")
    return np.asarray(trace, dtype=float)


def _collect_traces(optimizer_type: type) -> np.ndarray:
    budget = int(DATA["budget"])
    traces = []
    for instance_index, instance in enumerate(DATA["instances"]):
        instance_traces = []
        for seed_index in range(N_SEEDS):
            seed = _run_seed(
                instance,
                instance_index=instance_index,
                seed_index=seed_index,
            )
            instance_traces.append(
                _run_one(optimizer_type, instance, seed=seed, budget=budget)
            )
        traces.append(instance_traces)
    result = np.asarray(traces, dtype=float)
    expected = (len(DATA["instances"]), N_SEEDS, budget)
    if result.shape != expected:
        raise RuntimeError(f"trace tensor has shape {result.shape}, expected {expected}")
    return result


def _score(candidate_traces: np.ndarray, floor_traces: np.ndarray) -> dict[str, float]:
    candidate = np.median(candidate_traces, axis=1)
    floor = np.median(floor_traces, axis=1)
    oracle = np.zeros(len(DATA["instances"]), dtype=float)
    denominator = floor - oracle[:, None]
    threshold = _TOL * (1.0 + np.maximum(np.abs(floor), np.abs(oracle[:, None])))
    if np.any(denominator <= threshold):
        bad = np.argwhere(denominator <= threshold)[0].tolist()
        raise ValueError(f"invalid visible normalization at instance/checkpoint {bad}")

    quality = np.clip((floor - candidate) / denominator, 0.0, 1.0)
    anytime_per_instance = np.mean(quality[:, :-1], axis=1)
    final_per_instance = quality[:, -1]
    combined_per_instance = (
        ANYTIME_WEIGHT * anytime_per_instance + FINAL_WEIGHT * final_per_instance
    )
    score_anytime = float(np.mean(anytime_per_instance))
    score_final = float(np.mean(final_per_instance))
    score = float(np.clip(np.mean(combined_per_instance), 0.0, 1.0))
    return {
        "score": score,
        "score_anytime": score_anytime,
        "score_final": score_final,
        "mean_final_latent_objective": float(np.mean(candidate[:, -1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        type=Path,
        default=DEFAULT_SOLVER,
        help="optimizer module to evaluate (default: /app/methods/main/solver.py)",
    )
    parser.add_argument("--json", action="store_true", help="emit one JSON object")
    args = parser.parse_args()

    optimizer_type = _load_optimizer(args.solver.resolve())
    candidate_traces = _collect_traces(optimizer_type)
    floor_traces = _collect_traces(_UniformNormalizationBaseline)
    result: dict[str, Any] = {
        "metric": "oracle_normalized_auc70_final30",
        "direction": "higher",
        **_score(candidate_traces, floor_traces),
        "n_instances": len(DATA["instances"]),
        "n_seeds": N_SEEDS,
        "budget": int(DATA["budget"]),
    }
    if args.json:
        print(json.dumps(result, allow_nan=False, sort_keys=True))
        if Path("/app/budget.py").exists():
            import subprocess, sys
            subprocess.run([sys.executable, "/app/budget.py"], check=False)
        return

    print(
        "visible oracle-normalized score = "
        f"{result['score']:.9f} "
        "(higher is better; 70% anytime + 30% final)"
    )
    print(
        f"components: anytime={result['score_anytime']:.9f}, "
        f"final={result['score_final']:.9f}; "
        f"mean final latent objective={result['mean_final_latent_objective']:.6f}"
    )
    print(
        f"protocol: {result['n_instances']} visible instances x "
        f"{result['n_seeds']} deterministic seeds x {result['budget']} queries; "
        "the optimizer received noisy observations only"
    )


    if Path("/app/budget.py").exists():
        import subprocess, sys
        subprocess.run([sys.executable, "/app/budget.py"], check=False)

if __name__ == "__main__":
    main()
