#!/usr/bin/env python3
from __future__ import annotations
import hashlib, math
import numpy as np


def _sanitize(x, inst):
    lo = np.asarray(inst["lower"], dtype=np.float64)
    hi = np.asarray(inst["upper"], dtype=np.float64)
    z = np.clip(np.asarray(x, dtype=np.float64), lo, hi)
    types = inst.get("variable_types", ["continuous"] * z.size)
    if len(types) != z.size:
        raise ValueError("variable_types length mismatch")
    if any(variable_type != "continuous" for variable_type in types):
        raise ValueError("this task accepts continuous variables only")
    return z


def _stable_noise(x, seed, call_count, scale):
    arr = np.asarray(x, dtype=np.float64)
    key = arr.round(8).tobytes() + int(seed).to_bytes(8, "little", signed=False) + int(call_count).to_bytes(8, "little", signed=False)
    h = hashlib.sha256(key).digest()
    u1 = (int.from_bytes(h[:8], "little") + 1) / 2**64
    u2 = (int.from_bytes(h[8:16], "little") + 1) / 2**64
    return float(scale * math.sqrt(-2.0 * math.log(max(u1, 1e-15))) * math.cos(2.0 * math.pi * u2))


def latent_objective(inst, x):
    """Return the noise-free objective used to score visible trajectories.

    Optimizers never receive this value: :func:`make_objective` remains the
    only feedback channel passed to ``tell``. Keeping the latent evaluation
    separate prevents a lucky negative noise draw from looking like an
    optimization improvement in the visible self-check.
    """
    if inst.get("kind") != "noisy_feature_match":
        raise ValueError("this task accepts noisy continuous instances only")

    xx = _sanitize(x, inst)
    matrix = np.asarray(inst["feature_matrix"], dtype=np.float64)
    phase = np.asarray(inst["feature_phase"], dtype=np.float64)
    target = np.asarray(inst["feature_target"], dtype=np.float64)
    weights = np.asarray(inst["feature_weights"], dtype=np.float64)
    if matrix.shape != (target.size, xx.size) or phase.shape != target.shape:
        raise ValueError("feature asset shape mismatch")
    if weights.shape != target.shape or np.any(weights <= 0.0):
        raise ValueError("feature weights must be positive")
    residual = np.sin(matrix @ xx + phase) - target
    return float(
        float(inst["latent_scale"])
        * np.sum(weights * residual * residual)
        / np.sum(weights)
    )


def make_objective(inst, seed=0):
    if inst.get("kind") != "noisy_feature_match":
        raise ValueError("this task accepts noisy continuous instances only")

    lo = np.asarray(inst["lower"], dtype=np.float64)
    hi = np.asarray(inst["upper"], dtype=np.float64)
    call = {"n": 0}

    def f(x):
        call["n"] += 1
        xx = _sanitize(x, inst)
        base = latent_objective(inst, xx)
        return float(base + _stable_noise(xx, int(inst["noise_seed"]) + int(seed), call["n"], float(inst["noise_scale"])))

    return f, lo, hi
