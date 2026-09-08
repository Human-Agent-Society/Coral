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
    for j, typ in enumerate(types):
        if typ == "binary":
            z[j] = 1.0 if z[j] >= 0.5 else 0.0
        elif typ == "integer":
            z[j] = float(np.clip(np.rint(z[j]), lo[j], hi[j]))
        elif typ == "categorical":
            # Categories are encoded as 0,1,2,...; upper may be wider than the active category count.
            sizes = inst.get("category_sizes", [])
            cat_idx = sum(1 for t in types[:j] if t == "categorical")
            max_cat = (int(sizes[cat_idx]) - 1) if cat_idx < len(sizes) else int(hi[j])
            z[j] = float(np.clip(np.rint(z[j]), 0, max_cat))
    return z


def _rastrigin(z):
    return float(10.0 * z.size + np.sum(z * z - 10.0 * np.cos(2.0 * math.pi * z)))


def _ackley(z):
    z = np.asarray(z, dtype=np.float64)
    n = z.size
    return float(-20.0 * math.exp(-0.2 * math.sqrt(np.sum(z*z) / n)) - math.exp(np.sum(np.cos(2*math.pi*z)) / n) + 20.0 + math.e)


def _stable_noise(x, seed, call_count, scale):
    arr = np.asarray(x, dtype=np.float64)
    key = arr.round(8).tobytes() + int(seed).to_bytes(8, "little", signed=False) + int(call_count).to_bytes(8, "little", signed=False)
    h = hashlib.sha256(key).digest()
    u1 = (int.from_bytes(h[:8], "little") + 1) / 2**64
    u2 = (int.from_bytes(h[8:16], "little") + 1) / 2**64
    return float(scale * math.sqrt(-2.0 * math.log(max(u1, 1e-15))) * math.cos(2.0 * math.pi * u2))


def latent_objective(inst, x):
    """Return the noise-free target used only by the trusted evaluator.

    The runtime optimizer receives the noisy observation from ``make_objective``;
    this evaluator-only function prevents lucky negative noise from becoming a
    leaderboard improvement.
    """
    xx = _sanitize(x, inst)
    if inst.get("kind") == "noisy_feature_match":
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
    shift = np.asarray(inst["_shift"], dtype=np.float64)
    rotation = np.asarray(inst["_rotation"], dtype=np.float64)
    z = rotation @ (xx - shift)
    return float(0.68 * _rastrigin(z) + 0.32 * (15.0 * _ackley(0.7 * z)))


def _inventory_cost(x, inst, seed, call_count):
    # Encode policy: x[0]=reorder percentile, x[1]=order-up-to percentile, x[2]=expedite threshold, x[3]=smoothing.
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    mean = float(inst["demand_mean"])
    cv = float(inst["demand_cv"])
    periods = int(inst["periods"])
    reps = int(inst["replications"])
    s = 0.45 * mean + x[0] * 1.75 * mean
    S = s + 0.35 * mean + x[1] * 2.4 * mean
    expedite_trigger = x[2] * mean
    smooth = 0.3 + 0.7 * x[3]
    rng = np.random.default_rng(int(inst["scenario_seed"]) + 7919 * int(seed) + 104729 * int(call_count))
    total = 0.0
    stockouts = 0.0
    demand_total = 0.0
    for _ in range(reps):
        on_hand = S
        backlog = 0.0
        pipeline = []
        cost = 0.0
        for t in range(periods):
            arrivals = [q for due, q in pipeline if due <= t]
            if arrivals:
                on_hand += sum(arrivals)
            pipeline = [(due, q) for due, q in pipeline if due > t]
            sigma = max(1e-6, mean * cv)
            shape = (mean / sigma) ** 2
            scale = sigma * sigma / mean
            demand = float(rng.gamma(shape, scale))
            demand_total += demand
            served = min(on_hand, demand)
            on_hand -= served
            lost = demand - served
            backlog += lost
            stockouts += lost
            cost += float(inst["holding_cost"]) * on_hand + float(inst["backlog_cost"]) * backlog
            inv_position = on_hand - backlog + sum(q for _, q in pipeline)
            target_s = smooth * s + (1.0 - smooth) * mean
            if inv_position < target_s:
                qty = max(0.0, S - inv_position)
                lead = int(rng.poisson(float(inst["lead_mean"])))
                if backlog > expedite_trigger:
                    lead = max(0, lead - 1)
                    cost += float(inst["service_penalty"]) * 0.5
                pipeline.append((t + lead + 1, qty))
                cost += float(inst["fixed_order_cost"]) + float(inst["unit_order_cost"]) * qty
            backlog *= 0.92
        total += cost / periods
    stockout_rate = stockouts / max(demand_total, 1e-9)
    total = total / reps + float(inst["service_penalty"]) * max(0.0, stockout_rate - 0.10) * 100.0
    return float(total)


def make_objective(inst, seed=0):
    lo = np.asarray(inst["lower"], dtype=np.float64)
    hi = np.asarray(inst["upper"], dtype=np.float64)
    call = {"n": 0}
    kind = inst["kind"]

    if kind in {"noisy_continuous", "noisy_feature_match"}:
        def f(x):
            call["n"] += 1
            xx = _sanitize(x, inst)
            base = latent_objective(inst, xx)
            return float(base + _stable_noise(xx, int(inst["noise_seed"]) + int(seed), call["n"], float(inst["noise_scale"])))
        return f, lo, hi

    if kind == "constrained_continuous":
        shift = np.asarray(inst["_shift"], dtype=np.float64)
        center = np.asarray(inst["_center"], dtype=np.float64)
        rot = np.asarray(inst["_rotation"], dtype=np.float64)
        normal = np.asarray(inst["linear_normal"], dtype=np.float64)
        normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
        radius = float(inst["radius"])
        rhs = float(inst["linear_rhs"])
        def f(x):
            xx = _sanitize(x, inst)
            z = rot @ (xx - shift)
            obj = 0.5 * _rastrigin(0.8 * z) + 6.0 * float(np.sum((xx - shift) ** 2))
            g1 = max(0.0, float(np.sum((xx - center) ** 2) - radius * radius))
            g2 = max(0.0, float(np.dot(normal, xx - center) - rhs))
            wave = max(0.0, float(np.sum(np.sin(xx + center)) - 1.5))
            penalty = 500.0 * g1 + 350.0 * g2 + 150.0 * wave
            return float(obj + penalty)
        return f, lo, hi

    if kind == "mixed_integer":
        cont_shift = np.asarray(inst["_cont_shift"], dtype=np.float64)
        int_t = np.asarray(inst["_int_target"], dtype=np.float64)
        bin_t = np.asarray(inst["_bin_target"], dtype=np.float64)
        cat_t = np.asarray(inst["_cat_target"], dtype=np.float64)
        interaction = np.asarray(inst["_interaction"], dtype=np.float64)
        def f(x):
            xx = _sanitize(x, inst)
            cont = xx[:4]
            ints = xx[4:7]
            bins = xx[7:9]
            cats = xx[9:12].astype(int)
            val = 4.0 * _ackley(cont - cont_shift) + 2.0 * float(np.sum((ints - int_t) ** 2)) + 8.0 * float(np.sum(np.abs(bins - bin_t)))
            val += 7.0 * float(np.sum(cats != cat_t.astype(int)))
            val += float(sum(interaction[j, int(cats[j])] for j in range(3)))
            val += 2.0 * math.sin(float(np.sum(cont) + np.sum(ints) + np.sum(cats)))
            return float(val)
        return f, lo, hi

    if kind == "discrete_pbo":
        planted = np.asarray(inst["_planted"], dtype=int)
        weights = np.asarray(inst["_weights"], dtype=np.float64)
        edges = inst["_edges"]
        traps = inst["_traps"]
        def f(x):
            b = _sanitize(x, inst).astype(int)
            match = (b == planted).astype(float)
            score = float(np.sum(weights * match))
            for a, c, w in edges:
                if (b[a] ^ b[c]) == (planted[a] ^ planted[c]):
                    score += float(w)
            for block in traps:
                m = int(np.sum(b[block] == planted[block]))
                if m == len(block):
                    score += 4.0
                else:
                    score += max(0, len(block) - 1 - m) * 0.55
            return float(-score)
        return f, lo, hi

    if kind == "simopt_inventory":
        def f(x):
            call["n"] += 1
            xx = _sanitize(x, inst)
            return _inventory_cost(xx, inst, seed, call["n"])
        return f, lo, hi

    raise ValueError(f"unknown kind {kind}")
