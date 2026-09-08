"""Pure reward mapping shared by the sealed grader and review diagnostics."""

from __future__ import annotations

import math


def reward_for(metric: float, anchors: dict[str, float]) -> float:
    """baseline->0, reference->0.6, upper->1."""
    names = ("BASELINE", "REFERENCE", "UPPER_BOUND")
    values = (metric, *(float(anchors[name]) for name in names))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("metric and anchors must be finite")
    baseline, reference, upper = values[1:]
    if not (0.0 <= baseline < reference < upper == 1.0):
        raise ValueError("invalid bounded macro-F1 anchor ordering")
    points = [(baseline, 0.0), (reference, 0.6), (upper, 1.0)]
    if metric <= baseline:
        return 0.0
    for (x0, r0), (x1, r1) in zip(points, points[1:]):
        if metric <= x1:
            return float(r0 + (r1 - r0) * (metric - x0) / (x1 - x0))
    return 1.0
