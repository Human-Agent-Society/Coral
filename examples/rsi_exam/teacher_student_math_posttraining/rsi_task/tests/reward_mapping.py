from __future__ import annotations


def anchored_reward(metric: float, baseline: float, upper: float) -> float:
    """baseline->0, upper->1, linear in between."""
    if metric <= baseline:
        return 0.0
    if metric >= upper:
        return 1.0
    return (metric - baseline) / (upper - baseline)
