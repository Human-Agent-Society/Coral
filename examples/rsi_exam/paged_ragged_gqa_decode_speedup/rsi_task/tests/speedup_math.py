from __future__ import annotations

import math


def _positive_finite(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    return number


def bracketed_raw_speedup(
    baseline_before_ms: float,
    candidate_ms: float,
    baseline_after_ms: float,
) -> tuple[float, float]:
    """Return (geometric bracket baseline, baseline/candidate raw speedup)."""
    before = _positive_finite(baseline_before_ms, "baseline_before_ms")
    candidate = _positive_finite(candidate_ms, "candidate_ms")
    after = _positive_finite(baseline_after_ms, "baseline_after_ms")
    bracket = math.sqrt(before * after)
    return bracket, bracket / candidate


def geometric_mean_speedup(values: list[float]) -> float:
    if not values:
        raise ValueError("at least one per-case speedup is required")
    checked = [_positive_finite(value, "per_case_speedup") for value in values]
    return math.exp(math.fsum(math.log(value) for value in checked) / len(checked))
