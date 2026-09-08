import numpy as np


def estimate_logf(case):
    """Weak inherited baseline: a single homogeneous bottom-friction value."""
    return np.full((24, 24), -1.0, dtype=float)
