"""Starting method: window trend fit + straight-line extrapolation.

Interface (do not change): predict(record) -> list of len(record["extreme"])
floats, the predicted log10 of |I| in amperes at each grid point, in order.
The sealed evaluation loads THIS file from methods/main/ and calls predict()
once per device in a fresh process.

What this method does: on the reverse branch of the safe-window sweep it
fits, by regularized least squares in log space,

    log10|I| = a + b * (1000 / T) + c * log10(1 + |V|)

i.e. a single effective thermal-activation slope plus a single voltage
power law, and evaluates the same expression at the qualification grid --
except that, being unwilling to trust a window-fitted voltage law tens of
volts past the characterized range, it CLAMPS |V| at the deepest measured
reverse bias (it extrapolates in temperature only). It tracks the window
data reasonably well and carries a real trend signal into the
extrapolation, but it is far from what the records support -- everything
about it is yours to redesign, including whether you fit closed forms at
all or build a physics simulation of the lot (see
examples/devsim_diode_demo.py for a working device-simulation starting
point).
"""
from __future__ import annotations

import numpy as np

RIDGE = 1e-6
I_CLIP = 1e-14  # A, below the instrument floor


def _design(T, V):
    T = np.asarray(T, float)
    V = np.asarray(V, float)
    return np.column_stack([np.ones_like(T), 1000.0 / T,
                            np.log10(1.0 + np.abs(V))])


def predict(record):
    win = record["window"]
    T = np.array([p["T"] for p in win])
    V = np.array([p["V"] for p in win])
    I = np.array([p["I"] for p in win])
    rev = V < 0.0
    if rev.sum() < 6:          # degenerate record: use everything
        rev = np.ones_like(rev, bool)
    X = _design(T[rev], V[rev])
    y = np.log10(np.maximum(np.abs(I[rev]), I_CLIP))
    theta = np.linalg.solve(X.T @ X + RIDGE * np.eye(X.shape[1]), X.T @ y)
    v_edge = float(np.max(np.abs(V[rev])))   # deepest characterized bias
    Tq = np.array([p["T"] for p in record["extreme"]], float)
    Vq = np.array([p["V"] for p in record["extreme"]], float)
    Vq_c = np.sign(Vq) * np.minimum(np.abs(Vq), v_edge)
    Xq = _design(Tq, Vq_c)
    return (Xq @ theta).tolist()
