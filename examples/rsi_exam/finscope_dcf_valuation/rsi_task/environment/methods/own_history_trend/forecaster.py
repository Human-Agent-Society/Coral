"""methods/own_history_trend — a second inherited attempt (per-company trend).

A different direction the prior analyst sketched: instead of holding the last
value flat, fit each company's OWN recent history with a short linear trend and
extrapolate it forward, then let growth drift partway toward a low floor. It is
self-contained per company — it uses no cross-sectional information and no
sector prior — so it tends to chase idiosyncratic noise on short or messy
histories and it still folds the special-items years into the trend. It is here
as an alternative starting point to compare against `main`, not as a finished
method. (The grader runs `methods/main`; this directory is scanned by the
global-best fallback.)

Contract: solve(problem) -> {company_id: drivers}.
"""
from __future__ import annotations

DEFAULT_WACC = 0.09
DEFAULT_TERMINAL_GROWTH = 0.025


def _series(history, field):
    return [row.get(field) for row in history if row.get(field) is not None]


def _linfit_last(values, k=3):
    """Slope+level of the last <=k points (least squares); naive, keeps outliers."""
    pts = values[-k:]
    n = len(pts)
    if n == 0:
        return None, None
    if n == 1:
        return 0.0, pts[0]
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(pts) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0, my
    b = sum((x - mx) * (y - my) for x, y in zip(xs, pts)) / sxx
    return b, pts[-1]


def solve(problem) -> dict:
    H = problem.horizon
    out: dict = {}
    for c in problem.load_companies():
        if c["set"] != "holdout":
            continue
        h = c["history"]
        gs = _series(h, "rev_growth")
        ms = _series(h, "ebit_margin")
        rs = _series(h, "reinvest")
        gb, gl = _linfit_last(gs, 3)
        mb, ml = _linfit_last(ms, 3)
        g0 = gl if gl is not None else 0.05
        m0 = ml if ml is not None else 0.12
        gb = gb if gb is not None else 0.0
        mb = mb if mb is not None else 0.0
        r0 = rs[-1] if rs else 0.45
        rev_growth, ebit_margin = [], []
        for t in range(1, H + 1):
            # Raw linear extrapolation of the last-3 trend, no fade-to-floor and
            # no cross-sectional anchor: on a decelerating or outlier-laden
            # history this over-/under-shoots, which is exactly the flaw to fix.
            g_t = g0 + gb * t
            m_t = m0 + mb * t
            rev_growth.append(max(-0.10, min(0.50, g_t)))
            ebit_margin.append(max(0.02, min(0.55, m_t)))
        out[c["id"]] = {
            "rev_growth": rev_growth,
            "ebit_margin": ebit_margin,
            "reinvest": [max(0.10, min(0.90, r0))] * H,
            "wacc": DEFAULT_WACC,
            "terminal_growth": DEFAULT_TERMINAL_GROWTH,
        }
    return out
