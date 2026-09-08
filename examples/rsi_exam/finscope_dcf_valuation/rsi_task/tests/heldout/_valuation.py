"""Public revenue->FCF->DCF bridge for finscope_dcf_valuation.

This module is VISIBLE to the agent. It defines *how* a company's forward
value-drivers turn into an intrinsic value per share. What is hidden is the
*realized* driver values (only the verifier, under tests/, holds those).

The discounting itself is delegated to the vendored FinScope `fin-compute`
engine (`fin_compute_engine.dcf_value_checked`) -- the exact tested function the
trusted verifier uses to compute the perfect-foresight truth. So an agent that
uses this bridge values a company with *identical* mechanics to the judge; its
only job is to forecast good drivers.

Driver schema (per company), H = HORIZON = 5 explicit years:
    rev_growth      : [g_1 .. g_H]   revenue growth each year (e.g. 0.12 = +12%)
    ebit_margin     : [m_1 .. m_H]   EBIT margin each year   (e.g. 0.25 = 25%)
    reinvest        : [r_1 .. r_H]   reinvestment as a fraction of NOPAT in [0,1)
    wacc            : float          discount rate, in (0, 1)
    terminal_growth : float          Gordon terminal growth, must be < wacc

Observable facts (per company), given as-of the valuation date T:
    revenue_0  : latest annual revenue (level)
    tax_rate   : cash tax rate in [0, 1)
    net_debt   : net debt (negative = net cash)
    shares     : shares outstanding

Value chain for year t (1-indexed):
    revenue_t = revenue_0 * prod_{j<=t} (1 + rev_growth_j)
    nopat_t   = revenue_t * ebit_margin_t * (1 - tax_rate)
    fcf_t     = nopat_t   * (1 - reinvest_t)
then PV(explicit FCF) + PV(Gordon terminal on fcf_H) - net_debt, per share.
"""
from __future__ import annotations

import math
from typing import Any

try:
    import fin_compute_engine as engine          # agent-visible copy (environment/)
except ImportError:                              # trusted verifier copy (tests/)
    import _fin_compute_engine as engine

HORIZON = 5  # explicit forecast years

# Broad sanity bounds used by the verifier's correctness gate. In-range but
# suboptimal drivers are SCORED (not rejected); only structurally invalid drivers
# are rejected (-> reward 0).
BOUNDS = {
    "rev_growth": (-0.99, 5.0),
    "ebit_margin": (-1.0, 1.0),
    "reinvest": (0.0, 0.99),
    "wacc": (1e-4, 1.0),
    "terminal_growth": (-0.5, None),  # upper bound is wacc (checked separately)
}

DRIVER_FIELDS = ("rev_growth", "ebit_margin", "reinvest", "wacc", "terminal_growth")


def _is_num(x: Any) -> bool:
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


def validate_drivers(d: Any) -> list[str]:
    """Return a list of structural problems with one company's drivers.

    Empty list == structurally valid (will be SCORED). Non-empty == reject.
    """
    errs: list[str] = []
    if not isinstance(d, dict):
        return ["drivers is not an object"]
    for f in DRIVER_FIELDS:
        if f not in d:
            errs.append(f"missing field '{f}'")
    if errs:
        return errs
    for f in ("rev_growth", "ebit_margin", "reinvest"):
        seq = d[f]
        if not isinstance(seq, list) or len(seq) != HORIZON:
            errs.append(f"'{f}' must be a list of {HORIZON} numbers")
            continue
        lo, hi = BOUNDS[f]
        for i, v in enumerate(seq):
            if not _is_num(v):
                errs.append(f"'{f}[{i}]' is not a finite number")
            elif (lo is not None and v < lo) or (hi is not None and v > hi):
                errs.append(f"'{f}[{i}]'={v} out of range [{lo},{hi}]")
    w, tg = d["wacc"], d["terminal_growth"]
    if not _is_num(w):
        errs.append("'wacc' is not a finite number")
    elif not (BOUNDS["wacc"][0] <= w <= BOUNDS["wacc"][1]):
        errs.append(f"'wacc'={w} out of range {BOUNDS['wacc']}")
    if not _is_num(tg):
        errs.append("'terminal_growth' is not a finite number")
    elif tg < BOUNDS["terminal_growth"][0]:
        errs.append(f"'terminal_growth'={tg} below {BOUNDS['terminal_growth'][0]}")
    # Hard finance constraint: Gordon terminal requires terminal_growth < wacc.
    if _is_num(w) and _is_num(tg) and not (tg < w):
        errs.append(f"terminal_growth ({tg}) must be < wacc ({w}) (Gordon)")
    return errs


def fcf_series(drivers: dict, facts: dict) -> list[float]:
    """Forward free-cash-flow series fcf_1..fcf_H from drivers + observable facts."""
    rev0 = float(facts["revenue_0"])
    tax = float(facts["tax_rate"])
    g = drivers["rev_growth"]
    m = drivers["ebit_margin"]
    r = drivers["reinvest"]
    fcf: list[float] = []
    cum = 1.0
    for t in range(HORIZON):
        cum *= (1.0 + float(g[t]))
        rev_t = rev0 * cum
        nopat_t = rev_t * float(m[t]) * (1.0 - tax)
        fcf.append(nopat_t * (1.0 - float(r[t])))
    return fcf


def intrinsic_value(drivers: dict, facts: dict) -> tuple[float | None, str | None]:
    """Per-share intrinsic value via the vendored fin-compute DCF engine.

    Returns (per_share | None, reason | None). None when the engine's guards trip
    (wacc<=0, terminal_growth>=wacc, shares<=0) -- never raises, never sign-flips.
    """
    fcf = fcf_series(drivers, facts)
    # Re-express the explicit FCF series as the engine's (fcf_base, growth-path)
    # form so the discounting/terminal/net-debt/per-share math is the engine's,
    # bit for bit: year-1 = fcf_base, year-i = fcf_base * prod(1+a_j).
    fcf_base = fcf[0]
    if fcf_base == 0:
        return None, "year-1 FCF is zero -- growth path undefined"
    fcf_growth = [0.0]
    for i in range(1, HORIZON):
        prev = fcf[i - 1]
        if prev == 0:
            return None, "zero FCF in explicit window -- growth path undefined"
        fcf_growth.append(fcf[i] / prev - 1.0)
    val, reason = engine.dcf_value_checked(
        fcf_base=fcf_base,
        fcf_growth=fcf_growth,
        wacc=float(drivers["wacc"]),
        terminal_growth=float(drivers["terminal_growth"]),
        net_debt=float(facts["net_debt"]),
        shares=float(facts["shares"]),
    )
    if reason is not None:
        return None, reason
    return val["per_share"], None
