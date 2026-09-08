#!/usr/bin/env python3
"""FinScope deterministic valuation / calculation engine.

A library of pure, auditable financial formulas (the DCF valuation used here is
``dcf_value_checked``). Every number comes from running these functions, so any
result is reproducible.

Guard policy: a formula that cannot produce a meaningful value returns ``None``
and records WHY — it never raises mid-pipeline and never returns a sign-flipped
number (e.g. YoY growth on a negative base). Each public formula ``f(...)`` has a
checked companion ``f_checked(...) -> (value_or_None, reason_or_None)`` so a None
result propagates as an explicit reason rather than NaN or a silent omission.

CLI:
  python engine.py --facts facts.jsonl --assumptions assumptions.yaml \
      --out valuation_model.csv --log recompute_log.json \
      [--ticker NVDA] [--scenarios scenarios.json --price 205.10]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from typing import Any

__version__ = "2.0.0"

_MISSING = "missing input"


def _num(x: Any) -> bool:
    """True iff x is a finite real number (bools, None, NaN, inf rejected)."""
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(float(x))
    )


# --------------------------------------------------------------------------
# Core formulas (scale-invariant where it matters; see fin-theory-pack refs)
# Each `*_checked` returns (value | None, reason | None).
# --------------------------------------------------------------------------
def growth_checked(curr: float, prev: float) -> tuple[float | None, str | None]:
    if not _num(curr) or not _num(prev):
        return None, f"{_MISSING} — growth undefined"
    if prev == 0:
        return None, "zero base — growth undefined"
    if prev < 0:
        return None, "negative base — YoY undefined; report Δ (delta_change) instead"
    return curr / prev - 1.0, None


def growth(curr: float, prev: float) -> float | None:
    """YoY / QoQ = curr/prev - 1. None if base is zero/negative (no sign flips)."""
    return growth_checked(curr, prev)[0]


def delta_change_checked(curr: float, prev: float) -> tuple[float | None, str | None]:
    if not _num(curr) or not _num(prev):
        return None, _MISSING
    return curr - prev, None


def delta_change(curr: float, prev: float) -> float | None:
    """Absolute change curr - prev — the companion metric when growth() is
    undefined (negative base): −100 → −50 is Δ = +50 (loss halved)."""
    return delta_change_checked(curr, prev)[0]


def gross_margin_checked(revenue: float, cogs: float) -> tuple[float | None, str | None]:
    if not _num(revenue) or not _num(cogs):
        return None, _MISSING
    if revenue <= 0:
        return None, "non-positive revenue — margin not meaningful"
    return (revenue - cogs) / revenue, None


def gross_margin(revenue: float, cogs: float) -> float | None:
    return gross_margin_checked(revenue, cogs)[0]


def net_margin_checked(net_income: float, revenue: float) -> tuple[float | None, str | None]:
    if not _num(net_income) or not _num(revenue):
        return None, _MISSING
    if revenue <= 0:
        return None, "non-positive revenue — margin not meaningful"
    return net_income / revenue, None


def net_margin(net_income: float, revenue: float) -> float | None:
    return net_margin_checked(net_income, revenue)[0]


def roe_checked(net_income: float, avg_equity: float) -> tuple[float | None, str | None]:
    if not _num(net_income) or not _num(avg_equity):
        return None, _MISSING
    if avg_equity <= 0:
        return None, "negative/zero equity — ROE not meaningful"
    return net_income / avg_equity, None


def roe(net_income: float, avg_equity: float) -> float | None:
    return roe_checked(net_income, avg_equity)[0]


def roic_checked(nopat: float, invested_capital: float) -> tuple[float | None, str | None]:
    if not _num(nopat) or not _num(invested_capital):
        return None, _MISSING
    if invested_capital <= 0:
        return None, "negative/zero invested capital — ROIC not meaningful"
    return nopat / invested_capital, None


def roic(nopat: float, invested_capital: float) -> float | None:
    return roic_checked(nopat, invested_capital)[0]


def pe_checked(market_cap: float, net_income: float) -> tuple[float | None, str | None]:
    if not _num(market_cap) or not _num(net_income):
        return None, _MISSING
    if net_income <= 0:
        return None, "non-positive earnings — PE not meaningful"
    return market_cap / net_income, None


def pe(market_cap: float, net_income: float) -> float | None:
    return pe_checked(market_cap, net_income)[0]


def peg_checked(pe_ratio: float, cagr_pct: float) -> tuple[float | None, str | None]:
    """PEG = PE / (earnings CAGR in percentage points), e.g. cagr_pct=20 for 20%."""
    if not _num(pe_ratio) or not _num(cagr_pct):
        return None, _MISSING
    if pe_ratio <= 0:
        return None, "non-positive PE — PEG not meaningful"
    if cagr_pct <= 0:
        return None, "non-positive growth — PEG not meaningful"
    return pe_ratio / cagr_pct, None


def peg(pe_ratio: float, cagr_pct: float) -> float | None:
    return peg_checked(pe_ratio, cagr_pct)[0]


def ev_checked(market_cap: float, debt: float, minority: float,
               cash: float) -> tuple[float | None, str | None]:
    if not all(_num(x) for x in (market_cap, debt, minority, cash)):
        return None, _MISSING
    return market_cap + debt + minority - cash, None


def ev(market_cap: float, debt: float, minority: float, cash: float) -> float | None:
    return ev_checked(market_cap, debt, minority, cash)[0]


def fcf_checked(cfo: float, capex: float) -> tuple[float | None, str | None]:
    if not _num(cfo) or not _num(capex):
        return None, _MISSING
    return cfo - capex, None


def fcf(cfo: float, capex: float) -> float | None:
    return fcf_checked(cfo, capex)[0]


def cagr_checked(begin: float, end: float, years: float) -> tuple[float | None, str | None]:
    if not all(_num(x) for x in (begin, end, years)):
        return None, _MISSING
    if years <= 0:
        return None, "non-positive years — CAGR undefined"
    if begin <= 0:
        return None, "non-positive begin value — CAGR undefined"
    if end < 0:
        return None, "negative end value — CAGR undefined; report Δ (delta_change) instead"
    return (end / begin) ** (1.0 / years) - 1.0, None


def cagr(begin: float, end: float, years: float) -> float | None:
    return cagr_checked(begin, end, years)[0]


def dcf_value_checked(
    fcf_base: float,
    fcf_growth: list[float],
    wacc: float,
    terminal_growth: float,
    net_debt: float,
    shares: float,
) -> tuple[dict[str, float] | None, str | None]:
    """Single-stage explicit DCF with Gordon terminal value.

    Year i (1-indexed) FCF = fcf_base * prod(1+g_j for j<=i). Discount at (1+wacc)^i.
    Terminal = last_FCF*(1+terminal_growth)/(wacc-terminal_growth), discounted at (1+wacc)^N.
    equity_value = enterprise_value - net_debt  (net_debt negative = net cash -> adds value).
    """
    if not all(_num(x) for x in (fcf_base, wacc, terminal_growth, net_debt, shares)) \
            or not all(_num(g) for g in fcf_growth):
        return None, _MISSING
    if wacc <= 0:
        return None, "non-positive WACC — negative/zero discount rates rejected"
    if terminal_growth >= wacc:
        return None, (f"terminal growth ({terminal_growth}) >= WACC ({wacc}) "
                      "— Gordon terminal value undefined")
    if shares <= 0:
        return None, "non-positive share count — per-share value undefined"
    pv_explicit = 0.0
    cumulative = 1.0
    last_fcf = fcf_base
    n = len(fcf_growth)
    for i, g in enumerate(fcf_growth, start=1):
        cumulative *= (1.0 + g)
        fcf_i = fcf_base * cumulative
        pv_explicit += fcf_i / (1.0 + wacc) ** i
        last_fcf = fcf_i
    terminal = last_fcf * (1.0 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal / (1.0 + wacc) ** n
    enterprise_value = pv_explicit + pv_terminal
    equity_value = enterprise_value - net_debt
    return {
        "ev": enterprise_value,
        "pv_explicit": pv_explicit,
        "pv_terminal": pv_terminal,
        "equity_value": equity_value,
        "per_share": equity_value / shares,
    }, None


def dcf_value(
    fcf_base: float,
    fcf_growth: list[float],
    wacc: float,
    terminal_growth: float,
    net_debt: float,
    shares: float,
) -> dict[str, float] | None:
    """See dcf_value_checked. Returns None (instead of raising) when
    terminal_growth >= wacc, wacc <= 0, or shares <= 0."""
    return dcf_value_checked(fcf_base, fcf_growth, wacc, terminal_growth,
                             net_debt, shares)[0]


def relative_value_checked(net_income_fwd: float, target_pe: float,
                           shares: float) -> tuple[dict[str, float] | None, str | None]:
    """Relative (PE) valuation: target market cap = fwd net income * target PE."""
    if not all(_num(x) for x in (net_income_fwd, target_pe, shares)):
        return None, _MISSING
    if shares <= 0:
        return None, "non-positive share count — per-share value undefined"
    if target_pe <= 0:
        return None, "non-positive target PE — relative valuation undefined"
    if net_income_fwd <= 0:
        return None, ("non-positive forward earnings — PE-based target price "
                      "undefined (would sign-flip)")
    target_market_cap = net_income_fwd * target_pe
    return {
        "target_market_cap": target_market_cap,
        "target_price": target_market_cap / shares,
    }, None


def relative_value(net_income_fwd: float, target_pe: float,
                   shares: float) -> dict[str, float] | None:
    return relative_value_checked(net_income_fwd, target_pe, shares)[0]


def scenario_table(assumptions: dict[str, Any], base_revenue: float) -> dict[str, dict]:
    """Project revenue & net income per scenario and value via forward PE.

    assumptions: {<case>: {revenue_growth[], gross_margin[], net_margin[], target_pe}}.
    Uses the FIRST forecast year's net income as the forward base for the PE target.
    Requires assumptions['_shares'] and assumptions['_price'] injected by the CLI.

    If a case's PE valuation is undefined (e.g. it projects a loss), its
    target_* fields are None and a 'reason' key records why.
    """
    shares = assumptions["_shares"]
    price = assumptions.get("_price")
    out: dict[str, dict] = {}
    for case, a in assumptions.items():
        if case.startswith("_"):
            continue
        revenue: list[float] = []
        cum = 1.0
        for g in a["revenue_growth"]:
            cum *= (1.0 + g)
            revenue.append(base_revenue * cum)
        net_income = [r * m for r, m in zip(revenue, a["net_margin"])]
        rv, reason = relative_value_checked(net_income[0] if net_income else None,
                                            a["target_pe"], shares)
        entry: dict[str, Any] = {
            "revenue": revenue,
            "net_income": net_income,
            "target_pe": a["target_pe"],
        }
        if reason is None:
            implied = (rv["target_price"] / price - 1.0) if (_num(price) and price > 0) else None
            entry.update({
                "target_market_cap": rv["target_market_cap"],
                "target_price": rv["target_price"],
                "implied_return": implied,
            })
        else:
            entry.update({
                "target_market_cap": None,
                "target_price": None,
                "implied_return": None,
                "reason": reason,
            })
        out[case] = entry
    return out


def scenario_ev_checked(scenarios: list[dict[str, Any]],
                        price: float) -> tuple[dict[str, Any] | None, str | None]:
    """Probability-weighted expected value across scenarios vs current price.

    scenarios: [{"name", "probability", "target_price"}, ...]. Probabilities
    must sum to 1 (±0.001). Returns {"ev", "upside_pct", "rr_ratio", "best",
    "worst"}; rr_ratio = (best - price) / (price - worst), None (with
    'rr_reason') when price - worst <= 0.
    """
    if not scenarios:
        return None, "no scenarios provided"
    probs: list[float] = []
    targets: list[float] = []
    for s in scenarios:
        name = s.get("name", "?")
        p, t = s.get("probability"), s.get("target_price")
        if not _num(p) or not _num(t):
            return None, f"scenario '{name}': missing/invalid probability or target_price"
        if p < 0:
            return None, f"scenario '{name}': negative probability"
        probs.append(float(p))
        targets.append(float(t))
    if not _num(price) or price <= 0:
        return None, "non-positive price — EV upside/risk-reward undefined"
    total = sum(probs)
    if abs(total - 1.0) > 0.001:
        return None, f"probabilities sum to {total:.4f}, expected 1.0 (±0.001)"
    ev_ = sum(p * t for p, t in zip(probs, targets))
    best, worst = max(targets), min(targets)
    downside = price - worst
    out: dict[str, Any] = {
        "ev": ev_,
        "upside_pct": ev_ / price - 1.0,
        "best": best,
        "worst": worst,
        "rr_ratio": (best - price) / downside if downside > 0 else None,
    }
    if out["rr_ratio"] is None:
        out["rr_reason"] = ("price - worst <= 0 — no modeled downside; "
                            "risk-reward ratio undefined")
    return out, None


def scenario_ev(scenarios: list[dict[str, Any]], price: float) -> dict[str, Any] | None:
    return scenario_ev_checked(scenarios, price)[0]


def sensitivity(base_price: float, var_grid: dict[str, list[float]]) -> list[dict]:
    """For each variable and delta (fractional change to target price), tabulate the shifted price."""
    rows: list[dict] = []
    for var, deltas in var_grid.items():
        for d in deltas:
            rows.append({"variable": var, "delta": d, "price": base_price * (1.0 + d)})
    return rows


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
CSV_FIELDS = [
    "case", "method", "revenue_y1", "revenue_y2", "revenue_y3",
    "net_income_y1", "net_income_y2", "net_income_y3", "target_pe",
    "target_market_cap", "target_price", "implied_return", "note",
]

EV_FORMULA = ("ev = sum(p_i * target_i); upside_pct = ev/price - 1; "
              "rr_ratio = (best - price)/(price - worst)")


def _latest_annual(facts: list[dict], metric: str, ticker: str | None = None) -> tuple[float, str]:
    cands = [
        f for f in facts
        if f.get("metric") == metric and f.get("period_type") == "annual" and f.get("value") is not None
        and (ticker is None or f.get("company_ticker") == ticker)
    ]
    if not cands:
        raise SystemExit(f"engine: no annual fact for metric '{metric}'"
                         + (f" / ticker '{ticker}'" if ticker else ""))
    best = max(cands, key=lambda f: f["period"])
    return float(best["value"]), best["period"]


def _log_entry(value: Any, reason: str | None) -> Any:
    """recompute_log convention: plain value when valid, explicit
    {"value": null, "reason": ...} when not — never NaN, never omitted."""
    return value if reason is None else {"value": None, "reason": reason}


def _round_or_blank(value: float | None, ndigits: int) -> Any:
    return round(value, ndigits) if value is not None else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", required=True)
    ap.add_argument("--assumptions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--ticker", default=None,
                    help="primary company ticker; filters facts so a multi-entity "
                         "facts file (peers/macro) doesn't mis-select a peer's revenue")
    ap.add_argument("--scenarios", default=None,
                    help="optional scenarios.json — [{name, probability, target_price}, ...] "
                         "(target_price may be omitted when name matches a computed case); "
                         "appends a probability-weighted expected_value row")
    ap.add_argument("--price", type=float, default=None,
                    help="current price for the scenario-EV upside (defaults to the "
                         "assumptions file's price)")
    args = ap.parse_args()

    import yaml  # local import so the pure functions need no deps

    facts = [json.loads(l) for l in open(args.facts) if l.strip()]
    asm = yaml.safe_load(open(args.assumptions))

    base_revenue, rev_period = _latest_annual(facts, "revenue", args.ticker)
    try:
        net_income_latest, _ = _latest_annual(facts, "net_income_parent", args.ticker)
    except SystemExit:
        # Fallback: many filers carry only `net_income` (parent NI == total NI when
        # there is no noncontrolling interest). Avoids forcing callers to mint a
        # `net_income_parent` alias just to run the engine.
        net_income_latest, _ = _latest_annual(facts, "net_income", args.ticker)
    shares = float(asm["shares_outstanding"])
    price = float(asm["price"])
    market_cap = price * shares

    scen_in = dict(asm["scenarios"])
    scen_in["_shares"] = shares
    scen_in["_price"] = price
    scen = scenario_table(scen_in, base_revenue)

    log: dict[str, Any] = {
        "engine_version": __version__,
        "inputs": {
            "base_revenue": base_revenue, "base_revenue_period": rev_period,
            "net_income_latest": net_income_latest, "shares": shares, "price": price,
            "market_cap": market_cap,
        },
        "ratios": {
            "pe_ttm": _log_entry(*pe_checked(market_cap, net_income_latest)),
            "net_margin_latest": _log_entry(*net_margin_checked(net_income_latest, base_revenue)),
        },
        "scenarios": {},
        "dcf": None,
    }

    rows = []
    for case in ("bear", "base", "bull"):
        if case not in scen:
            continue
        s = scen[case]
        reason = s.get("reason")
        log["scenarios"][case] = {
            "revenue": s["revenue"], "net_income": s["net_income"],
            "target_pe": s["target_pe"],
            "formula": "target_market_cap = net_income[y1] * target_pe; target_price = target_market_cap / shares",
            "target_market_cap": s["target_market_cap"], "target_price": s["target_price"],
            "implied_return": s["implied_return"],
            **({"reason": reason} if reason else {}),
        }
        rows.append({
            "case": case, "method": "relative",
            "revenue_y1": _round_or_blank(s["revenue"][0] if len(s["revenue"]) > 0 else None, 2),
            "revenue_y2": _round_or_blank(s["revenue"][1] if len(s["revenue"]) > 1 else None, 2),
            "revenue_y3": _round_or_blank(s["revenue"][2] if len(s["revenue"]) > 2 else None, 2),
            "net_income_y1": _round_or_blank(s["net_income"][0] if len(s["net_income"]) > 0 else None, 2),
            "net_income_y2": _round_or_blank(s["net_income"][1] if len(s["net_income"]) > 1 else None, 2),
            "net_income_y3": _round_or_blank(s["net_income"][2] if len(s["net_income"]) > 2 else None, 2),
            "target_pe": s["target_pe"],
            "target_market_cap": _round_or_blank(s["target_market_cap"], 2),
            "target_price": _round_or_blank(s["target_price"], 2),
            "implied_return": _round_or_blank(s["implied_return"], 4),
            "note": reason or "",
        })

    if "dcf" in asm and asm["dcf"]:
        d = asm["dcf"]
        dcf, dcf_reason = dcf_value_checked(
            float(d["fcf_base"]), list(d["fcf_growth"]), float(d["wacc"]),
            float(d["terminal_growth"]), float(d["net_debt"]), shares,
        )
        if dcf_reason is None:
            log["dcf"] = {"inputs": d, **dcf,
                          "formula": "PV(explicit FCF) + PV(terminal Gordon) - net_debt; per_share = equity_value/shares"}
        else:
            log["dcf"] = {"inputs": d, "value": None, "reason": dcf_reason}

    if args.scenarios:
        raw = json.load(open(args.scenarios))
        scen_list = raw["scenarios"] if isinstance(raw, dict) else raw
        resolved = []
        for s in scen_list:
            s = dict(s)
            name = s.get("name")
            if s.get("target_price") is None and name in scen \
                    and scen[name].get("target_price") is not None:
                s["target_price"] = scen[name]["target_price"]
                s["target_price_source"] = "computed scenario table"
            resolved.append(s)
        ev_price = args.price if args.price is not None else price
        res_ev, ev_reason = scenario_ev_checked(resolved, ev_price)
        if ev_reason is None:
            log["scenario_ev"] = {
                "inputs": {"scenarios": resolved, "price": ev_price},
                **res_ev, "formula": EV_FORMULA,
            }
            rr = res_ev["rr_ratio"]
            rr_txt = (f"rr_ratio={rr:.2f}" if rr is not None
                      else f"rr_ratio=n/a ({res_ev.get('rr_reason', '')})")
            note = f"probability-weighted EV of {len(resolved)} scenarios; {rr_txt}"
            rows.append({"case": "expected_value", "method": "scenario_ev",
                         "target_price": round(res_ev["ev"], 2),
                         "implied_return": round(res_ev["upside_pct"], 4),
                         "note": note})
        else:
            log["scenario_ev"] = {
                "inputs": {"scenarios": resolved, "price": ev_price},
                "value": None, "reason": ev_reason,
            }
            rows.append({"case": "expected_value", "method": "scenario_ev",
                         "note": ev_reason})

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, restval="")
        w.writeheader()
        w.writerows(rows)
    json.dump(log, open(args.log, "w"), indent=2, ensure_ascii=False)
    print(f"engine: wrote {len(rows)} cases to {args.out}; recompute log -> {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
