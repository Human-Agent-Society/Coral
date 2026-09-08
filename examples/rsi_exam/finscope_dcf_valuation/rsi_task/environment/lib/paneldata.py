"""Agent-visible data access for finscope_dcf_valuation.

Reads the observable fundamentals panel (``coverage.csv`` + ``fundamentals.csv``)
and the visible calibration truth (``calibration_truth.csv``) into a single
read-only ``Problem`` handle that your forecaster receives. The panel is
deliberately MESSY (variable-length history, missing cells, special-items
outlier years, missing sector / beta); see ``data/data_dictionary.md`` for what
every column and convention means — handling that mess is part of the task.

A blank CSV cell is parsed to ``None`` (NOT zero): a missing observation is not
the same as a zero one. ``special_items_flag == 1`` marks a fiscal year whose
``rev_growth`` / ``ebit_margin`` are distorted by a one-off event (M&A,
divestiture, accounting one-timer) and do not reflect the underlying trend.

Your forecaster is HANDED a ``Problem`` by the grader; it does NOT read these
files itself at grading time. Use ``Problem`` here only for local development /
self-check (see ``fit_calibration.py``).
"""
from __future__ import annotations

import csv
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(os.path.dirname(_HERE), "data")   # environment/data

HORIZON = 5            # explicit forecast years per company
DRIVER_FIELDS = ("rev_growth", "ebit_margin", "reinvest", "wacc", "terminal_growth")
_FACT_FIELDS = ("revenue_0", "net_debt", "shares", "tax_rate")


def _f(x):
    """Parse a CSV cell to float, or None when blank/unparseable."""
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _i(x):
    v = _f(x)
    return int(v) if v is not None else None


def _read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def build_companies(coverage_rows, fundamentals_rows):
    """Assemble the per-company observable records from the two raw CSV tables.

    Shared by the workspace ``Problem`` and the sealed grading harness so an
    unmodified forecaster sees an identical record shape in both."""
    hist = {}
    for r in fundamentals_rows:
        hist.setdefault(r["company_id"], []).append({
            "fiscal_year": _i(r["fiscal_year"]),
            "rev_growth": _f(r["rev_growth"]),
            "ebit_margin": _f(r["ebit_margin"]),
            "reinvest": _f(r["reinvest"]),
            "special_items_flag": _i(r["special_items_flag"]) or 0,
        })
    for rows in hist.values():
        rows.sort(key=lambda d: (d["fiscal_year"] is None, d["fiscal_year"]))

    companies = []
    for c in coverage_rows:
        cid = c["company_id"]
        companies.append({
            "id": cid,
            "set": c["set"],
            "sector": c["sector"] or None,
            "beta_proxy": _f(c["beta_proxy"]),
            "fy_end_month": _i(c["fy_end_month"]),
            "ipo_year": _i(c["ipo_year"]),
            "first_fiscal_year": _i(c["first_fiscal_year"]),
            "last_fiscal_year": _i(c["last_fiscal_year"]),
            "facts": {
                "revenue_0": _f(c["revenue_0"]),
                "net_debt": _f(c["net_debt"]),
                "shares": _f(c["shares"]),
                "tax_rate": _f(c["tax_rate"]),
            },
            "history": hist.get(cid, []),
        })
    return companies


def parse_calibration_truth(rows, facts_by_id):
    """{cid: {realized_drivers, intrinsic_value, facts}} from calibration_truth rows."""
    out = {}
    for r in rows:
        cid = r["company_id"]
        drivers = {
            "rev_growth": [_f(r[f"rev_growth_{t+1}"]) for t in range(HORIZON)],
            "ebit_margin": [_f(r[f"ebit_margin_{t+1}"]) for t in range(HORIZON)],
            "reinvest": [_f(r[f"reinvest_{t+1}"]) for t in range(HORIZON)],
            "wacc": _f(r["wacc"]),
            "terminal_growth": _f(r["terminal_growth"]),
        }
        out[cid] = {"realized_drivers": drivers,
                    "intrinsic_value": _f(r["intrinsic_value"]),
                    "facts": facts_by_id.get(cid, {})}
    return out


class Problem:
    """Read-only handle to the agent-visible panel and the task contract."""

    def __init__(self, data_dir: str | None = None) -> None:
        d = data_dir or _DATA
        self._coverage = _read_csv(os.path.join(d, "coverage.csv"))
        self._fundamentals = _read_csv(os.path.join(d, "fundamentals.csv"))
        self._calib_rows = _read_csv(os.path.join(d, "calibration_truth.csv"))
        self.horizon = HORIZON
        self.driver_fields = DRIVER_FIELDS
        self._companies = build_companies(self._coverage, self._fundamentals)
        self.calibration_ids = [c["id"] for c in self._companies if c["set"] == "calibration"]
        self.holdout_ids = [c["id"] for c in self._companies if c["set"] == "holdout"]
        self.submission_contract = (
            "solve(problem) -> {company_id: {rev_growth:[5], ebit_margin:[5], "
            "reinvest:[5], wacc: float, terminal_growth: float}} for every id in "
            "problem.holdout_ids. terminal_growth must be < wacc."
        )

    def load_companies(self) -> list[dict]:
        return self._companies

    def companies_by_id(self) -> dict:
        return {c["id"]: c for c in self._companies}

    def load_calibration_truth(self) -> dict:
        facts = {c["id"]: c["facts"] for c in self._companies}
        return parse_calibration_truth(self._calib_rows, facts)
