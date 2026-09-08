"""methods/main — the desk's current first-pass driver model (sector reversion).

This is the model the grader runs (the score-0 baseline). It is the prior
analyst's standing method: for each company it reverts every forward driver from
the company's own clean-latest observation toward the cross-sectional SECTOR-MEAN
realized path (learned from the visible calibration set), with a fixed
own/sector blend; the discount rate is a flat beta-linear rule and terminal
growth is the panel mean capped below WACC. It is a real, defensible method --
it does use the calibration panel and reverts toward sector behaviour -- but a
DELIBERATELY WEAK one: it applies ONE blend weight to every name and horizon, it
does not fit a per-name cross-sectional model, it does not exploit history
length / IPO age / special-items structure, and it takes a crude flat WACC. A
proper cross-sectional fit (see a sibling method, or build your own) beats it.

Contract: solve(problem) -> {company_id: drivers} for every holdout company.
The grader hands you ``problem``; do not read data files here.
"""
from __future__ import annotations

H = 5
_OWN_WEIGHT = 0.35            # own clean-latest weight; the rest reverts to sector mean
_SECTOR_GROWTH_PRIOR = 0.06
_SECTOR_MARGIN_PRIOR = 0.18


def _clean_latest(history, field, default):
    """Most recent non-missing, non-special-items observation of ``field``."""
    for row in reversed(history):
        if row.get(field) is not None and not row.get("special_items_flag"):
            return row[field]
    for row in reversed(history):
        if row.get(field) is not None:
            return row[field]
    return default


def _mean(xs, default=0.0):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else default


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def solve(problem) -> dict:
    companies = problem.load_companies()
    by_id = {c["id"]: c for c in companies}
    truth = problem.load_calibration_truth()
    fit_ids = [cid for cid in truth if cid in by_id]
    sectors = sorted(set((c.get("sector") or "UNK") for c in companies))

    # cross-sectional SECTOR-MEAN realized path per driver (from calibration truth)
    sec_path = {f: {s: [[] for _ in range(H)] for s in sectors}
                for f in ("rev_growth", "ebit_margin", "reinvest")}
    glob_path = {f: [[] for _ in range(H)] for f in ("rev_growth", "ebit_margin", "reinvest")}
    for cid in fit_ids:
        s = by_id[cid].get("sector") or "UNK"
        rd = truth[cid]["realized_drivers"]
        for f in ("rev_growth", "ebit_margin", "reinvest"):
            for t in range(H):
                sec_path[f][s][t].append(rd[f][t])
                glob_path[f][t].append(rd[f][t])

    def sec_mean(field, sector, t):
        vals = sec_path[field][sector][t]
        return _mean(vals) if vals else _mean(glob_path[field][t], 0.1)

    tg_all = _mean([truth[cid]["realized_drivers"]["terminal_growth"] for cid in fit_ids], 0.025)

    out: dict = {}
    for cid in problem.holdout_ids:
        c = by_id[cid]
        sector = c.get("sector") or "UNK"
        hist = c["history"]
        rec: dict = {}
        for field, dflt in (("rev_growth", _SECTOR_GROWTH_PRIOR),
                            ("ebit_margin", _SECTOR_MARGIN_PRIOR),
                            ("reinvest", 0.45)):
            now = _clean_latest(hist, field, dflt)
            rec[field] = [_clip(_OWN_WEIGHT * now + (1.0 - _OWN_WEIGHT) * sec_mean(field, sector, t),
                                -0.30, 0.90) for t in range(H)]
        beta = c.get("beta_proxy")
        wacc = _clip(0.055 + 0.035 * (beta if beta is not None else 1.1), 0.06, 0.13)
        tg = _clip(min(tg_all, wacc - 0.02), 0.010, wacc - 0.005)
        rec["wacc"] = wacc
        rec["terminal_growth"] = tg
        out[cid] = rec
    return out
