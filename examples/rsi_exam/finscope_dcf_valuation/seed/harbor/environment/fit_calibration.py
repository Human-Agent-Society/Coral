#!/usr/bin/env python3
"""Free local dry-run: estimate a forecaster's valuation error before submitting.

Usage:
    python environment/fit_calibration.py [methods/main]          # leave-one-out (default)
    python environment/fit_calibration.py [methods/main] --kfold 6  # faster k-fold, for slow methods

The graded held-out truth is private, so this script gives you an HONEST proxy
using only the VISIBLE calibration companies. It rotates a small check set out of
the calibration companies, lets your forecaster fit on the REST (which it sees as
'calibration', with truth) and forecast the held-back names (which it sees as
'holdout'), values those forecasts with the SAME DCF engine the desk uses, and
reports the median absolute valuation error against the known truth.

IMPORTANT — this now uses LEAVE-ONE-OUT by default (fit on N-1 calibration names,
forecast the 1 held-back, rotated over all N). That matches the GRADING regime,
where your method fits on ALL visible calibration truth and then forecasts the
sealed holdout. A large fit set is what activates any full-calibration model path
in your forecaster; the old 50/50 split fit on only half the names, so methods
that only kick in with enough calibration truth looked frozen here even when they
improved at grading. Pass --kfold K to trade a little fidelity for speed (K fits
instead of N) if your forecaster is slow.

It does NOT call the grader and reveals no graded number; it only exercises your
method on visible data.
"""
from __future__ import annotations

import os
import sys
import statistics

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "lib"))
import paneldata  # noqa: E402
import valuation as VAL  # noqa: E402

APE_CAP = 2.0


class _FoldView:
    """A Problem view that exposes the fit-half calibration as 'calibration'
    (with truth) and the check-half as 'holdout' (truth withheld), so an
    unmodified forecaster fits and forecasts exactly as it would at grading."""

    def __init__(self, base: "paneldata.Problem", check_ids: set[str]) -> None:
        self.horizon = base.horizon
        self.driver_fields = base.driver_fields
        self._truth = base.load_calibration_truth()
        self._check_ids = check_ids
        self._companies = []
        for c in base.load_companies():
            if c["set"] == "calibration":
                c = dict(c)
                c["set"] = "holdout" if c["id"] in check_ids else "calibration"
                self._companies.append(c)
            # real holdout companies are dropped from this visible-only view
        self.calibration_ids = [c["id"] for c in self._companies if c["set"] == "calibration"]
        self.holdout_ids = [c["id"] for c in self._companies if c["set"] == "holdout"]

    def load_companies(self):
        return self._companies

    def companies_by_id(self):
        return {c["id"]: c for c in self._companies}

    def load_calibration_truth(self):
        return {cid: rec for cid, rec in self._truth.items() if cid not in self._check_ids}


def _folds(calib_ids: list[str], kfold: int | None) -> list[set[str]]:
    """Return the list of check-id sets. kfold=None -> leave-one-out (N folds of 1);
    kfold=K -> K contiguous-stride folds (fit on the rest)."""
    if kfold is None or kfold >= len(calib_ids):
        return [{cid} for cid in calib_ids]
    return [set(calib_ids[i::kfold]) for i in range(kfold)]


def main() -> int:
    argv = [a for a in sys.argv[1:]]
    kfold = None
    if "--kfold" in argv:
        i = argv.index("--kfold")
        kfold = int(argv[i + 1])
        del argv[i:i + 2]
    method_dir = argv[0] if argv else os.path.join(_HERE, "methods", "main")
    method_dir = os.path.abspath(method_dir)
    sys.path.insert(0, method_dir)
    import importlib.util
    spec = importlib.util.spec_from_file_location("forecaster", os.path.join(method_dir, "forecaster.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    base = paneldata.Problem()
    calib_ids = sorted(base.calibration_ids)
    full_truth = base.load_calibration_truth()
    folds = _folds(calib_ids, kfold)

    apes: list[float] = []
    misses: list[str] = []
    min_fit = len(calib_ids)
    for check_ids in folds:
        min_fit = min(min_fit, len(calib_ids) - len(check_ids))
        view = _FoldView(base, check_ids)          # fit-half = calibration (with truth), check-half = holdout
        drivers = mod.solve(view)
        for cid in sorted(check_ids):
            if cid not in drivers:
                misses.append(cid)
                continue
            errs = VAL.validate_drivers(drivers[cid])
            if errs:
                print(f"  {cid}: INVALID drivers -- {errs[0]}")
                apes.append(APE_CAP)
                continue
            facts = full_truth[cid]["facts"]
            vhat, _ = VAL.intrinsic_value(drivers[cid], facts)
            vstar = full_truth[cid]["intrinsic_value"]
            ape = APE_CAP if vhat is None else min(abs(vhat - vstar) / abs(vstar), APE_CAP)
            apes.append(ape)

    mode = "leave-one-out" if kfold is None else f"{kfold}-fold"
    print(f"method: {method_dir}")
    print(f"{mode}: fit on ~{min_fit} calibration names each fold (grading fits on all "
          f"{len(calib_ids)}), checked out-of-sample on all {len(calib_ids)} names")
    if misses:
        print(f"  WARNING: forecaster returned no drivers for {len(misses)} check names "
              f"(at grading, missing ids score the whole submission invalid)")
    if apes:
        print(f"  median abs valuation error (out-of-sample, visible): "
              f"{statistics.median(apes):.4f}  (lower is better)")
        print(f"  mean: {sum(apes)/len(apes):.4f}   worst: {max(apes):.4f}")

    # Budget reminder. This task has no selfcheck.py; instruction.md names
    # `python /app/fit_calibration.py` as the self-check entry point. Skipped entirely when
    # /app/budget.py is not mounted.
    import os as _os, subprocess as _sp, sys as _sys
    if _os.path.exists("/app/budget.py"):
        _sys.stdout.flush()          # without this the reminder prints before the score in a pipe
        _sp.run([_sys.executable, "/app/budget.py"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
