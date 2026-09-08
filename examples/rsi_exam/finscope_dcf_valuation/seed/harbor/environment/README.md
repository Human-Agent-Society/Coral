# FinScope coverage desk — workspace

You have inherited the desk's **forward value-driver model**. The DCF engine is
fixed and trusted; the work is forecasting good drivers for the coverage universe.
This directory is the project as your predecessor left it.

## Layout

```
environment/
  README.md                 this file
  data/
    coverage.csv            per-company as-of facts + metadata (ALL names)
    fundamentals.csv        observable per-year history (ALL names), MESSY
    calibration_truth.csv   realized drivers + true value (calibration names only)
    data_dictionary.md      what every column means + the data conventions — READ IT
  lib/
    valuation.py            the public DCF bridge: intrinsic_value(drivers, facts)
    fin_compute_engine.py   the discounting engine valuation.py delegates to
    paneldata.py            loads data/ into the `problem` object your solver gets
  methods/
    main/forecaster.py            <- the graded method; the current first-pass model
    own_history_trend/forecaster.py   a second, different stab (per-company trend)
  notes/
    analyst_notebook.md     your predecessor's exploration + open questions
  TODO.md                   leftover leads (some good, some half-baked)
  fit_calibration.py        free local dry-run: estimate your error before submitting
```

## The contract

`methods/main/forecaster.py` defines `solve(problem) -> {company_id: drivers}` for
every id in `problem.holdout_ids`. The grader hands you `problem`; `solve` does not
read files. You may add helper modules inside a method directory, restructure it,
or build a new method dir and point your work there — the whole directory runs.

`problem` exposes: `problem.holdout_ids`, `problem.load_companies()` (each a dict
with `id` / `set` / `sector` / `beta_proxy` / `facts` / `history` / `ipo_year` …),
`problem.companies_by_id()`, and `problem.load_calibration_truth()` (realized
drivers + true value for the calibration names — your fitting data).

## Develop

```bash
python fit_calibration.py methods/main      # out-of-sample error on visible names
```

Build against the calibration names (where the truth is known), then trust the
method on the rest. Matching the calibration names alone is not the point — the
grade is on names you never see, so what you want is a method that *generalizes*.

Two starting methods ship here; neither is finished. Compare them, improve one, or
write a better one. The notebook and `TODO.md` are where the open threads are.
