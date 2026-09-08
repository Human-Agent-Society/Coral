# FinScope coverage desk — forward value-driver model

You have inherited a buy-side equity-research desk's **intrinsic-value model**. The desk
values companies with a fixed, trusted discounted-cash-flow (DCF) engine that turns a set of
**forward value-drivers** into a per-share value; the open problem is **forecasting those
drivers** for every name in the coverage universe. You inherit a deliberately weak first-pass
model (a sector-reversion blend) plus a second weak stab and a few notes — improve one or write
a better method. Your forecaster is re-run on a **sealed held-out universe** whose true drivers
you never see, and scored by how close the DCF values it implies land to the truth: lower error
is better.

## Hard Constraints

- Submit an **algorithm** (`solve(problem)`), not precomputed drivers — the grader re-runs your
  code on hidden companies.
- Keep the exact signature `solve(problem) -> {company_id: drivers}`, returning a record for
  **every** id in `problem.holdout_ids` and **no others**.
- Each `drivers` record must be well-formed and in range: `rev_growth`, `ebit_margin`,
  `reinvest` are 5-element lists; `wacc` and `terminal_growth` are scalars; and
  `terminal_growth < wacc` (the Gordon terminal requires it). A malformed / out-of-range /
  wrong-id-set submission scores 0.
- `solve` is handed a `problem` object; it must **not** read data files or look up answers — the
  held-out truth is sealed in the grader and there is no network at grade time.
- **Grading budget** (stated so you can size your method). Your `solve` is run
  **once**, in a fresh subprocess capped at a **1200 s** wall-clock budget, inside a container
  declared at **2 CPUs / 512 MiB**; the whole verifier stage is capped
  at **3600 s**. Your own agent container is the same size (2 CPUs / 512 MiB). Relative to your
  own self-check the grading side is **about 0.1x the wall clock** (measured end-to-end at
  2 CPUs / 512 MiB: 4.85 s of `fit_calibration.py` versus ~0.5 s for the whole grading stage) —
  grading is *cheaper* because the self-check refits your method 60 times (leave-one-out over
  the 60 calibration names) while grading calls `solve` exactly once. Do not read that as
  slack in the other direction: that single call must produce drivers for **4x as many
  companies** (240 held-out versus 60 calibration names), so a method whose cost grows with the
  universe size sees 4x the work in one shot. A run killed by the time or memory cap produces
  no drivers and scores 0; the grader records the timeout and the subprocess's exit status
  separately from "produced no output", so both show up in the run log.

## What You Have

- The workspace `/app/` as your predecessor left it (`/app/README.md` orients you):
  - Data (`/app/data/`): `coverage.csv` (per-company as-of facts + metadata), `fundamentals.csv`
    (the **messy** observable per-year history), `calibration_truth.csv` (realized drivers + true
    value for the **calibration** names only), and `data_dictionary.md`. **Read the dictionary** —
    histories vary in length, cells are missing (MNAR), some fiscal years carry one-off
    `special_items`, and `sector` / `beta_proxy` are sometimes blank; how you treat the mess is a
    large part of the task.
  - The public DCF bridge (`/app/lib/`): `valuation.py` / `fin_compute_engine.py` value a driver
    record with the *same* mechanics the grader uses (you may import them for your own checks);
    `paneldata.py` builds the `problem` object.
  - The editable baselines (`/app/methods/`): `main/forecaster.py` — **this directory is what gets
    graded** — is a weak sector-reversion blend; `own_history_trend/` is a second, different weak
    stab. Improve one in place, or rewrite the method entirely.
- **Your self-check surface** (free, unlimited): `python /app/fit_calibration.py /app/methods/main`
  refits your method leave-one-out over the visible calibration names — fit on the other 59, forecast
  the one held back, rotated over all 60 — and reports the median absolute out-of-sample valuation
  error. Pass `--kfold K` to spend K fits instead of 60 if your method is slow. This matches the
  grading regime, where your method fits on *all* visible calibration truth before forecasting the
  sealed names. This is your *only* feedback loop, and it is a **proxy**: the sealed grade
  is on other companies, so a low calibration error is necessary but not sufficient.

## What You Submit

Edit `/app/methods/main/forecaster.py` to expose this exact signature:

```python
def solve(problem) -> dict:   # {company_id: drivers} for every id in problem.holdout_ids
```

where each `drivers` record is:

```python
{"rev_growth": [g1, g2, g3, g4, g5],   # revenue growth per year (0.12 = +12%)
 "ebit_margin": [m1, m2, m3, m4, m5],  # EBIT margin per year    (0.25 = 25%)
 "reinvest":    [r1, r2, r3, r4, r5],  # reinvestment as a fraction of NOPAT, in [0,1)
 "wacc": 0.09,                         # discount rate, in (0,1)
 "terminal_growth": 0.025}             # long-run growth; MUST be < wacc
```

`problem` exposes `holdout_ids`, `load_companies()` (each a dict with `id` / `set` / `sector` /
`beta_proxy` / `facts` / `history` / `ipo_year` / …), `companies_by_id()`, and
`load_calibration_truth()` (realized drivers + true value for the calibration names — your
fitting data). You may add helper modules next to `forecaster.py` or restructure the method
entirely; the whole `methods/main/` directory runs. The runtime provides `numpy`. **There is no
submit step and no per-attempt feedback** — self-check for as long as your run window allows, then
leave your best `forecaster.py` in place; it is graded once at the end on the hidden universe.

## How It Is Judged

After your run, the grader runs your `solve` on the sealed held-out companies' **observable**
panel (no truth), then recomputes — with its own sealed engine — each company's intrinsic value
`V_hat` from your drivers and the perfect-foresight value `V*` from its hidden realized drivers,
and scores the panel **median absolute valuation error**:

```
MAVE = median over a sealed TEST subset of the holdout companies of
           |V_hat - V*| / V*                                        (per-company error capped at 2.0)
```

You must still return a record for **every** id in `holdout_ids` — a missing (or extra) id is a
structural violation and scores 0 — but only the sealed `test` tier counts toward the reward; the
rest is a held-back `dev` tier you never see the composition of. So you cannot tune to the graded
subset, and a solver that works on only part of the universe is not safe.

The metric is **MAVE (lower is better)**; push it as low as you can. Only the submitted *drivers*
are trusted — both `V_hat` and `V*` are recomputed by the grader's engine, so a reported value
cannot be gamed.
