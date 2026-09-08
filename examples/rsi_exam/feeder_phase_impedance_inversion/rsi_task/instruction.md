# Repair a feeder's phase and impedance records from smart-meter archives

You are an AI agent assisting a distribution planning engineer at a mid-size
utility with a solar interconnection queue. Before any hosting-capacity or voltage-drop
study can be signed off, the feeder model has to be calibrated: the GIS
says which phase each customer is on, which conductor each segment uses,
and where the regulator tap sits — and all three are wrong at
well-documented rates, because crews re-phase laterals after storms,
re-conductor spans without closing the paperwork, and move taps in the
field. All you have per feeder is two to three days of 15-minute AMI
data from most (not all) customers — voltage magnitude only — plus head
SCADA and billing averages. A model calibrated too optimistically
overstates hosting capacity and customers get voltage complaints; too
pessimistically and interconnections are refused for no reason. You
inherit the utility's current first-pass script; your job is to build
the calibration method the utility will run fleet-wide — it is re-run,
unchanged, on feeders whose actual wiring you never see.

## Hard Constraints

- Submit a **method**: `methods/main/solver.py` defining exactly
  `calibrate(record) -> dict` with keys `"phase"` (list of one int in
  0/1/2 per customer), `"code"` (list of one conductor class per
  segment, from the record's catalog), `"scale"` (list of one positive
  impedance multiplier per segment), and `"tap_steps"` (number, in
  regulator steps). Optionally add `"z_rx"`: a list of
  `[R1_ohm, X1_ohm]` per segment, which then overrides `code`/`scale`
  in the impedance part of the score. A pure function of one feeder
  record.
- `record` is a dict with the feeder's network records and its AMI
  monitoring window — the same schema as the practice records, minus
  the truth field. Field-by-field documentation:
  `data/practice/DATA_CARD.md`.
- Each call runs in a **fresh process under a 300-second wall-clock
  budget** (measured outside your process, with a 15 s grace and a hard
  kill at 330 s); over-budget, crashed or malformed outputs are scored
  as the worst case for that feeder.
- **Evaluation budget as a whole**: the evaluation runs your method once
  per assessment feeder, serially, inside a **10800-second wall-clock
  cap** for the entire scoring pass, on **2 CPUs and 512 MB of memory**.
  The assessment fleet is **16 feeders — 1.33× the 12 practice feeders
  you have** — and its members are on average larger (more customers and
  more segments) than the practice ones. Budget your per-feeder cost
  accordingly: 16 feeders that each burn the full 300 s do fit, but
  nothing beyond that does.
- CPU only, no network. Runtime: Python 3 with numpy, scipy and the
  OpenDSS engine (`import opendssdirect`); you may run as many forward
  power-flow solves as the budget allows. Your own container has
  **2 CPUs / 1024 MB**. `/dev/shm` is only **64 MiB** on both sides, so a
  `multiprocessing` / `joblib` design that passes large arrays through
  POSIX shared memory will create the segment successfully and then die
  with SIGBUS on the first write — size any such buffer accordingly.
  BLAS/OpenMP thread counts are pinned to the
  same constant on both sides, so a timing you measure with
  `selfcheck.py` transfers to the evaluation.
- Only files under `methods/` are collected and re-run: keep everything
  `calibrate()` imports inside `methods/main/`. Do not modify `data/`,
  `selfcheck.py` or `run_solver.py`.
- Return plain python types (ints/floats/lists); numpy arrays are fine for `z_rx` rows, but
  every value must be finite and `code` entries must come from the record's catalog. A
  malformed return is scored as the worst case for that feeder.

## What You Have

- `data/practice/instances.json` — 12 practice feeders with **complete
  as-operated truth** (actual per-customer phases, per-segment conductor
  classes and impedance multipliers, actual tap), documented in
  `data/practice/DATA_CARD.md`. This is your only labeled data; study it
  in full.
- `data/assessment/records.json` — the 16 assessment feeders your
  method will actually be judged on: same monitoring protocol, **no
  truth**. They are unseen feeders drawn from the same population as
  the practice fleet (a subset weighted toward its hard end: the
  largest networks and longest segments the population produces).
- `methods/main/solver.py` — the inherited starting method: voltage-
  trace similarity grouping for metered phases, records-as-shipped
  impedances, and a tap estimate from the average voltage offset of an
  OpenDSS replay (`methods/main/dss_forward.py` is its forward engine;
  `python3 methods/main/dss_forward.py` runs a round-trip demo). It
  carries a real signal but is far from what the records support; its
  level is also the floor you must clearly beat before the evaluation
  awards any credit.
- `python3 selfcheck.py` — free and unlimited: scores your current
  `methods/main/solver.py` on the practice fleet against its truth and
  prints per-feeder component scores and the mean.

## What You Submit

Leave your best `methods/main/solver.py` (plus any helper files it
needs inside `methods/main/`) in place. There is no submit step and no
feedback from the assessment fleet: whatever sits in `methods/main/` at
the end is what the evaluation re-runs.

## How It Is Judged

The evaluation re-runs your `calibrate()` once per assessment feeder,
on records byte-identical to `data/assessment/records.json`, and
compares your output to the sealed as-operated truth. Per feeder it
computes a score in [0, 1] (HIGHER is better): **0.45 × phase score**
(label accuracy over metered customers, rescaled so that copying the
shipped GIS labels scores 0 and perfect labeling scores 1) **+ 0.40 ×
impedance score** (how much your cumulative substation-to-customer
positive-sequence R and X — the quantity a voltage-drop study consumes —
improves on the records as shipped, measured by log-error norm and
clipped to [0, 1]) **+ 0.15 × tap score** (1 − |tap step error| / 4,
clipped). Returning the records as shipped scores exactly 0 —
`selfcheck.py` computes the identical per-feeder score on practice.
Feeder scores are averaged within each assessment group (`f00`–`f07`
and `f08`–`f15`), then across the two groups. Your reward rises
monotonically with that sealed mean; at or below the shipped starting
method's level it is zero.
