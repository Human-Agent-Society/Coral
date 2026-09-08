# Commit terminal-current numbers for a diode lot beyond the safe test window

You are an AI agent assisting the device engineer who must qualify a new
power-diode lot at a semiconductor supplier. The probe station covers only a safe
bias/temperature window — 290–325 K, reverse bias to −3.6 V — but the
qualification sign-off due this quarter must commit leakage and conduction
numbers at mission-profile extremes: 385–400 K and reverse bias to −26 V,
where measurement is destructive and the burn-in rig is booked for months.
Overstate the lot's capability and an automotive customer eats field
returns; understate it and the socket goes to a competitor. You inherit a
starting prediction method that extrapolates a fitted window trend; your
job is to build the method the lab will run on every future lot — it is
re-run, unchanged, on qualification devices whose extreme-regime behavior
you never see.

## Hard Constraints

- Submit a **method**: `methods/main/solver.py` defining exactly
  `predict(record) -> list of floats` — the predicted `log10` of the
  absolute terminal current in amperes at each point of
  `record["extreme"]`, in order; a pure function of one device record.
- `record` is a dict with the device's safe-window measurements
  (`window`: 56 `{T, V, I}` points) and the target grid (`extreme`:
  `{T, V}` points) — the same schema as the practice records, minus the
  truth field.
- Each call runs in a **fresh process under a 180-second wall-clock
  budget** (measured outside your process). Anything that takes longer than
  **195 s** (budget + 15 s grace) is scored as the worst case for that
  device, and the process is hard-killed at **210 s** regardless; crashed or
  wrongly-shaped outputs are scored the same worst case. The grader records
  a timeout and the child's exit status separately from "produced no
  output", so both show up in the run log.
- **The whole grading stage is capped at 7200 s wall-clock**, inside a
  container declared at **4 CPUs / 1024 MiB** —
  the same shape as your own container, so a method that fits here fits
  there. Devices are graded **serially**, one subprocess at a time: with 16
  qualification devices the stage budget is the binding constraint only if
  your average device takes more than ~7.5 minutes, which the 180 s
  per-device budget already forbids.
- A note on shared memory: this container's `/dev/shm` is **64 MiB** and
  cannot be enlarged. A `multiprocessing.shared_memory` / `joblib` memmap
  segment larger than that is created successfully and then faults
  (**SIGBUS**) on first write, killing your process with no output. Pass
  large arrays to workers by fork inheritance or through ordinary files in
  `TMPDIR` instead. The grader reports a SIGBUS kill as its own failure
  reason, distinct from "produced no output".
- The graded lot is **16** devices against your **24** practice devices —
  about **0.67x** the visible count, on the identical 12-point extreme grid.
  Grading is therefore *cheaper* than one full `selfcheck.py` pass; sizing
  your method against your own self-check wall-clock is safe here.
- CPU only, no network. Runtime: Python 3 with numpy and scipy, plus the
  DEVSIM TCAD device simulator (`import devsim`) — build physics models of
  the lot if you choose to. BLAS/OpenMP threading is pinned to **4** in both
  this image and the grader, so your local timings match the graded ones.
- Only files under `methods/` are collected and re-run: keep everything
  `predict()` imports inside `methods/main/`. Do not modify `data/`,
  `examples/`, `selfcheck.py` or `run_solver.py`.

## What You Have

- `data/practice/instances.json` — 24 practice devices with **complete
  extreme-regime truth** (noise-free `log10|I|` on the full 12-point
  extreme grid), documented field by field in `data/practice/DATA_CARD.md`.
  This is your only labeled data; study it in full.
- `data/qualification/records.json` — the 16 qualification devices your
  method will actually be judged on: same window protocol, same extreme
  grid, **no truth**. They are a fresh draw from the same production
  population as the practice fleet, with part of the lot weighted toward
  the harder corner of that same population.
- `examples/devsim_diode_demo.py` — a working DEVSIM drift-diffusion model
  of this device architecture (mesh, doping, equation assembly, bias
  ramping with convergence back-off, temperature stepping, terminal
  current readout). A starting point, not a prescription.
- `methods/main/solver.py` — the inherited starting method: a log-space
  window-trend fit extrapolated in temperature with the voltage law
  clamped at the window edge. It carries a real trend signal but is far
  from what the records support; its level is also the floor you must
  clearly beat before the evaluation awards any credit.
- `python3 selfcheck.py` — free and unlimited: scores your current
  `methods/main/solver.py` on the practice fleet against its truth and
  prints per-device errors and the mean.
  **The practice mean is not a point estimate of your qualification
  score.** The 24-device practice fleet separates a broken method from a
  working one cleanly, but it is systematically *optimistic* about the
  qualification lot: for a strong method we have measured, the practice mean
  understates the qualification error by roughly a factor of **1.75**. That
  gap is not sampling noise and a better method does not make it go away —
  the two fleets are different draws and the qualification lot leans toward
  the harder corner of the population, so part of the gap is priced in by
  construction. Use the practice mean for what it is good at: a floor check,
  a debugging tool, a regression guard, and *relative* comparisons between
  two of your own variants measured on the same 24 devices, where the
  fleet-specific offset largely cancels. Do not read an absolute practice
  number as the score you will get, and do not let a few percent of practice
  advantage outweigh physics you can justify.

## What You Submit

Leave your best `methods/main/solver.py` (plus any helper files it needs
inside `methods/main/`) in place. There is no submit step and no feedback
from the graded lot: whatever sits in `methods/main/` at the end is what
the evaluation re-runs.

## How It Is Judged

The evaluation re-runs your `predict()` once per qualification device, on
records byte-identical to `data/qualification/records.json`, and compares
your numbers to the sealed truth of that lot. Per device it computes the
mean of `|log10 I_pred − log10 I_true|` over the device's extreme grid
(LOWER is better; capped at 6.0, which is also the score for an invalid or
over-budget run) — `selfcheck.py` computes the identical per-device error
on practice. Device errors are averaged within each qualification
condition (the main lot draw and the harder-corner draw from the same
population), then across the two conditions. Your reward rises monotonically
as that sealed mean error falls; at or above the shipped starting method's
error it is zero.
