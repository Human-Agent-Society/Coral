# Sparse-Gauge Tidal Friction Field Inversion

Design a calibration method for the spatially varying seabed friction field of a tidal basin, starting from a deliberately weak homogeneous-friction solver. Minimize complex tidal-elevation RMSE at unobserved gauges; your submitted method is re-run on sealed basins and sealed gauges for scoring.

## Hard Constraints

- Modify only `/app/methods/main/solver.py` as the final submission.
- Keep the public function `estimate_logf(case)` and return one finite `numpy` array with shape `(24, 24)`.
- Use only information in `case`; do not read validation heads inside the solver.
- The returned log-friction field must stay in `[-3.0, 1.0]`.
- A solver call must finish within 60 seconds on two CPU cores and use no network.

## What You Have

- `/app/problem.py`: the frequency-domain shallow-water tidal simulator (`solve_field` / `simulate`) and case schema.
- `/app/data/visible_cases.npz`: six calibration cases, each with known bathymetry, the forced open-boundary tide, sparse observed-gauge complex elevations, and separate validation gauges.
- `/app/methods/main/solver.py`: a homogeneous weak baseline.
- `/app/selfcheck.py`: runs the solver on all visible cases and reports validation-gauge elevation RMSE only.

The inverse problem is deliberately ill-posed: friction is a spatial field, the elliptic tidal response smooths away its fine structure, and gauges are sparse, so many friction fields fit the observed gauges while disagreeing elsewhere. Additionally, **a fraction of the observed gauges are faulty** — they carry gross, non-Gaussian errors unrelated to the true tide — and which gauges are faulty is not disclosed. The separate validation gauges are clean.

## What You Submit

Submit `/app/methods/main/solver.py`. It should infer a full log-friction field from each case's bathymetry, boundary forcing, observed gauge locations, and observed complex elevations. You may implement optimization, basis design, regularization, adjoint/gradient methods, ensembling, or other numerical methods inside that file.

## How It Is Judged

The verifier imports your solver in a restricted child process, gives it new anonymized cases without validation heads, and uses the trusted simulator to compute complex tidal-elevation RMSE at sealed gauges. Lower RMSE is better; the score increases monotonically after clearing the disclosed weak-baseline gate of `0.14`, and beating the reference solver is the goal.
