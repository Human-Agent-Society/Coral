# Noisy Continuous Black-Box Optimizer Design

Your task is to design a reproducible, noise-aware optimizer that minimizes 10-dimensional continuous black-box objectives with smooth periodic, multimodal structure. You inherit a weak uniform-random baseline, and every run uses the box `[-5, 5]^10` with a strict budget of 120 objective queries. Your submitted optimizer is re-run on disjoint sealed instances.

## Hard Constraints

- Edit only `/app/methods/main/`; `solver.py` must define `Optimizer`.
- Use only the Python standard library and NumPy 2.2.6.
- The required constructor is `Optimizer(dim, lower, upper, budget, seed, rng)`. No task-information argument is supplied.
- `ask(n)` must return a finite NumPy-compatible matrix with 1 through `n` rows, exactly `dim` columns, and all coordinates within the supplied bounds.
- The verifier owns the objective and query counter. Extra returned points do not increase the 120-query budget.
- Import failures, crashes, malformed output, non-finite values, and out-of-bounds proposals invalidate the complete submission.
- The submitted process cannot read or modify trusted evaluator assets and has no verifier network access.

**Runtime budget.**

Submitted optimizer code receives a 120-second aggregate soft budget across the complete sealed suite,
not a separate 120 seconds for every run. The suite contains 400 independent optimizer runs, so
each run must average about 0.3 seconds. There is no five-second scoring cutoff on every individual
`ask` or `tell` response; when the aggregate soft budget expires, the verifier stops requesting new
work and evaluates the best completed state. A stalled process can still be terminated at a sealed
safety cap. Use bounded, vectorized per-query work; repeated dense refits or hundreds-wide candidate
scans at every observation are unlikely to fit. This budget applies to sealed execution, not to your
research time.

## What You Have

- `/app/data/visible.json` contains twelve public development instances from the same noisy continuous family; sealed instances are distinct.
- `/app/methods/main/solver.py` is a uniform-random baseline.
- `/app/selfcheck.py` evaluates the same higher-is-better normalized anytime/final metric family used by the sealed evaluator on 20 deterministic runs per visible instance. It also reports diagnostic components and latent final-objective summaries.
- The supplied `rng` is `np.random.default_rng(seed)` and should drive all randomness for deterministic replay.
- The verifier repeatedly calls `ask(n)`, evaluates the returned points, and calls `tell(X, y)` (or `tell(X, y, metadata)` if accepted). Each value in `y` is an observed noisy loss, so lower is better even though the aggregate self-check score is higher-is-better.
- A positive integer `self.batch` may request a preferred batch size; the verifier negotiates and caps it to the remaining budget.

## What You Submit

Submit optimizer code, not a one-shot point or precomputed answer. The entire submitted optimizer must be self-contained in `/app/methods/main/solver.py`, which must define the required `Optimizer` class; sibling modules are not copied to the trusted verifier.

## How It Is Judged

The trusted parent evaluates your optimizer on sealed instances and fixed seeds. It records authoritative latent best-so-far traces, aggregates them robustly across seeds, and combines anytime quality with final-query quality under the same metric definition exposed by the visible self-check. Higher normalized quality is better; sealed instances, calibration assets, and evaluator internals remain hidden.
