# Multi-view point-cloud reconstruction post-processing

Improve a Python post-processor that combines noisy three-dimensional observations from several views into one clean scene cloud. The objective is low geometric error on unseen cases while retaining broad surface coverage and fine detail.

## Hard Constraints

- Edit only `/app/methods/main/solver.py`, keeping `predict(export_dir)` as the entry point.
- Use the Python standard library and NumPy already installed in the image. Do not access the network, launch external programs, read files outside the supplied observation directory, or read verifier-owned files.
- Read every case from the supplied manifest. Case identifiers, case counts, view identifiers, view counts, and point counts all differ between the visible and sealed packages; nothing about them may be assumed or hard-coded.
- Return exactly one cloud per manifest case: 4 to 2,048 finite three-dimensional points, at least four of them distinct. Missing keys, extra keys, empty clouds, malformed points, and non-finite values are rejected outright and score nothing. Exact duplicate points are collapsed before scoring, so padding a cloud with copies of a good point neither helps nor hurts.
- `predict` must be deterministic for identical input and must not modify any input file.
- Private reference geometry is not present when `predict` runs.

## What You Have

- `/app/public/observations/` — the visible package: a manifest and the case files it references, in the same schema as the sealed one. Each view carries `points` and a same-length `confidence` in `[0, 1]`.
  **The views of a case are not in a common coordinate system.** The first view listed for a case is the reference view; its points are already in the scene frame. Every other view is expressed in its own frame, related to the scene frame by an unknown rigid transform — a small rotation and a small translation, no scale and no reflection. Recovering those transforms is part of the problem.
- `/app/methods/main/solver.py` — the editable starter.
- `/app/selfcheck.py` — checks the submission contract only: structure, coverage, finiteness, determinism, input immutability, standard-library-only imports. **It contains no reference geometry, does not measure quality, and cannot predict sealed performance.**

## What You Submit

The edited `/app/methods/main/solver.py`, exposing:

```python
def predict(export_dir: str) -> dict:
    ...
```

mapping each manifest `case_id` to its point cloud.

## How It Is Judged

The sealed evaluator invokes `predict` on private cases with the same schema, first validating case coverage, cloud shape, point-count bounds, finiteness, distinct-point coverage, and input immutability.

Private reference geometry is expressed in the scene frame — the frame of each case's reference view — so a returned cloud is compared in that frame and is not aligned to the truth before scoring.

For each valid case it computes symmetric precision/recall F-scores from nearest-neighbour distances at three fixed geometric tolerances (`0.0125`, `0.025`, `0.05`) and averages the three. Being symmetric, the score charges both for points that miss the surface and for surface that no point covers. The per-case scores are averaged and that single mean is calibrated once; higher is better.
