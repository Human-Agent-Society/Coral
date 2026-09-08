# Drive coloring conflicts to zero at a fixed color budget

You inherit a TabuCol graph-coloring baseline. For every graph you are given a **fixed
color budget** `target_k`. Assign each vertex a color in `{0, 1, ..., target_k-1}` so as
to **minimize the number of monochromatic edges** (edges whose two endpoints share a
color). You may not use any color `>= target_k`.

`target_k` is set to each graph's **best-known chromatic number**, so a conflict-free
(proper) `target_k`-coloring **provably exists** — but finding one sits at the coloring
frontier and is very hard: a reference local search plateaus well above zero. Your job is
to get the conflict count as **low as possible** — ideally all the way to 0.

This datapoint is **method-scored**, not answer-scored: your code is re-run by a separate
verifier on sealed hidden graphs. The visible graphs are only for development.

## Hard Constraints

- You may only edit code under `/app/methods/main/`; you may add sibling `.py` modules.
- Keep the entrypoint signature `solve(data) -> {"colorings": [...]}`.
- **Standard library + `numpy` only** — no internet, no other third-party packages. The
  verifier sandbox ships the SAME `numpy` and nothing else, so `import numpy` is fine but
  any other third-party import (networkx, scipy, a solver package, …) makes the submission
  score 0.
- **Per-instance 45s hard cap.** The grader solves each graph in isolation and **kills any
  graph that runs longer than 45 seconds**, scoring that graph 0 (the others are
  unaffected). Self-pace so every graph returns within 45s.
- Return one coloring per instance, in the same order as `data["instances"]`. Each coloring
  must have length `n_vertices`; **every color must be an integer in `[0, target_k)`**. A
  color `>= target_k` or `< 0`, a wrong length, a non-integer, or a graph that exceeds 45s
  scores 0 **for that graph** (per-instance fail-closed), not for the whole submission.

## What You Have

- `/app/data/visible.json`: visible graphs with anonymized names `pub0`, `pub1`, … . Each is
  `{name, n_vertices, n_edges, edges, target_k}` with 0-indexed undirected edges.
- `/app/methods/main/solver.py` and `gc_lib.py`: the editable baseline (a single TabuCol run)
  and helpers, including `reference_tabucol`. **This directory is what gets graded.** Matching
  the baseline gains you nothing; the task is to beat it.
- `/app/selfcheck.py`: a free local dry-run (`python /app/selfcheck.py`) that **mirrors the
  grader** — runs your solver on each visible graph under the SAME 45s per-instance cap,
  validates the coloring, prints each graph's runtime + raw conflict count, and
  tells you exactly which graphs were KILLED for exceeding 45s. A proxy only — the hidden set
  is a different, sealed batch.

## What You Submit

Edit `/app/methods/main/solver.py`, keeping this interface:

```python
def solve(data):
    # data: {"instances": [{"name","n_vertices","n_edges","edges","target_k"}, ...], ...}
    # returns: {"colorings": [[color_for_vertex_0, ...], ...]}   # colors in [0, target_k)
    ...
```

You may add helper modules next to `solver.py`. There is no submit step; Harbor grades
whatever remains under `/app/methods/main/` at the end of the run.

## How It Is Judged

After your run, the verifier copies your `methods/main/` into a clean, no-network sandbox and
re-runs `solve` on sealed hidden graphs drawn from the **same generators** as the visible
set (n from 250 to 1000), **one graph at a time
under the 45s per-instance cap**. For each graph it checks the color range and counts
conflicting edges — fewer is better. A crash, out-of-range color, wrong shape, or a graph over
45s forfeits that graph, so a constant or hard-coded answer cannot score.

The hidden graphs have **randomly permuted vertex labels** and anonymized names. Do not rely
on graph names, vertex-index patterns, downloaded public instances, or a precomputed coloring
table: a published best-known coloring is keyed to upstream labels and does not
apply here. The task is to build a robust conflict-minimization method that transfers from the
visible to the hidden graphs.
