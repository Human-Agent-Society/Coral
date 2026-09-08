# Plan the contraction order for exact decision-diagram circuit equivalence checking

Two quantum circuits are checked for exact equivalence by contracting a network of Tensor
Decision Diagrams (TDDs) down to a single tensor and testing it against the identity. Every
contraction order reaches the same verdict, but the orders differ enormously in how large the
intermediate diagrams grow and how long the whole check takes. Your job is the planner: at each
step, pick which pair to contract next.

## Hard Constraints

- Edit only files under `/app/methods/main/`. That directory is what gets graded.
- Keep the exact contract: `select_edge(observation) -> int` returns one **enabled, stable**
  `edge_id`. The exact TDD engine contracts that pair and hands you the next observation.
- Return only an edge that `action_mask` marks as enabled. An invalid action forfeits the case.
- Inference must be **deterministic** and independent of case IDs. Do not encode case IDs, fixed
  edge sequences, tensor IDs, or per-case tables — the sealed circuits have different hashes and
  sizes, and anything keyed on identity will simply miss.
- Do not train a model and do not download artifacts. There is no network at run time; whatever
  you use has to be already installed or written here.
- Do not attempt to read verifier or held-out files.
- Your planner gets **30 seconds of total decision time per circuit**, and no single
  `select_edge` call may take more than 5 seconds. Thinking longer per step is fine within that
  budget — the time you spend deciding is not part of your score.
- Every circuit must finish and pass exact identity checking. Exceptions, invalid actions,
  incomplete contractions, failed identity checks, native resource failures and budget overruns
  all score zero for that case.

## What You Have

- `/app/methods/main/solver.py` — a crude planner that always takes the strongest shared bond.
  It is yours to rewrite or replace entirely.
- `python /app/selfcheck.py` — free and unlimited. It always evaluates all 12 frozen visible
  cases (eight standard, four larger scalability cases) and reports the raw metrics. The visible
  set is fixed so successive planner revisions stay comparable.
- `/app/data/visible_cases.json` — those 12 cases.

Each observation gives you:

| key | contents |
|---|---|
| `nodes` | stable node IDs, current exact TDD sizes, rank, degree, and how many original gates each active tensor represents |
| `edges` | stable `edge_id`, endpoints, shared-index count, bond dimension, normalized bond strength / local topology, endpoint sizes and degrees |
| `action_mask` | which edge IDs are currently enabled |
| `global_features` | active node/edge counts, step, qubits, latest size, peak size so far |

Two things that bite: **list positions are not stable IDs**, and the scores you computed last
step are stale — recompute after every contraction.

## What You Submit

Leave your best planner in `/app/methods/main/solver.py`, exposing exactly:

```python
def select_edge(observation: dict) -> int:
    """Return one edge_id enabled by observation["action_mask"]."""
```

There is no submit step and no per-attempt feedback. Work and self-check for as long as your run
window allows, then leave your best `solver.py` in place.

## How It Is Judged

The grader replays every contraction on **12 sealed cases you never see** — four in-family
standard sizes, four zero-shot Bernstein–Vazirani circuits, and four larger scalability cases,
none of them overlapping the visible set. All contractions use the same 23-bit TDD normalization
precision as your workbench.

A submission must first be **correct**: every circuit finishes and passes exact identity
checking. Among correct planners, two numbers are measured per case and combined in log space:

- **peak intermediate TDD node count** — 75% of the weight;
- **the time the native engine spends contracting** — 25%. Your own decision time is not
  counted here; it only has to stay inside the budget above.

Both are lower-is-better. The grader runs the trusted starter and a frozen reference planner on
the same hardware in the same pass, so the comparison is not affected by machine load. How the
raw measurements map to the final reward is deliberately not disclosed — optimise the raw
measurements themselves. The starter as shipped is the zero of that scale: submitted unchanged it
scores 0.
