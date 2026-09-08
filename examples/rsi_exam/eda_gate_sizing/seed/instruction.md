# Optimize logic gate sizing under timing and design-rule penalties

## Background

A chip is built from millions of tiny logic gates, and the same gate comes in several sizes that
do the identical job. A bigger one switches faster, but burns more power and is heavier for the
gate feeding it to push. Picking a size for every gate, so that no signal arrives late and as
little power as possible is wasted, is called **gate sizing**.

It is one of the last steps in chip design before a layout goes to a factory. It is hard because
the right size differs from gate to gate and the gates pull on each other: enlarging one slows the
gate in front of it, and the timing you start from describes the chip before you touched it.

## Instruction

Given a placed netlist described by CircuitOps IR tables, assign a library cell to every instance
so as to minimize **one score** charging timing (TNS) and design-rule (slew / capacitance)
violations, plus the leakage you spend clearing them. Your submission is re-run on sealed hidden
cases; **lower is better**.

`/app/methods/main/` ships a working but crude reference implementation. It is there to define the
zero point and to show the plumbing, **not to be the starting shape of your answer**. You may
delete all of it and design your method from scratch. The only fixed parts of a submission are the
`solve(...)` contract and the `.size` output format below. What your method looks at, what it
decides per instance, and how it searches are entirely yours to choose.

### Hard Constraints

- Submit an **algorithm** (`solve`), not precomputed answers. Do not key on case names.
- Keep the exact signature `solve(input_dir: str, output_path: str) -> None`: read one case from
  `input_dir`, write a valid `.size` file to `output_path`.
- **Sequential cells and macros** keep their original library cell. Only **combinational** cells
  may be resized.
- A combinational cell may only move to a cell with the **same `func_id`** (logically equivalent,
  different drive strength or threshold), read from `libcell_properties.csv.bz2`.
- List **every instance exactly once**, with no unknown instance or library-cell names. An illegal
  sizing scores nothing.
- `solve` must be deterministic and case-name-independent.

### What You Have

- Visible cases at `/app/data/<case>/` for each case in `/app/data/manifest.json`, each with
  `IR_Tables/` (`cell_properties`, `libcell_properties`, `pin_properties`, `net_properties` and the
  graph edge tables, all `*.csv.bz2`) and `design/` (compressed `.v`/`.def`/`.sdc`). The shared
  library is at `/app/data/platform/ASAP7`. Hidden cases are different designs, same format.
- `/app/methods/main/` — **this directory is what gets graded**. The shipped `solver.py` handles
  the contract and the `.size` format and applies a uniform strongest-drive upsize. **Its score is
  the zero point**: matching it earns nothing, and scoring worse earns nothing either. Rewrite it,
  throw it away and start over, or add any helper Python next to it (`python3`, `numpy`, `pandas`,
  `scikit-learn` available).
- **Your self-check surface** (free, unlimited): `python /app/selfcheck.py` runs your current
  `solver.py` on the visible cases through the **same legality gate and the same OpenROAD score**
  the sealed grader uses, and prints the breakdown into `10*|TNS|`, `20*slew`, `20*cap` and
  leakage. A case name limits it to that case (`python /app/selfcheck.py <case>`); the scoring
  pass takes minutes on the large cases.

### What You Submit

`/app/methods/main/solver.py` must expose this exact signature. It and the `.size` format
below are the only parts of a submission that are fixed:

```python
def solve(input_dir: str, output_path: str) -> None:
    ...
```

The `.size` output is one line per instance:

```text
<instance name> <library cell name>
```

There is no submit step and no per-attempt feedback. Work and self-check for as long as your run
window allows, then leave your best `solver.py` in place; it is graded once at the end.

### How It Is Judged

The grader copies `methods/main/` into a clean sandbox, runs `solve(...)` on each hidden case,
checks legality, then evaluates the sizing with a timing engine. The per-case score is

```text
score = leakage_delta_uW + 10*|TNS_ns| + 20*slew_violation + 20*cap_violation
```

`leakage_delta_uW` is your leakage minus the original netlist's, so leaving a cell alone costs
nothing. `TNS` is total negative slack; `slew` and `cap` are the summed amounts by which pins
exceed their transition-time and load-capacitance limits. **Each penalty term switches off
entirely once its violation reaches zero**, so driving a category to exactly 0 is worth more than
driving it low. Scoring is per case against that case's own reference points, then averaged, so a
case you ignore cannot be carried by one you optimize. **Lower is better.** Your solver's
wall-clock time is not scored, though it is capped.
