#!/usr/bin/env python3
"""PHASE-1 child (UNTRUSTED side of the two-phase verifier, see grade.py).

Fresh process, unprivileged uid, own session — NOT a fork of the parent. Imports
the agent's submitted solver and runs solve() on each sealed held-out case,
writing one `.size` per case into an output dir owned by that uid. It never sees
the scoring code, the anchor table or the reward path.

    python3 child_solve.py <submission_dir> <heldout_dir> <out_dir> <budget_s> <grace_s>

Per-case SOFT deadline: running out of time is not "the submission
failed". At budget+grace the case is interrupted and, if solve() left no usable
output, a legal do-nothing sizing is written so the case still scores (at the
baseline anchor, i.e. 0 for that case) instead of being thrown out as illegal.
"""
from __future__ import annotations

import bz2
import csv
import importlib.util
import signal
import sys
from pathlib import Path


class _CaseTimeout(Exception):
    pass


def _on_alarm(signum, frame):  # noqa: ARG001
    raise _CaseTimeout()


def _load_solver(submission_dir: Path):
    spec = importlib.util.spec_from_file_location(
        "submission_solver", submission_dir / "solver.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(submission_dir))
    spec.loader.exec_module(mod)
    return mod


def _write_do_nothing(case_dir: Path, out_path: Path) -> None:
    """Legal fallback: every instance at its original library cell."""
    with bz2.open(case_dir / "IR_Tables" / "cell_properties.csv.bz2", "rt",
                  newline="") as f, out_path.open("w") as w:
        for row in csv.DictReader(f):
            w.write(f"{row['cell_name']} {row['libcell_name']}\n")


def main() -> int:
    submission_dir = Path(sys.argv[1])
    heldout = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    budget = float(sys.argv[4])
    grace = float(sys.argv[5])
    out_dir.mkdir(parents=True, exist_ok=True)

    solver = _load_solver(submission_dir)
    cases = sorted(p.name for p in heldout.iterdir()
                   if p.is_dir() and (p / "IR_Tables").exists())
    signal.signal(signal.SIGALRM, _on_alarm)

    for case in cases:
        out_path = out_dir / f"{case}.size"
        signal.setitimer(signal.ITIMER_REAL, budget + grace)
        try:
            solver.solve(str(heldout / case), str(out_path))
        except _CaseTimeout:
            print(f"child: solve() exceeded {budget}s (+{grace}s grace) on {case}",
                  file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 — parent scores what is on disk
            print(f"child: solve() failed on {case}: {exc}", file=sys.stderr)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        if not out_path.exists() or out_path.stat().st_size == 0:
            try:
                _write_do_nothing(heldout / case, out_path)
                print(f"child: wrote do-nothing fallback for {case}", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                print(f"child: fallback failed on {case}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
