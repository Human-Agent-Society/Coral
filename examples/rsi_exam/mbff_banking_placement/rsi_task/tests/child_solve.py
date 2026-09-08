#!/usr/bin/env python3
"""CHILD side of the two-phase verifier (UNTRUSTED — imports + runs the submission).

Spawned by grade.py as a fresh process (NOT a fork). It imports the submitted solver and
runs solve(input_path, output_path) on each held-out case, writing ONLY placement outputs to
OUTPUT_DIR. It never receives the scoring binaries or the reward path, so anything it does to
its own process (monkeypatching, etc.) cannot affect the score the trusted parent computes
from these output files.

argv: child_solve.py <submission_dir> <cases_dir> <output_dir>
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main() -> int:
    submission_dir = Path(sys.argv[1])
    cases_dir = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])
    output_dir.mkdir(parents=True, exist_ok=True)

    solver_path = submission_dir / "solver.py"
    if not solver_path.exists():
        print(f"missing solver.py under {submission_dir}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(submission_dir))
    spec = importlib.util.spec_from_file_location("submitted_solver", str(solver_path))
    if spec is None or spec.loader is None:
        print(f"cannot import {solver_path}", file=sys.stderr)
        return 2
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # top-level submission code runs HERE (untrusted)
    if not hasattr(mod, "solve"):
        print("solver.py must define solve(input_path, output_path)", file=sys.stderr)
        return 2

    cases = sorted(p for p in cases_dir.iterdir() if p.is_file())
    for case in cases:
        out = output_dir / f"{case.name}.out"
        try:
            mod.solve(str(case), str(out))
        except Exception as exc:  # noqa: BLE001 — a missing/partial output fails that case in the parent
            print(f"{case.name}: solve failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
