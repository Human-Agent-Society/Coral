#!/usr/bin/env python3
"""CHILD side of the two-phase verifier (UNTRUSTED — imports + runs the submission).

Spawned by grade.py as a fresh process (NOT a fork). It imports the submitted solver, runs
triage(cases, resources, seed) on the held-out studies (gold labels already removed by the
parent), and writes ONLY the per-study risks to OUTPUT_JSON. It never sees the gold_critical
labels or the scorer, so anything it does to its own process cannot affect the score the
trusted parent computes.

argv: child_triage.py <submission_dir> <cases_json> <resources_json> <output_json>
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SEED = 20260626


def main() -> int:
    submission_dir = Path(sys.argv[1])
    cases_json = Path(sys.argv[2])
    resources_json = Path(sys.argv[3])
    output_json = Path(sys.argv[4])

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
    if not hasattr(mod, "triage"):
        print("solver.py must define triage(cases, resources, seed)", file=sys.stderr)
        return 2

    cases = json.loads(cases_json.read_text())
    resources = json.loads(resources_json.read_text())
    preds = mod.triage(cases, resources, SEED)
    json.dumps(preds)  # must be JSON-serializable
    output_json.write_text(json.dumps(preds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
