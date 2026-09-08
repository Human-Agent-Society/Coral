#!/usr/bin/env python3
"""CHILD side of the two-phase verifier (UNTRUSTED — imports + runs the submission),
with PER-INSTANCE isolation and a hard per-instance time cap.

  batch (default):  child_solve.py <submission_dir> <hidden_data_json> <output_json>
     For EACH held-out instance, run a fresh grandchild (`--one`) in its OWN session /
     process group with a hard per-instance timeout (PER_INSTANCE_TIMEOUT_SEC). On timeout
     the whole grandchild group is SIGKILLed and that instance's coloring is left EMPTY
     (scored 0 by the sealed per-instance scorer), the rest unaffected. Assembles
     {"colorings": [...]} and writes it.

  one:  child_solve.py --one <submission_dir> <one_instance_json> <out_json>
     Import the submission, run solve() on a SINGLE-instance batch; write that instance's
     coloring as {"colors": [...]}.

It never sees the scorer or the per-instance anchor table, so anything it does to its own
process cannot affect the score the trusted parent computes from this decision file.
"""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

PER_INSTANCE_TIMEOUT = float(os.environ.get("PER_INSTANCE_TIMEOUT_SEC", "45"))


def _colors_of(entry):
    return entry.get("colors") if isinstance(entry, dict) else entry


def run_one(submission_dir: str, one_json: str, out_json: str) -> int:
    solver_path = Path(submission_dir) / "solver.py"
    if not solver_path.exists():
        print(f"missing solver.py under {submission_dir}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(submission_dir))
    spec = importlib.util.spec_from_file_location("submitted_solver", str(solver_path))
    if spec is None or spec.loader is None:
        print(f"cannot import {solver_path}", file=sys.stderr)
        return 2
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so a submission that uses multiprocessing can pickle its own
    # module-level worker functions (pickle re-imports them by module name). Without this,
    # any parallel solver dies with "Can't pickle ...: import of module 'submitted_solver'
    # failed" even though it runs fine under selfcheck (which imports it as `solver`).
    sys.modules["submitted_solver"] = mod
    spec.loader.exec_module(mod)  # top-level submission code runs HERE (untrusted)
    if not hasattr(mod, "solve"):
        print("solver.py must define solve(data)", file=sys.stderr)
        return 2

    data = json.loads(Path(one_json).read_text())
    decision = mod.solve(data)
    seq = decision.get("colorings") if isinstance(decision, dict) else None
    if seq is None and isinstance(decision, dict):
        seq = decision.get("assignments")
    colors = _colors_of(seq[0]) if isinstance(seq, list) and seq else []
    if not isinstance(colors, list):
        colors = []
    json.dumps(colors)  # must be JSON-serializable
    Path(out_json).write_text(json.dumps({"colors": colors}))
    return 0


def run_batch(submission_dir: str, data_json: str, output_json: str) -> int:
    data = json.loads(Path(data_json).read_text())
    instances = data.get("instances", [])
    meta = {k: v for k, v in data.items() if k != "instances"}
    colorings: list[list[int]] = []

    with tempfile.TemporaryDirectory(prefix="gc_pi_") as tmp:
        for i, inst in enumerate(instances):
            one = Path(tmp) / f"in_{i}.json"
            out = Path(tmp) / f"out_{i}.json"
            one.write_text(json.dumps({**meta, "instances": [inst]}))
            proc = subprocess.Popen(
                [sys.executable, __file__, "--one", str(submission_dir), str(one), str(out)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                start_new_session=True,
            )
            timed_out = False
            try:
                stdout, _ = proc.communicate(timeout=PER_INSTANCE_TIMEOUT)
                if stdout:
                    print(f"[{inst.get('name')}] {stdout.strip()[-300:]}", file=sys.stderr)
            except subprocess.TimeoutExpired:
                timed_out = True
                print(f"[{inst.get('name')}] TIMEOUT (>{PER_INSTANCE_TIMEOUT:.0f}s) -> killed, scored 0",
                      file=sys.stderr)
            finally:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

            colors: list[int] = []
            if not timed_out and out.exists() and out.stat().st_size > 0:
                try:
                    colors = json.loads(out.read_text()).get("colors", []) or []
                except Exception:
                    colors = []
            colorings.append(colors)

    Path(output_json).write_text(json.dumps({"colorings": colorings}))
    return 0


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--one":
        return run_one(sys.argv[2], sys.argv[3], sys.argv[4])
    return run_batch(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    raise SystemExit(main())
