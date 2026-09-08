#!/usr/bin/env python3
"""Free local dry-run — MIRRORS the grader's
per-instance time cap.

For each VISIBLE graph it runs your methods/main/solver.py in a fresh subprocess with a HARD
per-instance timeout (PER_INSTANCE_TIMEOUT_SEC). If an instance runs longer it is KILLED and
scored 0 — and this script tells you EXACTLY which instance was killed. For instances that
finish it validates the coloring (length n, every color in [0, target_k)) and prints the
RAW conflict count. Optimise that directly: fewer conflicts is better, 0 = a proper coloring.
How the raw conflict counts map to the final reward is deliberately not shown here.

    python /app/selfcheck.py

The hidden test set is a DIFFERENT sealed batch from the same generators, scored the
same way (same per-instance cap). Do not overfit to the visible set.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PER_INSTANCE_TIMEOUT = float(os.environ.get("PER_INSTANCE_TIMEOUT_SEC", "45"))
VERIFIER_CEILING = float(os.environ.get("VERIFIER_CEILING_SEC", "1800"))
N_HIDDEN = int(os.environ.get("N_HIDDEN_INSTANCES", "7"))


def _colors_of(entry):
    return entry.get("colors") if isinstance(entry, dict) else entry


def run_one(one_json: str, out_json: str) -> int:
    sys.path.insert(0, str(HERE / "methods" / "main"))
    from solver import solve
    data = json.loads(Path(one_json).read_text())
    decision = solve(data)
    seq = decision.get("colorings") if isinstance(decision, dict) else None
    if seq is None and isinstance(decision, dict):
        seq = decision.get("assignments")
    colors = _colors_of(seq[0]) if isinstance(seq, list) and seq else []
    if not isinstance(colors, list):
        colors = []
    Path(out_json).write_text(json.dumps({"colors": colors}))
    return 0


def check_coloring(inst: dict, colors) -> tuple[bool, int, str]:
    n = int(inst["n_vertices"])
    tk = int(inst["target_k"])
    if not isinstance(colors, list) or len(colors) != n:
        got = len(colors) if isinstance(colors, list) else "?"
        return False, 0, f"coloring length {got} != n={n}"
    for v, c in enumerate(colors):
        if isinstance(c, bool) or not isinstance(c, int) or c < 0 or c >= tk:
            return False, 0, f"vertex {v} color {c} outside [0,{tk})"
    conf = sum(1 for u, v in inst["edges"] if colors[int(u)] == colors[int(v)])
    return True, conf, ""



def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--one":
        return run_one(sys.argv[2], sys.argv[3])

    data = json.loads((HERE / "data" / "visible.json").read_text())
    instances = data.get("instances", [])
    meta = {k: v for k, v in data.items() if k != "instances"}

    import tempfile
    n_ok, killed, failed, total_conf, total_time = 0, [], [], 0, 0.0
    print(f"per-instance cap = {PER_INSTANCE_TIMEOUT:.0f}s (same as grading); "
          f"{len(instances)} visible graphs\n")
    print(f"  {'graph':8s} {'k':>3s} {'time':>7s}  result")
    with tempfile.TemporaryDirectory(prefix="gc_sc_") as tmp:
        for i, inst in enumerate(instances):
            one = Path(tmp) / f"in_{i}.json"; out = Path(tmp) / f"out_{i}.json"
            one.write_text(json.dumps({**meta, "instances": [inst]}))
            t0 = time.perf_counter()
            proc = subprocess.Popen([sys.executable, str(HERE / "selfcheck.py"),
                                     "--one", str(one), str(out)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    start_new_session=True)
            timed_out = False
            try:
                proc.communicate(timeout=PER_INSTANCE_TIMEOUT)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            dt = time.perf_counter() - t0
            total_time += min(dt, PER_INSTANCE_TIMEOUT)
            nm = inst["name"]; tk = int(inst["target_k"])

            if timed_out:
                killed.append(nm)
                print(f"  {nm:8s} {tk:3d} {PER_INSTANCE_TIMEOUT:6.0f}s  "
                      f"*** KILLED (>{PER_INSTANCE_TIMEOUT:.0f}s) — counts as a failure ***")
                continue
            colors = []
            if out.exists() and out.stat().st_size > 0:
                try:
                    colors = json.loads(out.read_text()).get("colors", []) or []
                except Exception:
                    colors = []
            ok, conf, err = check_coloring(inst, colors)
            if not ok:
                failed.append(nm)
                print(f"  {nm:8s} {tk:3d} {dt:6.2f}s  INVALID: {err}")
                continue
            total_conf += conf
            n_ok += 1
            print(f"  {nm:8s} {tk:3d} {dt:6.2f}s  conflicts={conf:3d}")

    print(f"\n  total conflicts over {n_ok}/{len(instances)} solved graph(s) = {total_conf}"
          f"   (lower is better; 0 = a proper coloring)")
    if failed:
        print(f"  *** {len(failed)} graph(s) produced an INVALID coloring: {', '.join(failed)} ***")
    print(f"  this dry-run used ~{total_time:.0f}s over {len(instances)} instances")
    if killed:
        print(f"  *** {len(killed)} instance(s) KILLED for exceeding {PER_INSTANCE_TIMEOUT:.0f}s: "
              f"{', '.join(killed)} — make solve() finish within the cap ***")
    print(f"\n  GRADING BUDGET: the hidden test has {N_HIDDEN} instances, each with the SAME "
          f"{PER_INSTANCE_TIMEOUT:.0f}s hard cap (an instance over {PER_INSTANCE_TIMEOUT:.0f}s is "
          f"killed and scored 0); verifier ceiling ~{VERIFIER_CEILING:.0f}s. Self-pace so every "
          f"instance returns within {PER_INSTANCE_TIMEOUT:.0f}s.")
    import os as _os, subprocess as _sp, sys as _sys
    if _os.path.exists("/app/budget.py"):
        _sys.stdout.flush()
        _sp.run([_sys.executable, "/app/budget.py"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
