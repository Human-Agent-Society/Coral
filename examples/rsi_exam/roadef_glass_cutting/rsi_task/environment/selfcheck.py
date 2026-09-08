#!/usr/bin/env python3
"""Free, unlimited local dev bench for glass_cutting_v1.

Runs YOUR methods/main/run.sh on the visible instances (tools/instances), validates
+ measures each produced solution with the bundled checker (./checker), and
reports:

  * per-case absolute WASTE (area of used plates minus items minus the reusable
    last-plate residual; lower is better; a checker-rejected solution scores 0).

The graded evaluation reruns this protocol on the sealed instances (different
instances, same distribution) and gives no feedback. How the raw waste maps to the
final reward is deliberately not shown here -- optimise the raw metric itself.

Usage:  python3 selfcheck.py            # all visible cases
        python3 selfcheck.py 3          # first 3 (faster)
"""
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "methods" / "main" / "run.sh"
BUILD = HERE / "methods" / "main" / "build.sh"
CHECKER = HERE / "checker"
IN_DIR = HERE / "tools" / "instances"
GLOBAL_PARAM = IN_DIR / "global_param.csv"







def plate_dims():
    W, H = 6000, 3210
    for row in csv.reader(open(GLOBAL_PARAM), delimiter=';'):
        if row and row[0] == "widthPlates":
            W = int(row[1])
        elif row and row[0] == "heightPlates":
            H = int(row[1])
    return W, H


def item_area(batch):
    tot = 0
    r = csv.reader(open(batch), delimiter=';'); next(r, None)
    for row in r:
        if row and row[0].strip():
            tot += int(row[1]) * int(row[2])
    return tot


def check_waste(idx, batch, defects, sol_path, W, H):
    work = tempfile.mkdtemp()
    inst = Path(work) / "instances_checker"; logs = Path(work) / "logs"
    inst.mkdir(); logs.mkdir()
    (inst / f"{idx}_batch.csv").write_bytes(Path(batch).read_bytes())
    (inst / f"{idx}_defects.csv").write_bytes(Path(defects).read_bytes())
    (inst / "global_param.csv").write_bytes(Path(GLOBAL_PARAM).read_bytes())
    (inst / f"{idx}_solution.csv").write_bytes(Path(sol_path).read_bytes())
    try:
        subprocess.run([str(CHECKER), idx], cwd=work, input="0\n",
                       capture_output=True, text=True, timeout=120)
    except subprocess.SubprocessError:
        return None
    stat = logs / f"{idx}_statistics.csv"
    if not stat.exists():
        return None
    lines = stat.read_text().strip().splitlines()
    if len(lines) < 2:
        return None
    p = lines[1].split(";")
    try:
        valid = int(p[1]); nplates = int(p[2]); resid = int(float(p[4]))
    except (ValueError, IndexError):
        return None
    if valid != 1:
        return None
    return ((nplates - 1) * W + (W - resid)) * H - item_area(batch)


def run_case(idx, batch, defects, W, H):
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
        out = tf.name
    try:
        subprocess.run(["bash", str(RUN), batch, defects, str(GLOBAL_PARAM), out],
                       cwd=str(RUN.parent), stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=120)
        if os.path.getsize(out) == 0:
            return None
        return check_waste(idx, batch, defects, out, W, H)
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass


def main():
    if BUILD.exists():
        print("running build.sh ...")
        subprocess.run(["bash", str(BUILD)], cwd=str(BUILD.parent))
    W, H = plate_dims()
    batches = sorted(IN_DIR.glob("*_batch.csv"))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(batches)
    batches = batches[:limit]
    t0 = time.time()
    wastes, valid = [], 0
    for b in batches:
        idx = b.name[:-len("_batch.csv")]
        defects = IN_DIR / f"{idx}_defects.csv"
        waste = run_case(idx, str(b), str(defects), W, H)
        if waste and waste > 0:
            valid += 1
            wastes.append(waste)
        print(f"{idx}: your_waste={waste}")
    mean_waste = sum(wastes) / len(wastes) if wastes else 0.0
    print(f"\nmean waste          : {mean_waste:.2f}  (your solver, over valid cases; lower is better)")
    print(f"valid cases         : {valid}/{len(batches)}")
    print(f"elapsed             : {time.time()-t0:.0f}s")

    # Budget reminder. /app/budget.py is mounted by the harness; when it is not mounted the
    # whole block is skipped, so running this task standalone is unaffected. check=False so a
    # failing budget.py can never take selfcheck down with it. Placed at the end of main() so
    # the reminder prints after this run's score.
    import os as _os, subprocess as _sp, sys as _sys
    if _os.path.exists("/app/budget.py"):
        _sys.stdout.flush()          # without this the reminder prints before the score in a pipe
        _sp.run([_sys.executable, "/app/budget.py"], check=False)


if __name__ == "__main__":
    main()
