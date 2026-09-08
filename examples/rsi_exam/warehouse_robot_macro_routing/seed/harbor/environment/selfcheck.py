#!/usr/bin/env python3
"""Free, unlimited local dev bench for macro_controller_v1.

Runs YOUR methods/main/run.sh on the 100 visible instances (tools/in, seeds 0-99),
scores each with the OFFICIAL visualiser (tools/vis), and reports:

  * per-case absolute score (button-sequence length; lower is better,
    T*(M-V) penalty if not all balls are basketed),
  * per-case reference score from a reproduced strong solver, for comparison.

The graded evaluation reruns this exact protocol on sealed instances (fresh seeds,
same generator/distribution) and gives no feedback. How the raw scores map to the
final reward is deliberately not shown here -- optimise the raw metric itself.

Usage:  python3 selfcheck.py            # all 100 visible cases
        python3 selfcheck.py 20         # first 20 (faster)
"""
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
VIS = HERE / "tools" / "vis"
IN_DIR = HERE / "tools" / "in"
STRONG = json.loads((HERE / "reference" / "visible_rank1_lengths.json").read_text())

SCORE_RE = re.compile(r"Score\s*=\s*(-?\d+)")




def run_case(f):
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as tf:
        out = tf.name
    try:
        with open(f) as fi, open(out, "w") as fo:
            subprocess.run(["bash", str(RUN)], cwd=str(RUN.parent),
                           stdin=fi, stdout=fo, stderr=subprocess.DEVNULL, timeout=60)
        r = subprocess.run([str(VIS), str(f), out], capture_output=True, text=True, timeout=60)
        m = SCORE_RE.search(r.stdout + r.stderr)
        return int(m.group(1)) if m else None
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass


def main():
    if BUILD.exists():
        print("running build.sh ...")
        subprocess.run(["bash", str(BUILD)], cwd=str(BUILD.parent))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    files = sorted(IN_DIR.glob("*.txt"))[:limit]
    t0 = time.time()
    scores, valid = [], 0
    for f in files:
        s = run_case(f)
        r1 = STRONG.get(f.name)
        if s and s > 0:
            valid += 1
            scores.append(s)
        print(f"{f.name}: your={s}  reference={r1}")
    mean_raw = sum(scores) / len(scores) if scores else 0.0
    print(f"\nmean raw score      : {mean_raw:.2f}  (your solver, over completed cases)")
    print(f"completed cases     : {valid}/{len(files)}")
    print(f"elapsed             : {time.time()-t0:.0f}s")
    import os as _os, subprocess as _sp, sys as _sys
    if _os.path.exists("/app/budget.py"):
        _sys.stdout.flush()
        _sp.run([_sys.executable, "/app/budget.py"], check=False)


if __name__ == "__main__":
    main()
