#!/usr/bin/env python3
"""Self-check: score methods/main/solver.py on the 24 practice devices
against their included full-range truth. Free and unlimited. Prints the mean
practice error (LOWER is better; the same per-device error definition is
used by the sealed evaluation, on a device lot you never see the truth of).

Per-device error: mean |log10 I_pred - log10 I_true| over that device's
extreme grid, capped at 6.0 (a crashed / over-budget / malformed prediction
counts as the 6.0 worst case in the evaluation).
"""
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUDGET_SEC = 180.0     # per-device wall budget enforced by the evaluation
ERR_CAP = 6.0


def _one(args):
    inst, solver_dir = args
    import importlib.util as iu
    import sys as _sys
    import time as _time
    _sys.path.insert(0, solver_dir)
    spec = iu.spec_from_file_location("solver", solver_dir + "/solver.py")
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    feats = {k: v for k, v in inst.items() if k != "truth_logI"}
    t0 = _time.monotonic()
    pred = np.asarray(mod.predict(feats), float)
    wall = _time.monotonic() - t0
    y = np.asarray(inst["truth_logI"], float)
    if pred.shape != y.shape or not np.all(np.isfinite(pred)):
        return ERR_CAP, wall
    return min(float(np.mean(np.abs(pred - y))), ERR_CAP), wall


def main() -> None:
    solver_dir = str(HERE / "methods" / "main")
    data = json.loads(
        (HERE / "data" / "practice" / "instances.json").read_text())
    insts = data["instances"]
    t0 = time.monotonic()
    try:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=4) as ex:
            outs = list(ex.map(_one, [(i, solver_dir) for i in insts]))
    except Exception:
        outs = [_one((i, solver_dir)) for i in insts]
    slow = 0
    for inst, (err, wall) in zip(insts, outs):
        flag = ""
        if wall > BUDGET_SEC:
            slow += 1
            flag = f"  [over {BUDGET_SEC:.0f}s budget!]"
        print(f"{inst['id']}: error={err:.4f} log10-decades "
              f"{wall:6.1f}s{flag}")
    errs = [o[0] for o in outs]
    print(f"devices: {len(errs)}, wall {time.monotonic()-t0:.0f}s")
    if slow:
        print(f"WARNING: {slow} devices over the per-device budget "
              "(the evaluation scores those as worst case)")
    print(f"mean practice error: {np.mean(errs):.4f}")

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
