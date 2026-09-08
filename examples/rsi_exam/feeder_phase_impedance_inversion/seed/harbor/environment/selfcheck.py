#!/usr/bin/env python3
"""Self-check: score methods/main/solver.py on the 12 practice feeders
against their included truth. Free and unlimited. Prints per-feeder
component scores and the mean (HIGHER is better; the same per-feeder
score definition is used by the evaluation, on feeders you never see the
truth of).

Per-feeder score in [0, 1]:
    0.45 * phase score      accuracy over metered customers, rescaled so
                            copying the shipped GIS labels = 0, perfect = 1
  + 0.40 * impedance score  1 - ||log path errors|| / ||log path errors of
                            the records-as-shipped||, where a path error is
                            the log ratio of predicted to actual cumulative
                            positive-sequence R and X from the substation
                            to each metered customer's bus; clipped to [0,1]
  + 0.15 * tap score        1 - |tap step error| / 4, clipped to [0,1]

Returning the records as shipped (GIS labels + recorded conductors +
recorded tap 0) scores exactly 0.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUDGET_SEC = 300.0     # per-feeder wall budget enforced by the evaluation


def score_instance(inst, pred):
    tr = inst["truth"]
    met = inst["meter_load_idx"]
    tp = np.array(tr["phase"])[met]
    gp = np.array([ld["gis_phase"] for ld in inst["loads"]])[met]
    pp = np.array(pred["phase"])[met]
    acc = float((pp == tp).mean())
    acc_gis = float((gp == tp).mean())
    phase_score = float(np.clip((acc - acc_gis) /
                                max(1.0 - acc_gis, 0.05), 0.0, 1.0))

    segs = inst["segments"]
    parent = {s["tb"]: (s["fb"], s["id"]) for s in segs}

    def path_ids(bus):
        out = []
        b = bus
        while b in parent:
            fb, sid = parent[b]
            out.append(sid)
            b = fb
        return out

    met_paths = [path_ids(inst["loads"][li]["bus"]) for li in met]

    def seg_rx(codes, scales, z_rx=None):
        if z_rx is not None:
            arr = np.clip(np.array(z_rx, float), 1e-4, None)
            return arr[:, 0], arr[:, 1]
        r = np.array([inst["catalog"][c]["r1"] * s["km"] * sc
                      for s, c, sc in zip(segs, codes, scales)])
        x = np.array([inst["catalog"][c]["x1"] * s["km"] * sc
                      for s, c, sc in zip(segs, codes, scales)])
        return r, x

    def pathvec(r, x):
        out = []
        for pth in met_paths:
            out += [sum(r[i] for i in pth), sum(x[i] for i in pth)]
        return np.array(out)

    z_true = pathvec(*seg_rx(tr["code"], tr["scale"]))
    z_pred = pathvec(*seg_rx(pred["code"], pred["scale"],
                             pred.get("z_rx")))
    z_nom = pathvec(*seg_rx([s["rec_code"] for s in segs],
                            np.ones(len(segs))))
    e_pred = np.log(z_pred / z_true)
    e_nom = np.log(z_nom / z_true)
    imp_score = float(np.clip(
        1.0 - np.linalg.norm(e_pred) / max(np.linalg.norm(e_nom), 1e-9),
        0.0, 1.0))

    dt = abs(float(pred["tap_steps"]) - tr["tap_steps"])
    tap_score = float(np.clip(1.0 - dt / 4.0, 0.0, 1.0))

    total = 0.45 * phase_score + 0.40 * imp_score + 0.15 * tap_score
    return {"phase_score": phase_score, "imp_score": imp_score,
            "tap_score": tap_score, "score": total}


def main() -> None:
    solver_dir = str(HERE / "methods" / "main")
    sys.path.insert(0, solver_dir)
    spec = importlib.util.spec_from_file_location(
        "solver", solver_dir + "/solver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data = json.loads(
        (HERE / "data" / "practice" / "instances.json").read_text())
    insts = data["instances"]
    scores, slow = [], 0
    t_all = time.monotonic()
    for inst in insts:
        feats = {k: v for k, v in inst.items() if k != "truth"}
        t0 = time.monotonic()
        pred = mod.calibrate(feats)
        wall = time.monotonic() - t0
        sc = score_instance(inst, pred)
        scores.append(sc["score"])
        flag = ""
        if wall > BUDGET_SEC:
            slow += 1
            flag = f"  [over {BUDGET_SEC:.0f}s budget!]"
        print(f"{inst['id']}: score={sc['score']:.4f} "
              f"(phase={sc['phase_score']:.3f} imp={sc['imp_score']:.3f} "
              f"tap={sc['tap_score']:.3f}) {wall:5.1f}s{flag}")
    print(f"feeders: {len(scores)}, wall {time.monotonic()-t_all:.0f}s")
    if slow:
        print(f"WARNING: {slow} feeders over the per-feeder budget "
              "(the evaluation scores those as worst case)")
    print(f"mean practice score: {np.mean(scores):.4f}")

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
