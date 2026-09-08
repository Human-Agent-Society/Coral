#!/usr/bin/env python3
"""Run one feeder calibration in an isolated process (mirrors the sealed
evaluation's calling convention).

Usage: run_solver.py --solver-dir D --record RECORD.json --out OUT.json

RECORD.json is one feeder record WITHOUT the truth field. Writes
{"pred": {...}} to OUT.json.
"""
import argparse
import importlib.util
import json
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver-dir", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.solver_dir)
    spec = importlib.util.spec_from_file_location(
        "solver", a.solver_dir + "/solver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rec = json.load(open(a.record))
    pred = mod.calibrate(rec)
    out = {"phase": [int(p) for p in pred["phase"]],
           "tap_steps": float(pred["tap_steps"])}
    if pred.get("z_rx") is not None:
        out["z_rx"] = [[float(r), float(x)] for r, x in pred["z_rx"]]
    else:
        out["code"] = [str(c) for c in pred["code"]]
        out["scale"] = [float(s) for s in pred["scale"]]
    json.dump({"pred": out}, open(a.out, "w"))


if __name__ == "__main__":
    main()
