#!/usr/bin/env python3
"""Run one device prediction in an isolated process (mirrors the sealed
evaluation's calling convention).

Usage: run_solver.py --solver-dir D --record RECORD.json --out OUT.json

RECORD.json is one device record WITHOUT truth fields. Writes
{"pred": [len(record["extreme"]) floats]} to OUT.json.
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
    pred = [float(v) for v in mod.predict(rec)]
    if len(pred) != len(rec["extreme"]):
        raise SystemExit(4)
    json.dump({"pred": pred}, open(a.out, "w"))


if __name__ == "__main__":
    main()
