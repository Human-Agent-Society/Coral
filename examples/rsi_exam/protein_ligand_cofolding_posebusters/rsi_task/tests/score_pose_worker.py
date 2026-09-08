#!/usr/bin/env python3
"""Trusted, time-bounded worker for native ligand parsing and PoseBusters."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def _load_metric(path: Path):
    spec = importlib.util.spec_from_file_location("isolated_cofold_metric", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load metric")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_exclusive(path: Path, payload: dict) -> None:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--metric", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        if set(request) != {
            "prediction",
            "crystal_ligand",
            "crystal_protein",
            "expected_chains",
            "expected_ligand_smiles",
            "work_dir",
        }:
            raise ValueError("request schema mismatch")
        metric = _load_metric(args.metric)
        prediction = metric.load_prediction_json(Path(request["prediction"]))
        score = metric.score_pose(
            prediction,
            crystal_ligand_path=Path(request["crystal_ligand"]),
            crystal_protein_path=Path(request["crystal_protein"]),
            expected_chains=request["expected_chains"],
            expected_ligand_smiles=request["expected_ligand_smiles"],
            work_dir=Path(request["work_dir"]),
        )
        _write_exclusive(
            args.output,
            {
                "passed": bool(score.passed),
                "pb_valid": bool(score.pb_valid),
                "rmsd_within_2a": bool(score.rmsd_within_2a),
            },
        )
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
