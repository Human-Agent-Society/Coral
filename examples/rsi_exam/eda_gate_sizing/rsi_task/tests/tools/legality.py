"""OpenROAD-free legality gate for a ``.size`` submission.

This is the *fast, free* contract check the agent can run unlimited times. It
enforces exactly the structural rules the grader also enforces before it hands
the sizing to the (heavier) power/timing evaluator:

  * every instance in the case appears exactly once,
  * no unknown instance names, no unknown library cells,
  * sequential cells and macros keep their original library cell,
  * combinational cells may only move to a SAME ``func_id`` library cell.

It does NOT compute the real score — power under timing/DRV constraints needs
the timing engine (see the grader). It also reports an IR-table leakage proxy
purely as a sanity signal; treat it as indicative only.
"""
from __future__ import annotations

import bz2
import csv
from pathlib import Path


def _ff(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def read_libcells(path: Path) -> dict:
    out = {}
    with bz2.open(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            out[row["libcell_name"]] = {
                "func_id": int(row["func_id"]),
                "leakage": _ff(row.get("libcell_leakage")),
            }
    return out


def read_cells(path: Path) -> dict:
    out = {}
    with bz2.open(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            out[row["cell_name"]] = {
                "libcell_name": row["libcell_name"],
                "fixed": _ff(row["is_seq"]) != 0.0 or _ff(row["is_macro"]) != 0.0,
            }
    return out


def parse_submission(path: Path):
    mapping, errors = {}, []
    if not path.exists():
        return mapping, [f"missing output file: {path}"]
    with path.open() as f:
        for ln, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                errors.append(f"line {ln}: expected 2 columns, got {len(parts)}")
                continue
            inst, lib = parts
            if inst in mapping:
                errors.append(f"line {ln}: duplicate instance {inst}")
                continue
            mapping[inst] = lib
    return mapping, errors


def check_case(ir_dir: Path, size_path: Path) -> dict:
    """Return a legality report for one case's ``.size`` file."""
    libcells = read_libcells(ir_dir / "libcell_properties.csv.bz2")
    cells = read_cells(ir_dir / "cell_properties.csv.bz2")
    submitted, errors = parse_submission(size_path)

    missing = set(cells) - set(submitted)
    unexpected = set(submitted) - set(cells)
    if missing:
        errors.append(f"missing {len(missing)} instance(s)")
    if unexpected:
        errors.append(f"contains {len(unexpected)} unknown instance(s)")

    illegal = []
    orig_leak = sub_leak = 0.0
    for inst, info in cells.items():
        ol = str(info["libcell_name"])
        op = libcells.get(ol)
        if op:
            orig_leak += op["leakage"]
        nl = submitted.get(inst)
        if nl is None:
            continue
        npp = libcells.get(nl)
        if npp is None:
            illegal.append(f"{inst}: unknown libcell {nl}")
            continue
        sub_leak += npp["leakage"]
        if info["fixed"]:
            if nl != ol:
                illegal.append(f"{inst}: fixed cell must stay {ol}, got {nl}")
            continue
        if op is None:
            illegal.append(f"{inst}: original libcell {ol} not in table")
            continue
        if npp["func_id"] != op["func_id"]:
            illegal.append(f"{inst}: func_id mismatch {ol}->{nl}")

    if illegal:
        errors.append(f"contains {len(illegal)} illegal assignment(s)")

    return {
        "legal": not errors,
        "expected_instances": len(cells),
        "submitted_instances": len(submitted),
        "proxy_leakage_delta_uw": (sub_leak - orig_leak) * 1_000_000.0,
        "errors": errors[:50],
        "illegal_examples": illegal[:20],
    }
