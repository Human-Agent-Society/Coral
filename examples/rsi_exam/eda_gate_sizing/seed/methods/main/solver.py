"""Starting template for gate sizing. Going from here to strong is the task.

The grader re-runs ``solve`` on sealed held-out cases, so submit an ALGORITHM,
not precomputed answers. Keep the exact signature.

``solve(input_dir, output_path)``:
  * ``input_dir`` — one case directory containing ``IR_Tables/`` (at least
    ``cell_properties.csv.bz2`` and ``libcell_properties.csv.bz2``) and
    ``design/`` (compressed ``.v``/``.def``/``.sdc``). The shared cell library
    lives at ``<input_dir>/../platform/ASAP7``.
  * ``output_path`` — where to write the ``.size`` file: one line per instance,
    ``<instance_name> <library_cell_name>``.

WHAT THIS TEMPLATE DOES — read it, because it is the score you have to beat.
The score is dominated by timing/DRV violations, with leakage as a smaller
counterweight. `choose_cell` below decides the library cell for ONE instance and
is handed everything the IR tables know about that instance. The rule shipped in
it throws all of that away and looks only at the instance's ``func_id``, so every
instance of a gate type gets the same cell: the STRONGEST-DRIVE one, ranked by
``fo4_delay``. That fixes many transition/slew and load-capacitance violations
and pulls TNS toward zero, and pays for all of it in leakage everywhere,
including on paths that had slack to spare and never needed the drive.

It is uniform where the real problem is local: a gate on a critical path and a
gate three levels off it get the same answer, so nothing here can size up only
where that buys timing and size down where slack is plentiful. That gap is the
task. Nothing about this file is load-bearing except the ``solve`` signature and
the ``.size`` format — rewrite it, or delete it and start from an empty file.

Sequential cells and macros must keep their original library cell; combinational
cells may only move to a library cell with the SAME ``func_id``.
"""
from __future__ import annotations

import bz2
import csv
from pathlib import Path


def _ff(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _read_libcells(ir: Path):
    lib = {}
    with bz2.open(ir / "libcell_properties.csv.bz2", "rt", newline="") as f:
        for row in csv.DictReader(f):
            lib[row["libcell_name"]] = {
                "func_id": int(row["func_id"]),
                "fo4": _ff(row.get("fo4_delay")),
            }
    return lib


def choose_cell(row, lib, strongest) -> str:
    """Pick the library cell for ONE instance. This is the whole task.

    ``row`` is that instance's full ``cell_properties`` record, so anything the
    tables know about it is available here. The shipped rule uses none of it
    beyond ``func_id``.
    """
    orig = row["libcell_name"]
    if (_ff(row["is_seq"]) or 0) != 0 or (_ff(row["is_macro"]) or 0) != 0:
        return orig                      # sequential cells and macros are fixed
    op = lib.get(orig)
    if op is None:
        return orig
    return strongest.get(op["func_id"], orig)


def solve(input_dir: str, output_path: str) -> None:
    ir = Path(input_dir) / "IR_Tables"
    lib = _read_libcells(ir)

    # strongest-drive (min fo4_delay) cell per func_id
    strongest: dict[int, str] = {}
    for name, p in lib.items():
        if p["fo4"] is None:
            continue
        fid = p["func_id"]
        cur = strongest.get(fid)
        if cur is None or p["fo4"] < lib[cur]["fo4"]:
            strongest[fid] = name

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with bz2.open(ir / "cell_properties.csv.bz2", "rt", newline="") as f, out.open("w") as w:
        for row in csv.DictReader(f):
            w.write(f"{row['cell_name']} {choose_cell(row, lib, strongest)}\n")

if __name__ == "__main__":
    import sys

    solve(sys.argv[1], sys.argv[2])
