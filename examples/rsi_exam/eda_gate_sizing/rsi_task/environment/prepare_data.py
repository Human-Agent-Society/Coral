#!/usr/bin/env python3
"""Fetch + anonymize + split the public gate-sizing benchmark into task-ready cases.

Fixed, deterministic processing script (README S6.2, second method): it takes a raw
checkout of the public source repository and emits the neutral, de-identified
data this task ships to the agent (visible) and the grader (held-out).

What it strips (anti-cheat, README §9):
  * the reference ``*.size`` answer files that ship inside ``design/<d>/``
    (the source repo bundles a solved output per design — the answer key)
  * the contest PDFs, top-teams spreadsheet, and the source ``README.md``
    (all of which name the contest and link the winning solution repo)
  * the original ``src/`` tutorial/eval scripts that name the contest

What it renames:
  * every design directory ``<real-name>/`` -> a neutral ``caseN/`` id, so the
    instruction and on-disk paths never expose which public design this is
    (the internal netlist module names are left intact — see NOTE below).

NOTE / residual leak: instance/module names *inside* the netlists still carry
the original design identity, so this is not cryptographic anonymization. The
primary anti-cheat guarantee for a real run is network isolation (README §7.2
egress overlay) + the sealed held-out split; this script removes the trivial
look-ups (answer files, contest framing, design->case mapping).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# Deterministic design -> neutral-case map and the visible/held-out split.
# Sizes (post-placement instance counts) chosen so the visible set spans the
# held-out set (README/skill §4③ distribution consistency): visible covers the
# small / large extremes, held-out sits in the middle.
DESIGNS = {
    # real source name          neutral id   split       ~cells
    "NV_NVDLA_partition_m": ("case1", "visible"),   # ~27k  (small)
    "aes_256":              ("case2", "visible"),   # ~278k (large)
    "ariane136":            ("case3", "visible"),   # ~146k (mid-large)
    "NV_NVDLA_partition_p": ("case4", "heldout"),   # ~80k  (mid)
    "mempool_tile_wrap":    ("case5", "heldout"),   # ~188k (mid-large)
}

# Per design we keep only these (compressed) inputs. Reference ``*.size`` answer
# files that the source bundles under design/<d>/ are intentionally NOT listed.
DESIGN_INPUT_SUFFIXES = (".v.bz2", ".def.bz2", ".sdc.bz2")


def _copy_ir_tables(src_ir: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_ir.glob("*.csv.bz2")):
        shutil.copy2(f, dst / f.name)


def _copy_design_inputs(src_design: Path, dst: Path, neutral_id: str) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_design.iterdir()):
        if not f.is_file():
            continue
        low = f.name.lower()
        if not low.endswith(DESIGN_INPUT_SUFFIXES):
            continue  # skips *.size / *.size.bz2 answer files and anything else
        # rename <real>.v.bz2 -> <caseN>.v.bz2 so the filename is neutral too
        suffix = "".join(f.suffixes[-2:]) if len(f.suffixes) >= 2 else f.suffix
        shutil.copy2(f, dst / f"{neutral_id}{suffix}")


def build(src: Path, out: Path, split: str) -> None:
    """Emit ``out/<caseN>/{IR_Tables,design}`` for every design in ``split``
    (``visible`` | ``heldout`` | ``all``) plus the shared ASAP7 platform once."""
    out.mkdir(parents=True, exist_ok=True)
    selected = []
    for real, (neutral, design_split) in sorted(DESIGNS.items(), key=lambda kv: kv[1][0]):
        if split != "all" and design_split != split:
            continue
        case_dir = out / neutral
        _copy_ir_tables(src / "IR_Tables" / real, case_dir / "IR_Tables")
        _copy_design_inputs(src / "design" / real, case_dir / "design", neutral)
        selected.append(neutral)

    # shared cell library (needed for legality + OpenROAD scoring), copied once
    platform_src = src / "platform" / "ASAP7"
    platform_dst = out / "platform" / "ASAP7"
    if platform_src.exists() and not platform_dst.exists():
        shutil.copytree(platform_src, platform_dst)

    (out / "manifest.json").write_text(
        json.dumps({"cases": sorted(selected), "split": split}, indent=2) + "\n"
    )
    print(f"[prepare_data] split={split} cases={sorted(selected)} -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="raw checkout of the public benchmark repo")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", choices=["visible", "heldout", "all"], required=True)
    args = ap.parse_args()
    build(args.src, args.out, args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
