#!/usr/bin/env python3
"""OpenROAD-side per-case evaluator. Run as: `openroad -python evaluate_case.py
--case_dir <dir> --size_file <f>`; prints a single `SCORE: <x>` line.

INTEGRATION POINT (unverified without OpenROAD). It reuses the public
benchmark's BSD-licensed `evaluation.py` / `check_validity.py` / OpenROAD helper
(vendored alongside this file) to compute the contest score:

    score = leakage_uW + 10*|TNS| + 20*(slew viol) + 20*(cap viol)   (lower better)

The vendored `OpenROAD_helper.load_design(name)` expects, relative to CWD:
    design/<name>/<name>.{v,def,sdc}   and   ../../platform/ASAP7/{lib,lef,...}
This wrapper stages a temp tree matching that layout from the neutral case dir
(<case_dir>/design/<case>.{v,def,sdc}.bz2 + <case_dir>/../platform/ASAP7), then
calls the vendored evaluator. Validate end-to-end when the OpenROAD verifier
image is first built, then measure the BASELINE/REFERENCE anchors (task.toml).
"""
from __future__ import annotations

import argparse
import bz2
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _decompress(src_bz2: Path, dst: Path) -> None:
    with bz2.open(src_bz2, "rb") as f, dst.open("wb") as w:
        shutil.copyfileobj(f, w)


def _stage(case_dir: Path, work: Path) -> tuple[str, Path]:
    """Recreate the layout load_design() expects and return (design_name, cwd).

    load_design resolves everything relative to CWD:
        ../../platform/ASAP7/{lib,lef,setRC.tcl}
        ../../design/<name>/<name>.{def,sdc,v}
    so CWD must sit exactly two levels under a root holding platform/ + design/.
    """
    name = "design0"
    root = work                      # holds platform/ and design/
    cwd = work / "cwd" / "here"       # two levels down -> ../.. == root
    cwd.mkdir(parents=True, exist_ok=True)

    ddir = root / "design" / name
    ddir.mkdir(parents=True, exist_ok=True)
    src_design = case_dir / "design"
    for suffix in (".v", ".def", ".sdc"):
        cands = list(src_design.glob(f"*{suffix}.bz2")) + list(src_design.glob(f"*{suffix}"))
        if not cands:
            if suffix == ".v":
                continue  # DEF flow does not need the netlist
            raise FileNotFoundError(f"missing {suffix} in {src_design}")
        src = cands[0]
        if src.suffix == ".bz2":
            _decompress(src, ddir / f"{name}{suffix}")
        else:
            shutil.copy2(src, ddir / f"{name}{suffix}")

    shutil.copytree(case_dir.parent / "platform" / "ASAP7", root / "platform" / "ASAP7")
    return name, cwd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_dir", type=Path, required=True)
    ap.add_argument("--size_file", type=Path, required=True)
    args = ap.parse_args()

    work = Path(os.environ.get("EVAL_WORKDIR", "/tmp/gs_eval_work"))
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    name, cwd = _stage(args.case_dir, work)

    # load_design uses CWD-relative paths (../../platform, ../../design/<name>).
    size_abs = str(args.size_file.resolve())
    os.chdir(cwd)
    from OpenROAD_helper import load_design  # noqa: E402  (needs openroad runtime)
    from openroad import Timing  # noqa: E402
    from evaluation import ICCAD_evaluation  # noqa: E402

    tech, design = load_design(name, False)
    timing = Timing(design)
    # ICCAD_evaluation prints "Score: <x>"; re-emit as the SCORE: line score.py parses.
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ICCAD_evaluation(size_abs, design, timing)
    text = buf.getvalue()
    print(text)
    for line in text.splitlines():
        if line.strip().startswith("Score:"):
            print(f"SCORE: {line.split(':',1)[1].strip()}", flush=True)
            os._exit(0)  # avoid OpenROAD's -python SystemExit traceback on clean exit
    print("SCORE: parse_error", file=sys.stderr, flush=True)
    os._exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
