#!/usr/bin/env python3
"""Untrusted-side entry point for one anonymized co-folding case.

The trusted grader starts this immutable launcher as root so it can create a
private per-case mount namespace.  The launcher permanently becomes the
unprivileged ``runner`` user and applies Landlock/seccomp before it reads the
case or imports the submission.  Its sole output is one JSON file in the
current case scratch.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import grade as trusted_sandbox


MAX_CHILD_JSON_BYTES = 24 * 1024 * 1024
SANDBOX_SETUP_EXIT = 125


def _load_item(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        item = json.load(handle)
    if not isinstance(item, dict):
        raise ValueError("case input must be a JSON object")
    if set(item) != {"protein_chains", "ligand_smiles", "msa_dir"}:
        raise ValueError("case input has an invalid schema")
    return item


def _load_predictor(solver_path: Path):
    """Import submitted code inside the untrusted child only."""
    if solver_path.name != "solver.py" or not solver_path.is_file():
        raise ValueError("submission must contain solver.py")
    sys.argv = [str(solver_path)]
    sys.path.insert(0, str(solver_path.parent))
    spec = importlib.util.spec_from_file_location("submitted_solver", solver_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot create a module spec for solver.py")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
    predictor = getattr(module, "predict_complex", None)
    if not callable(predictor):
        raise TypeError("solver.py must define callable predict_complex(item)")
    return predictor


def _write_result(path: Path, prediction: Any) -> None:
    # The parent performs the authoritative schema, inode, size, parser, and
    # finite-coordinate checks.  The child-side cap avoids needlessly writing
    # an obviously unusable payload.
    payload = json.dumps(
        prediction,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if not payload or len(payload) > MAX_CHILD_JSON_BYTES:
        raise ValueError("prediction JSON is empty or too large")

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write while emitting prediction")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--item", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner-uid", type=int, required=True)
    parser.add_argument("--runner-gid", type=int, required=True)
    args = parser.parse_args(argv)

    # Apply the filesystem policy in this still-trusted launcher, before the
    # item is read or any submission module is imported.  Keeping Python code
    # out of Popen(preexec_fn=...) avoids post-fork deadlocks after RDKit,
    # NumPy, CUDA, or PoseBusters have initialized threads in the parent.
    try:
        scratch = args.scratch.resolve(strict=True)
        for path in (args.solver, args.item):
            path.resolve(strict=True).relative_to(scratch)
        args.output.parent.resolve(strict=True).relative_to(scratch)
        private_shm_device = trusted_sandbox.prepare_private_shm_namespace(
            args.runner_uid, args.runner_gid
        )
        trusted_sandbox.drop_child_privileges(args.runner_uid, args.runner_gid)
        trusted_sandbox.restrict_child_filesystem(
            scratch, private_shm_device=private_shm_device
        )
        trusted_sandbox.restrict_child_persistent_ipc()
    except Exception:  # fail closed with a parent-recognizable isolation code
        return SANDBOX_SETUP_EXIT
    # The helper module contains no loaded cases or anchors, but remove even
    # its inert API before importing adversarial code.  Landlock also prevents
    # reopening /tests/grade.py, because only this child script was granted.
    sys.modules.pop("grade", None)
    globals().pop("trusted_sandbox", None)

    item = _load_item(args.item)
    predictor = _load_predictor(args.solver)
    with contextlib.redirect_stdout(sys.stderr):
        prediction = predictor(item)
    _write_result(args.output, prediction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
