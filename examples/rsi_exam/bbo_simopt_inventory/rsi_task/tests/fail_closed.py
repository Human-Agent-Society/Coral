#!/usr/bin/env python3
"""Secure verifier log-leaf setup and fail-closed output writing."""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any


LOG_DIR = Path("/logs/verifier")
OUTPUT_NAMES = ("reward.txt", "reward.json", "score_details.json", "grade_debug.json")
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_MARKER_BYTES = 64 * 1024
METRIC = "mean_log_cost_progress_auc70_final30"
AGGREGATION = "median repeated trace per instance; per-instance log1p progress; 70% anytime plus 30% final; normalized complement-log mapping per instance; arithmetic mean"

def empty_score_details() -> dict[str, Any]:
    return {"metric": METRIC, "direction": "higher", "aggregation": AGGREGATION, "instances": [], "aggregate": {"raw_metric": 0.0, "baseline": 0.0, "reference": 0.0, "upper_bound": 1.0, "median_final_cost_diagnostic": 0.0, "reward": 0.0}}


def _remove_path(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISDIR(info.st_mode) and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _clear_log_dir(log_dir: Path) -> None:
    for entry in os.scandir(log_dir):
        _remove_path(Path(entry.path))


def _secure_log_dir(log_dir: Path = LOG_DIR) -> None:
    if os.geteuid() != 0:
        raise RuntimeError("grader failed: verifier must run as root")
    parent = log_dir.parent
    try:
        parent_info = parent.lstat()
    except FileNotFoundError:
        parent.mkdir(mode=0o755)
        parent_info = parent.lstat()
    if not stat.S_ISDIR(parent_info.st_mode) or parent.is_symlink():
        raise RuntimeError("grader failed: verifier log parent is unsafe")

    try:
        info = log_dir.lstat()
    except FileNotFoundError:
        os.mkdir(log_dir, 0o700)
    else:
        if not stat.S_ISDIR(info.st_mode) or log_dir.is_symlink():
            _remove_path(log_dir)
            os.mkdir(log_dir, 0o700)
    os.chown(log_dir, 0, 0)
    os.chmod(log_dir, 0o700)
    verified = log_dir.lstat()
    if (
        not stat.S_ISDIR(verified.st_mode)
        or log_dir.is_symlink()
        or (verified.st_uid, verified.st_gid) != (0, 0)
        or stat.S_IMODE(verified.st_mode) != 0o700
    ):
        raise RuntimeError("grader failed: verifier log leaf is unsafe")


def prepare_log_dir(log_dir: Path = LOG_DIR) -> None:
    _secure_log_dir(log_dir)
    _clear_log_dir(log_dir)


def atomic_write(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chown(path, 0, 0, follow_symlinks=False)
        os.chmod(path, 0o600)
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or (info.st_uid, info.st_gid) != (0, 0)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size > MAX_OUTPUT_BYTES
        ):
            raise RuntimeError("grader failed: verifier output is unsafe")
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _trusted_existing_marker(log_dir: Path) -> str | None:
    for name in ("grade_debug.json", "score_details.json"):
        path = log_dir / name
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if (
            not stat.S_ISREG(info.st_mode)
            or path.is_symlink()
            or (info.st_uid, info.st_gid) != (0, 0)
            or info.st_nlink != 1
            or info.st_size > MAX_MARKER_BYTES
            or stat.S_IMODE(info.st_mode) & 0o022
        ):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        candidates = payload.get("errors")
        if isinstance(candidates, list):
            for value in candidates:
                if isinstance(value, str) and value.startswith("grader failed:"):
                    return value[:1000]
        value = payload.get("error")
        if isinstance(value, str) and value.startswith("grader failed:"):
            return value[:1000]
    return None


def write_failure_outputs(
    marker: str,
    *,
    log_dir: Path = LOG_DIR,
    preserve_existing: bool = False,
) -> None:
    _secure_log_dir(log_dir)
    if preserve_existing:
        marker = _trusted_existing_marker(log_dir) or marker
    if not isinstance(marker, str) or not marker.startswith("grader failed:"):
        marker = "grader failed: verifier entrypoint failed"
    marker = marker[:1000]
    _clear_log_dir(log_dir)
    atomic_write(log_dir / "reward.txt", "0.0\n")
    atomic_write(
        log_dir / "reward.json",
        json.dumps({"reward": 0.0}, allow_nan=False, sort_keys=True) + "\n",
    )
    atomic_write(
        log_dir / "score_details.json",
        json.dumps(empty_score_details(), allow_nan=False, sort_keys=True) + "\n",
    )
    atomic_write(
        log_dir / "grade_debug.json",
        json.dumps(
            {
                "reward": 0.0,
                "best_objective": 0.0,
                "num_evals": 0,
                "correctness": False,
                "errors": [marker],
            },
            allow_nan=False,
            sort_keys=True,
        )
        + "\n",
    )


def publish_log_dir(log_dir: Path = LOG_DIR) -> None:
    """Transfer final outputs to the trusted mount owner after grading is over."""
    _secure_log_dir(log_dir)
    names = {entry.name for entry in os.scandir(log_dir)}
    if names != set(OUTPUT_NAMES):
        raise RuntimeError("grader failed: verifier output set is not exact")
    parent_info = log_dir.parent.lstat()
    if not stat.S_ISDIR(parent_info.st_mode) or log_dir.parent.is_symlink():
        raise RuntimeError("grader failed: verifier log parent is unsafe")
    for name in OUTPUT_NAMES:
        path = log_dir / name
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or path.is_symlink()
            or (info.st_uid, info.st_gid) != (0, 0)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size > MAX_OUTPUT_BYTES
        ):
            raise RuntimeError("grader failed: verifier output metadata is unsafe")
    for name in OUTPUT_NAMES:
        os.chown(log_dir / name, parent_info.st_uid, parent_info.st_gid)
        os.chmod(log_dir / name, 0o644)
    os.chown(log_dir, parent_info.st_uid, parent_info.st_gid)
    os.chmod(log_dir, 0o755)


def main() -> int:
    if sys.argv[1:] == ["--initialize"]:
        write_failure_outputs("grader failed: verifier did not complete")
        publish_log_dir()
        return 0
    if sys.argv[1:] == ["--publish"]:
        publish_log_dir()
        return 0
    return_code = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    write_failure_outputs(
        f"grader failed: verifier entrypoint failed closed (exit {return_code})",
        preserve_existing=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
