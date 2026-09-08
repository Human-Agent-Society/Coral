#!/usr/bin/env python3
"""Secure fail-closed reward publication shared by the grader and shell wrapper."""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

LOG_DIR = Path("/logs/verifier")
OUTPUT_NAMES = ("reward.txt", "reward.json", "score_details.json", "grade_debug.json")
MAX_MARKER_BYTES = 64 * 1024


def atomic_write(path: Path, text: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chown(path, 0, 0)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _remove_path(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
        shutil.rmtree(path)
    else:
        path.unlink()


def _secure_log_dir(log_dir: Path) -> None:
    if os.geteuid() != 0:
        raise RuntimeError("grader failed: verifier must run as root")
    parent = log_dir.parent
    try:
        parent_info = parent.lstat()
    except FileNotFoundError:
        parent.mkdir(mode=0o755)
        parent_info = parent.lstat()
    if not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode):
        raise RuntimeError("grader failed: verifier log parent is unsafe")
    os.chown(parent, 0, 0)
    os.chmod(parent, 0o755)
    try:
        leaf_info = log_dir.lstat()
    except FileNotFoundError:
        os.mkdir(log_dir, 0o700)
    else:
        if not stat.S_ISDIR(leaf_info.st_mode) or stat.S_ISLNK(leaf_info.st_mode):
            _remove_path(log_dir)
            os.mkdir(log_dir, 0o700)
    os.chown(log_dir, 0, 0)
    os.chmod(log_dir, 0o700)


def prepare_log_dir(log_dir: Path = LOG_DIR) -> None:
    _secure_log_dir(log_dir)
    for entry in list(os.scandir(log_dir)):
        _remove_path(Path(entry.path))


def _trusted_marker(log_dir: Path) -> str | None:
    for name in ("grade_debug.json", "score_details.json"):
        path = log_dir / name
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != 0
            or info.st_gid != 0
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 1
            or info.st_size > MAX_MARKER_BYTES
        ):
            continue
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
            try:
                opened = os.fstat(descriptor)
                if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_ctime_ns) != (
                    info.st_dev, info.st_ino, info.st_size, info.st_ctime_ns
                ):
                    continue
                payload = os.read(descriptor, info.st_size + 1)
            finally:
                os.close(descriptor)
            value = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        marker = value.get("error") if isinstance(value, dict) else None
        if isinstance(marker, str) and marker.startswith("grader failed:"):
            return marker[:4096]
    return None


def write_failure_outputs(
    marker: str,
    *,
    log_dir: Path = LOG_DIR,
    preserve_existing: bool = False,
) -> None:
    _secure_log_dir(log_dir)
    if preserve_existing:
        marker = _trusted_marker(log_dir) or marker
    if not isinstance(marker, str) or not marker.startswith("grader failed:"):
        marker = "grader failed: verifier entrypoint failed"
    marker = marker[:4096]
    for entry in list(os.scandir(log_dir)):
        _remove_path(Path(entry.path))
    details = {
        "correctness": False,
        "metric": "macro_f1",
        "raw_metric": 0.0,
        "reward": 0.0,
        "error": marker,
    }
    debug = {**details, "status": "grader_failure"}
    atomic_write(log_dir / "reward.txt", "0.0\n")
    atomic_write(log_dir / "reward.json", '{"reward":0.0}\n')
    atomic_write(log_dir / "score_details.json", json.dumps(details, allow_nan=False, sort_keys=True) + "\n")
    atomic_write(log_dir / "grade_debug.json", json.dumps(debug, allow_nan=False, sort_keys=True) + "\n")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--initialize":
        write_failure_outputs("grader failed: verifier grading did not complete")
        return 0
    return_code = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    write_failure_outputs(
        f"grader failed: grade.py exited with status {return_code}",
        preserve_existing=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
