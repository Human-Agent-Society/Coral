#!/usr/bin/env python3
"""Replace invalid verifier artifacts with a zero reward, preserving grader faults."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any


REWARD_DIR = Path("/logs/verifier")
MAX_DIAGNOSTIC_BYTES = 1 << 20
GENERIC_MARKER = "grader failed: verifier entrypoint or output validation failed"


def _read_grader_marker(path: Path) -> str | None:
    """Return only a bounded, regular-file grader marker from an old artifact."""

    try:
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or not 1 <= info.st_size <= MAX_DIAGNOSTIC_BYTES
        ):
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    marker = payload.get("error")
    if isinstance(marker, str) and marker.startswith("grader failed:"):
        return marker
    return None


def _atomic_write(path: Path, payload: dict[str, Any] | str) -> None:
    data = (
        (json.dumps(payload, allow_nan=False, sort_keys=True) + "\n").encode()
        if isinstance(payload, dict)
        else payload.encode()
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def write_zero_results(directory: Path = REWARD_DIR) -> str:
    """Zero both reward files and retain only a genuine grader-failure marker."""

    marker = next(
        (
            value
            for name in ("grade_debug.json", "score_details.json")
            if (value := _read_grader_marker(directory / name)) is not None
        ),
        GENERIC_MARKER,
    )
    _atomic_write(directory / "reward.json", {"reward": 0.0})
    _atomic_write(directory / "reward.txt", "0.0\n")
    _atomic_write(
        directory / "grade_debug.json",
        {"correctness": False, "error": marker, "reward": 0.0},
    )
    _atomic_write(
        directory / "score_details.json",
        {
            "correctness": False,
            "error": marker,
            "metric": "mean_video_average_jaccard_under_four_switch_budget",
            "raw_metric": None,
            "reward": 0.0,
        },
    )
    return marker


if __name__ == "__main__":
    write_zero_results()
