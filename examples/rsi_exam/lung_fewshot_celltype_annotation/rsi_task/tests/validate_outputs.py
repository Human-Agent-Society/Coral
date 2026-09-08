#!/usr/bin/env python3
"""Validate the exact coherent, root-owned Harbor reward output set."""

from __future__ import annotations

import json
import math
import os
import stat
from pathlib import Path

LOG_DIR = Path("/logs/verifier")
OUTPUT_NAMES = {"reward.txt", "reward.json", "score_details.json", "grade_debug.json"}
MAX_BYTES = 4 * 1024 * 1024


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_uid,
        info.st_gid,
        info.st_mode,
        info.st_nlink,
    )


def _read_regular(path: Path) -> str:
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != 0
        or info.st_gid != 0
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_nlink != 1
        or not 0 <= info.st_size <= MAX_BYTES
    ):
        raise ValueError(f"invalid output metadata: {path.name}")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if _identity(before) != _identity(info):
            raise ValueError(f"output changed while opening: {path.name}")
        payload = bytearray()
        while len(payload) <= MAX_BYTES:
            chunk = os.read(descriptor, min(1 << 20, MAX_BYTES + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if len(payload) != info.st_size or _identity(after) != _identity(before):
            raise ValueError(f"output changed while reading: {path.name}")
    finally:
        os.close(descriptor)
    return bytes(payload).decode("utf-8")


def _strict_object(text: str, name: str) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"duplicate key in {name}: {key}")
            value[key] = item
        return value

    value = json.loads(text, object_pairs_hook=pairs, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def validate_outputs(log_dir: Path = LOG_DIR) -> float:
    info = log_dir.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != 0
        or info.st_gid != 0
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError("verifier log leaf is unsafe")
    names = {entry.name for entry in os.scandir(log_dir)}
    if names != OUTPUT_NAMES:
        raise ValueError("verifier log leaf must contain exactly four outputs")
    reward_json = _strict_object(_read_regular(log_dir / "reward.json"), "reward.json")
    if set(reward_json) != {"reward"}:
        raise ValueError("reward.json must contain exactly reward")
    reward_value = reward_json["reward"]
    if isinstance(reward_value, bool) or not isinstance(reward_value, (int, float)):
        raise ValueError("reward.json reward is not numeric")
    reward = float(reward_value)
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        raise ValueError("reward.json reward is outside [0,1]")
    text_reward = float(_read_regular(log_dir / "reward.txt").strip())
    if not math.isfinite(text_reward) or text_reward != reward:
        raise ValueError("reward.txt and reward.json disagree")
    details = _strict_object(_read_regular(log_dir / "score_details.json"), "score_details.json")
    debug = _strict_object(_read_regular(log_dir / "grade_debug.json"), "grade_debug.json")
    required = {"correctness", "metric", "raw_metric", "reward"}
    if not required.issubset(details) or not required.issubset(debug):
        raise ValueError("aggregate verifier outputs lack required fields")
    if details["metric"] != "macro_f1" or debug["metric"] != "macro_f1":
        raise ValueError("aggregate verifier metric is inconsistent")
    for payload in (details, debug):
        value = float(payload["reward"])
        raw = float(payload["raw_metric"])
        if not math.isfinite(value) or not math.isfinite(raw) or value != reward:
            raise ValueError("aggregate verifier outputs disagree")
    return reward


def main() -> int:
    validate_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
