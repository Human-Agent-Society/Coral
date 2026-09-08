#!/usr/bin/env python3
"""Validate the exact coherent root-owned SimOpt verifier output set."""

from __future__ import annotations

import json
import math
import os
import stat
from pathlib import Path
from typing import Any

from fail_closed import AGGREGATION, LOG_DIR, MAX_OUTPUT_BYTES, METRIC, OUTPUT_NAMES


def _read_regular(path: Path) -> str:
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or path.is_symlink()
        or (info.st_uid, info.st_gid) != (0, 0)
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_nlink != 1
        or info.st_size > MAX_OUTPUT_BYTES
    ):
        raise ValueError(f"invalid output metadata: {path.name}")
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        data = os.read(descriptor, info.st_size + 1)
    finally:
        os.close(descriptor)
    if len(data) != info.st_size:
        raise ValueError(f"output changed while reading: {path.name}")
    return data.decode("utf-8")


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def validate_outputs(log_dir: Path = LOG_DIR) -> float:
    info = log_dir.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or log_dir.is_symlink()
        or (info.st_uid, info.st_gid) != (0, 0)
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError("invalid verifier log leaf metadata")
    if {entry.name for entry in os.scandir(log_dir)} != set(OUTPUT_NAMES):
        raise ValueError("verifier output set is not exact")

    reward_doc = json.loads(_read_regular(log_dir / "reward.json"))
    if type(reward_doc) is not dict or set(reward_doc) != {"reward"}:
        raise ValueError("reward.json must contain exactly reward")
    reward = reward_doc["reward"]
    if not _finite_number(reward) or not 0.0 <= float(reward) <= 1.0:
        raise ValueError("reward.json reward is invalid")
    reward = float(reward)
    if _read_regular(log_dir / "reward.txt") != f"{reward!r}\n":
        raise ValueError("reward.txt and reward.json disagree")

    details = json.loads(_read_regular(log_dir / "score_details.json"))
    if type(details) is not dict or set(details) != {"metric", "direction", "aggregation", "instances", "aggregate"}: raise ValueError("score_details.json has an invalid schema")
    if details["metric"] != METRIC or details["direction"] != "higher" or details["aggregation"] != AGGREGATION: raise ValueError("score_details.json metric contract mismatch")
    if type(details["instances"]) is not list: raise ValueError("score_details.json instances must be a list")
    instance_keys={"id","raw_metric","raw_final_cost","baseline_final_cost","reference_raw_metric","reference_final_cost","upper_bound_raw_metric","upper_bound_final_cost","reward","anytime_raw","final_raw"}
    for instance in details["instances"]:
        if type(instance) is not dict or set(instance)!=instance_keys: raise ValueError("score_details.json instance schema mismatch")
        if type(instance["id"]) is not str or not instance["id"]: raise ValueError("score_details.json instance id is invalid")
        if not all(_finite_number(instance[k]) for k in instance_keys-{"id"}): raise ValueError("score_details.json instance contains non-finite data")
        if not all(0.0<=float(instance[k])<=1.0 for k in ("raw_metric","reference_raw_metric","upper_bound_raw_metric","reward","anytime_raw","final_raw")): raise ValueError("score_details.json normalized value is invalid")
        if not all(float(instance[k])>=0.0 for k in ("raw_final_cost","baseline_final_cost","reference_final_cost","upper_bound_final_cost")): raise ValueError("score_details.json cost is invalid")
    aggregate=details["aggregate"];aggregate_keys={"raw_metric","baseline","reference","upper_bound","median_final_cost_diagnostic","reward"}
    if type(aggregate) is not dict or set(aggregate)!=aggregate_keys or not all(_finite_number(aggregate[k]) for k in aggregate_keys): raise ValueError("score_details.json aggregate schema mismatch")
    if not all(0.0<=float(aggregate[k])<=1.0 for k in ("raw_metric","baseline","reference","upper_bound","reward")): raise ValueError("score_details.json aggregate normalization is invalid")
    if float(aggregate["median_final_cost_diagnostic"])<0.0 or float(aggregate["reward"])!=reward: raise ValueError("score_details.json aggregate is incoherent")

    debug = json.loads(_read_regular(log_dir / "grade_debug.json"))
    if type(debug) is not dict or not _finite_number(debug.get("reward")):
        raise ValueError("grade_debug.json schema mismatch")
    if float(debug["reward"]) != reward:
        raise ValueError("grade_debug.json reward disagrees")
    if type(debug.get("correctness")) is not bool:
        raise ValueError("grade_debug.json correctness is invalid")
    if type(debug.get("errors")) is not list or not all(
        type(error) is str for error in debug["errors"]
    ):
        raise ValueError("grade_debug.json errors are invalid")
    if debug["correctness"]:
        if debug.get("score_details") != details:
            raise ValueError("grade_debug.json score details disagree")
    elif reward != 0.0 or details["instances"]:
        raise ValueError("failed grading must publish the empty zero schema")
    return reward


if __name__ == "__main__":
    validate_outputs()
