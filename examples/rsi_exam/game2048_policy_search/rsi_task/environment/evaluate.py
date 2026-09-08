#!/usr/bin/env python3
"""Evaluate a 2048 policy on a fixed seed suite."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import statistics
from types import ModuleType

from game2048 import play


def load_policy(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("submitted_policy", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import policy from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "choose_move", None)):
        raise ValueError("policy.py must define callable choose_move(board)")
    return module


def load_suite(path: Path) -> tuple[list[int], int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in data["seeds"]]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seed suite must contain unique seeds")
    return seeds, int(data.get("max_moves", 10000))


def evaluate(policy_path: Path, suite_path: Path) -> dict[str, object]:
    policy = load_policy(policy_path)
    seeds, max_moves = load_suite(suite_path)
    games = [play(seed, policy.choose_move, max_moves=max_moves) for seed in seeds]
    scores = [game.score for game in games]
    return {
        "mean_score": statistics.fmean(scores),
        "median_score": statistics.median(scores),
        "mean_max_tile": statistics.fmean(game.max_tile for game in games),
        "valid_fraction": statistics.fmean(game.error is None for game in games),
        "instances": [game.__dict__ for game in games],
    }
