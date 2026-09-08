#!/usr/bin/env python3
"""Evaluate a 2048 policy on a fixed seed suite."""

from __future__ import annotations

import json
from pathlib import Path
import statistics

from game2048 import play
from policy_sandbox import PolicyProcess


def load_suite(path: Path) -> tuple[list[int], int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        int(item["seed"] if isinstance(item, dict) else item)
        for item in data["seeds"]
    ], int(data.get("max_moves", 10000))


def evaluate(policy_path: Path, suite_path: Path) -> dict[str, object]:
    seeds, max_moves = load_suite(suite_path)
    with PolicyProcess(policy_path, "2048") as policy:
        def choose_move(board):
            value = policy.call([board])
            if type(value) is not str:
                raise TypeError("choose_move must return a string")
            return value

        games = [play(seed, choose_move, max_moves=max_moves) for seed in seeds]
    scores = [game.score for game in games]
    return {
        "mean_score": statistics.fmean(scores),
        "median_score": statistics.median(scores),
        "mean_max_tile": statistics.fmean(game.max_tile for game in games),
        "valid_fraction": statistics.fmean(game.error is None for game in games),
        "instances": [game.__dict__ for game in games],
    }
