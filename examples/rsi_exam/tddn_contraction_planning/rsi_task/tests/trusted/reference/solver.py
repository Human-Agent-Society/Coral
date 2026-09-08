"""Generic public-feature reference selected on author-only calibration cases.

This is not labeled SOTA and performs no hidden lookahead. It first prioritizes
strong shared bonds, then prefers smaller current endpoint products.
"""
from __future__ import annotations

from typing import Any


def select_edge(observation: dict[str, Any]) -> int:
    mask = observation["action_mask"]
    valid = [
        edge
        for edge in observation["edges"]
        if bool(mask.get(str(edge["edge_id"]), mask.get(edge["edge_id"], False)))
    ]
    if not valid:
        raise RuntimeError("No valid contraction edge is available.")

    def score(edge: dict[str, Any]) -> tuple[float, int, int, float, int]:
        left = int(edge["left_tdd_size"])
        right = int(edge["right_tdd_size"])
        return (
            -float(edge["normalized_bond_strength"]),
            left * right,
            left + right,
            float(edge["normalized_local_topology"]),
            int(edge["edge_id"]),
        )

    return int(min(valid, key=score)["edge_id"])
