"""Trusted Counting baseline used for runtime-relative scoring."""
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
    return int(min(
        valid,
        key=lambda edge: (
            -float(edge["normalized_bond_strength"]),
            float(edge["normalized_local_topology"]),
            int(edge["edge_id"]),
        ),
    )["edge_id"])
