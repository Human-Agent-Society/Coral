"""Editable inference-time contraction planner."""
from __future__ import annotations

from typing import Any


def _valid_edges(observation: dict[str, Any]) -> list[dict[str, Any]]:
    mask = observation["action_mask"]
    return [
        edge
        for edge in observation["edges"]
        if bool(mask.get(str(edge["edge_id"]), mask.get(edge["edge_id"], False)))
    ]


def _score(edge: dict[str, Any]) -> tuple[float, float, int]:
    """Lower is better: stronger bonds, lower local topology, stable tie-break."""
    return (
        -float(edge.get("normalized_bond_strength", edge.get("bond_dimension", 0.0))),
        float(edge.get("normalized_local_topology", 0.0)),
        int(edge["edge_id"]),
    )


def select_edge(observation: dict[str, Any]) -> int:
    """Return one enabled stable edge_id."""
    valid = _valid_edges(observation)
    if not valid:
        raise RuntimeError("No valid contraction edge is available.")
    return int(min(valid, key=_score)["edge_id"])
