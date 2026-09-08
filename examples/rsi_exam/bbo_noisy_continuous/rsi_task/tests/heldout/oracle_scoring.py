"""Sealed, oracle-normalized scoring primitives for BBO leaderboard tasks.

This module is copied into each task's verifier-only assets.  The sealed
per-instance oracle defines quality one; the frozen frontier anchor defines
the 0.60 landing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


SCORING_NAME = "oracle_normalized_auc70_final30"
ANYTIME_WEIGHT = 0.70
FINAL_WEIGHT = 0.30
_TOL = 1e-10
FRONTIER_LANDING = 0.60


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def load_oracle_manifest(here: Path, anchors: dict[str, Any]) -> dict[str, Any]:
    """Load and validate the per-instance evaluator-only oracle manifest."""
    manifest = _load_json(here / "oracle_values.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("oracle_values.json schema_version must be 1")
    if manifest.get("objective_direction") != "min":
        raise ValueError("oracle_values.json must declare minimization")
    if manifest.get("oracle_kind") not in {"exact", "empirical"}:
        raise ValueError("oracle_values.json oracle_kind must be exact or empirical")
    if manifest.get("task_id") != anchors.get("task_id"):
        raise ValueError("oracle_values.json task_id does not match frozen anchors")

    values = np.asarray(manifest.get("per_instance_objective"), dtype=float)
    expected = int(anchors["n_hidden"])
    if values.shape != (expected,) or not np.isfinite(values).all():
        raise ValueError("oracle_values.json per_instance_objective has wrong shape or non-finite values")

    scoring = manifest.get("scoring")
    if not isinstance(scoring, dict):
        raise ValueError("oracle_values.json requires a scoring object")
    if not np.isclose(float(scoring.get("anytime_weight", np.nan)), ANYTIME_WEIGHT):
        raise ValueError("oracle anytime weight must be 0.70")
    if not np.isclose(float(scoring.get("final_weight", np.nan)), FINAL_WEIGHT):
        raise ValueError("oracle final weight must be 0.30")
    if manifest["oracle_kind"] == "empirical":
        tolerance = manifest.get("breach_tolerance")
        if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool) or tolerance < 0:
            raise ValueError("empirical oracle requires a non-negative breach_tolerance")
    return manifest


def _coerce_traces(decision: dict[str, Any], anchors: dict[str, Any]) -> np.ndarray:
    traces = np.asarray(decision.get("traces"), dtype=float)
    expected = (
        int(anchors["n_hidden"]),
        int(anchors["n_seeds"]),
        int(anchors["budget"]),
    )
    if traces.shape != expected:
        raise ValueError(f"bad trace shape {traces.shape}, expected {expected}")
    if not np.isfinite(traces).all():
        raise ValueError("traces must contain only finite values")
    if np.any(np.diff(traces, axis=2) > 1e-8):
        raise ValueError("best-so-far traces must be non-increasing")
    return traces


def _quality_trace(values: np.ndarray, floor: np.ndarray, oracle: np.ndarray) -> np.ndarray:
    """Return clipped per-instance quality; floor=0 and oracle=1."""
    denominator = floor - oracle[:, None]
    threshold = _TOL * (1.0 + np.maximum(np.abs(floor), np.abs(oracle[:, None])))
    if np.any(denominator <= threshold):
        bad = np.argwhere(denominator <= threshold)[0].tolist()
        raise ValueError(f"invalid oracle calibration: floor must be above oracle at instance/checkpoint {bad}")
    return np.clip((floor - values) / denominator, 0.0, 1.0)


def _reward_from_combined_quality(
    quality: np.ndarray,
    frontier_quality: np.ndarray,
) -> np.ndarray:
    """baseline->0, frontier->0.6, upper->1."""
    values = np.asarray(quality, dtype=float)
    frontier = np.asarray(frontier_quality, dtype=float)
    if values.shape != frontier.shape:
        raise ValueError("candidate/frontier combined-quality shape mismatch")
    if (
        not np.isfinite(values).all()
        or not np.isfinite(frontier).all()
        or np.any(frontier <= _TOL)
        or np.any(frontier >= 1.0 - _TOL)
    ):
        raise ValueError("frontier quality must be strictly between floor and oracle")
    lower = FRONTIER_LANDING * values / frontier
    upper = FRONTIER_LANDING + (1.0 - FRONTIER_LANDING) * (
        values - frontier
    ) / (1.0 - frontier)
    return np.clip(np.where(values <= frontier, lower, upper), 0.0, 1.0)


def score_minimization_traces(decision: dict[str, Any], anchors: dict[str, Any], here: Path) -> dict[str, Any]:
    """Score median seed traces with 70% pre-final AUC and 30% final quality."""
    traces = _coerce_traces(decision, anchors)
    budget = int(anchors["budget"])
    if budget < 2:
        raise ValueError("oracle leaderboard scoring requires at least two evaluations")
    floor = np.asarray(anchors["floor_trace_median"], dtype=float)
    frontier_combined = np.asarray(anchors["frontier_combined"], dtype=float)
    expected_shape = (int(anchors["n_hidden"]), budget)
    if floor.shape != expected_shape:
        raise ValueError("frozen floor traces do not match the task protocol")
    if frontier_combined.shape != (int(anchors["n_hidden"]),):
        raise ValueError("frozen frontier anchor must hold one combined quality per instance")
    if not np.isfinite(floor).all() or not np.isfinite(frontier_combined).all():
        raise ValueError("frozen floor traces and frontier anchor must be finite")
    if np.any(np.diff(floor, axis=1) > 1e-8):
        raise ValueError("frozen traces must be best-so-far")

    manifest = load_oracle_manifest(here, anchors)
    oracle = np.asarray(manifest["per_instance_objective"], dtype=float)
    medians = np.median(traces, axis=1)

    if manifest["oracle_kind"] == "empirical":
        tolerance = float(manifest["breach_tolerance"])
        breached = medians[:, -1] < oracle - tolerance
        if np.any(breached):
            return {
                "feasible": False,
                "score": 0.0,
                "score_anytime": 0.0,
                "score_final": 0.0,
                "metric": SCORING_NAME,
                "reason": "empirical oracle breach; recalibration required",
                "oracle_breach": True,
                "breached_instances": np.flatnonzero(breached).astype(int).tolist(),
            }

    quality = _quality_trace(medians, floor, oracle)
    final_per_instance = quality[:, -1]
    anytime_per_instance = np.mean(quality[:, :-1], axis=1)
    combined_per_instance = ANYTIME_WEIGHT * anytime_per_instance + FINAL_WEIGHT * final_per_instance

    floor_quality = _quality_trace(floor, floor, oracle)
    floor_combined = (
        ANYTIME_WEIGHT * np.mean(floor_quality[:, :-1], axis=1)
        + FINAL_WEIGHT * floor_quality[:, -1]
    )
    reward_per_instance = _reward_from_combined_quality(
        combined_per_instance,
        frontier_combined,
    )
    score_anytime = float(np.mean(anytime_per_instance))
    score_final = float(np.mean(final_per_instance))
    raw_score = float(np.clip(np.mean(combined_per_instance), 0.0, 1.0))
    score = float(np.mean(reward_per_instance))
    return {
        "feasible": True,
        "metric": SCORING_NAME,
        "score": score,
        "raw_score": raw_score,
        "score_anytime": score_anytime,
        "score_final": score_final,
        "kpi": float(np.mean(medians[:, -1])),
        "reward_per_inst": [float(value) for value in reward_per_instance],
        "raw_quality_per_inst": [float(value) for value in combined_per_instance],
        "anytime_per_inst": [float(value) for value in anytime_per_instance],
        "final_per_inst": [float(value) for value in final_per_instance],
        "oracle_kind": manifest["oracle_kind"],
        "frontier_score_diagnostic": float(
            np.mean(
                _reward_from_combined_quality(
                    frontier_combined,
                    frontier_combined,
                )
            )
        ),
        "frontier_raw_metric_diagnostic": float(np.mean(frontier_combined)),
        "floor_score_diagnostic": float(
            np.mean(_reward_from_combined_quality(floor_combined, frontier_combined))
        ),
        "reason": None,
    }
