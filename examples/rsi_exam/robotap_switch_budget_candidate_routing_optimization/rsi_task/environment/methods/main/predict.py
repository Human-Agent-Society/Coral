"""Weak persistent router for the switch-budgeted candidate-routing task.

The baseline canonicalizes candidates by the public model/stage token, then
chooses the one state with the greatest summed native confidence over all
scored frames of each query.  It uses zero of the four allowed switches.
"""

from __future__ import annotations

import numpy as np


NUM_STATES = 10


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def _canonical_order(
    candidate_model_id: np.ndarray, candidate_stage: np.ndarray
) -> np.ndarray:
    token = candidate_model_id.astype(np.int16) * 5 + candidate_stage.astype(np.int16)
    order = np.argsort(token, axis=1, kind="stable")
    ordered = np.take_along_axis(token, order, axis=1)
    expected = np.broadcast_to(np.arange(NUM_STATES), ordered.shape)
    if not np.array_equal(ordered, expected):
        raise ValueError("each query must contain semantic states 0..9 exactly once")
    return order


def _take_candidate_axis(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    if values.ndim == 3:
        gather = np.broadcast_to(order[:, None, :], values.shape)
    elif values.ndim == 4:
        gather = np.broadcast_to(order[:, None, :, None], values.shape)
    else:
        raise ValueError("candidate-axis array must have rank three or four")
    return np.take_along_axis(values, gather, axis=2)


def _validate(
    query_points: np.ndarray,
    candidate_tracks: np.ndarray,
    occlusion_logits: np.ndarray,
    expected_dist_logits: np.ndarray,
    candidate_model_id: np.ndarray,
    candidate_stage: np.ndarray,
) -> tuple[int, int, np.ndarray]:
    if candidate_tracks.ndim != 4 or candidate_tracks.shape[-1] != 2:
        raise ValueError("candidate_tracks must have shape [Q,T,K,2]")
    query_count, frame_count, state_count, _ = candidate_tracks.shape
    if query_count < 1 or frame_count < 1 or state_count != NUM_STATES:
        raise ValueError("candidate_tracks requires Q>0, T>0, and K=10")
    if query_points.shape != (query_count, 3):
        raise ValueError("query_points must have shape [Q,3]")
    if occlusion_logits.shape != (query_count, frame_count, state_count):
        raise ValueError("occlusion_logits must have shape [Q,T,K]")
    if expected_dist_logits.shape != occlusion_logits.shape:
        raise ValueError("expected_dist_logits must have shape [Q,T,K]")
    if candidate_model_id.shape != (query_count, state_count):
        raise ValueError("candidate_model_id must have shape [Q,K]")
    if candidate_stage.shape != candidate_model_id.shape:
        raise ValueError("candidate_stage must have shape [Q,K]")
    for name, value in (
        ("query_points", query_points),
        ("candidate_tracks", candidate_tracks),
        ("occlusion_logits", occlusion_logits),
        ("expected_dist_logits", expected_dist_logits),
    ):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or infinity")
    query_frame = np.rint(query_points[:, 0]).astype(np.int64)
    if np.any(query_frame < 0) or np.any(query_frame >= frame_count):
        raise ValueError("query frame is outside the video")
    return query_count, frame_count, _canonical_order(
        candidate_model_id, candidate_stage
    )


def predict(
    query_points: np.ndarray,
    candidate_tracks: np.ndarray,
    occlusion_logits: np.ndarray,
    expected_dist_logits: np.ndarray,
    candidate_model_id: np.ndarray,
    candidate_stage: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a semantic state token and occlusion flag for every query/frame."""

    query_points = np.asarray(query_points)
    candidate_tracks = np.asarray(candidate_tracks)
    occlusion_logits = np.asarray(occlusion_logits)
    expected_dist_logits = np.asarray(expected_dist_logits)
    candidate_model_id = np.asarray(candidate_model_id)
    candidate_stage = np.asarray(candidate_stage)
    query_count, frame_count, order = _validate(
        query_points,
        candidate_tracks,
        occlusion_logits,
        expected_dist_logits,
        candidate_model_id,
        candidate_stage,
    )

    native = (1.0 - _sigmoid(occlusion_logits)) * (
        1.0 - _sigmoid(expected_dist_logits)
    )
    native = _take_candidate_axis(native, order).astype(np.float16).astype(np.float32)
    states = np.argmax(native, axis=2).astype(np.uint8)
    query_frame = np.rint(query_points[:, 0]).astype(np.int64)
    for query in range(query_count):
        start = int(query_frame[query]) + 1
        if start < frame_count:
            # Accumulate chronologically in float64 so near ties are deterministic.
            total = np.asarray(native[query, start], dtype=np.float64).copy()
            for frame in range(start + 1, frame_count):
                total += native[query, frame]
            states[query, start:] = np.uint8(np.argmax(total))

    q_index = np.arange(query_count)[:, None]
    t_index = np.arange(frame_count)[None, :]
    selected_visibility = native[q_index, t_index, states.astype(np.int64)]
    occluded = selected_visibility <= 0.5
    return np.ascontiguousarray(states), np.ascontiguousarray(occluded, dtype=bool)
