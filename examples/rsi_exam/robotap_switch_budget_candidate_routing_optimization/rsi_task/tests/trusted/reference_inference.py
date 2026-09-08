"""GT-free inference for the frozen RoboTAP visible-reference artifact."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np


MODEL_SCHEMA = "robotap_visible_public_stage_reference_v1"
FEATURE_SCHEMA = "robotap_node_public_stage_features_v1"
NUM_STATES = 10
BOOT_FINAL_STATE = 9
IMAGE_SIZE = 256.0
PUBLIC_FIELDS = (
    "query_points",
    "candidate_tracks",
    "occlusion_logits",
    "expected_dist_logits",
    "candidate_model_id",
    "candidate_stage",
)


def load_model(path: Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    if artifact.get("schema_version") != MODEL_SCHEMA:
        raise ValueError("trusted reference model schema mismatch")
    if artifact.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("trusted reference feature schema mismatch")
    return artifact


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def evaluation_mask(query_points: np.ndarray, num_frames: int) -> np.ndarray:
    query_frame = np.rint(query_points[:, 0]).astype(np.int64)
    if np.any(query_frame < 0) or np.any(query_frame >= num_frames):
        raise ValueError("query frame lies outside video")
    return np.arange(num_frames)[None, :] > query_frame[:, None]


def take_candidate_axis(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    if values.ndim == 3:
        index = np.broadcast_to(order[:, None, :], values.shape)
    elif values.ndim == 4:
        index = np.broadcast_to(order[:, None, :, None], values.shape)
    else:
        raise ValueError("candidate-axis reorder expects rank three or four")
    return np.take_along_axis(values, index, axis=2)


def canonical_public(public: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(public) != set(PUBLIC_FIELDS):
        raise ValueError("trusted reference received a non-public field")
    model = np.asarray(public["candidate_model_id"], dtype=np.int16)
    stage = np.asarray(public["candidate_stage"], dtype=np.int16)
    token = model * 5 + stage
    order = np.argsort(token, axis=1, kind="stable")
    sorted_token = np.take_along_axis(token, order, axis=1)
    if not np.array_equal(sorted_token, np.broadcast_to(np.arange(10), sorted_token.shape)):
        raise ValueError("public model/stage metadata is not a ten-state permutation")
    return {
        "query_points": np.asarray(public["query_points"], dtype=np.float32),
        "candidate_tracks": take_candidate_axis(
            np.asarray(public["candidate_tracks"], dtype=np.float32), order
        ),
        "occlusion_logits": take_candidate_axis(
            np.asarray(public["occlusion_logits"], dtype=np.float32), order
        ),
        "expected_dist_logits": take_candidate_axis(
            np.asarray(public["expected_dist_logits"], dtype=np.float32), order
        ),
        "candidate_model_id": np.take_along_axis(model, order, axis=1).astype(np.uint8),
        "candidate_stage": np.take_along_axis(stage, order, axis=1).astype(np.uint8),
    }


def native_visibility(public: Mapping[str, np.ndarray]) -> np.ndarray:
    return (1.0 - stable_sigmoid(public["occlusion_logits"])) * (
        1.0 - stable_sigmoid(public["expected_dist_logits"])
    )


def build_features(data: Mapping[str, np.ndarray]) -> np.ndarray:
    tracks = np.asarray(data["candidate_tracks"], dtype=np.float32)
    occlusion_logits = np.asarray(data["occlusion_logits"], dtype=np.float32)
    expected_logits = np.asarray(data["expected_dist_logits"], dtype=np.float32)
    queries = np.asarray(data["query_points"], dtype=np.float32)
    q, t, k, _ = tracks.shape
    occ_probability = stable_sigmoid(occlusion_logits)
    expected_probability = stable_sigmoid(expected_logits)
    visibility = (1.0 - occ_probability) * (1.0 - expected_probability)
    frame_median = np.median(visibility, axis=2, keepdims=True)
    frame_std = np.std(visibility, axis=2, keepdims=True)
    current = visibility[..., :, None]
    other = visibility[..., None, :]
    visibility_rank = (
        np.sum(other < current, axis=-1) + 0.5 * np.sum(other == current, axis=-1)
    ) / float(k)
    consensus = np.median(tracks, axis=2, keepdims=True)
    consensus_delta = tracks - consensus
    consensus_distance = np.linalg.norm(consensus_delta, axis=-1)
    consensus_scale = np.median(consensus_distance, axis=2, keepdims=True)
    pair_distance = np.linalg.norm(
        tracks[..., :, None, :] - tracks[..., None, :, :], axis=-1
    )
    support = np.mean(np.exp(-np.square(pair_distance / 8.0)), axis=-1)
    step = np.zeros_like(tracks)
    step[:, 1:] = tracks[:, 1:] - tracks[:, :-1]
    speed = np.linalg.norm(step, axis=-1)
    acceleration = np.zeros_like(tracks)
    acceleration[:, 2:] = step[:, 2:] - step[:, 1:-1]
    acceleration_norm = np.linalg.norm(acceleration, axis=-1)
    query_xy = queries[:, [2, 1]]
    query_delta = tracks - query_xy[:, None, None, :]
    query_distance = np.linalg.norm(query_delta, axis=-1)
    post = evaluation_mask(queries, t).astype(np.float32)[..., None]
    count = np.maximum(np.sum(post, axis=1), 1.0)
    track_mean_visibility = np.sum(visibility * post, axis=1) / count
    centered = visibility - track_mean_visibility[:, None, :]
    track_std_visibility = np.sqrt(np.sum(np.square(centered) * post, axis=1) / count)
    track_mean_speed = np.sum(speed * post, axis=1) / count
    track_mean_consensus = np.sum(consensus_distance * post, axis=1) / count
    frame_fraction = np.arange(t, dtype=np.float32) / max(t - 1, 1)
    query_frame = np.rint(queries[:, 0]).astype(np.float32)
    time_since_query = (
        np.arange(t, dtype=np.float32)[None, :] - query_frame[:, None]
    ) / max(t - 1, 1)
    x = tracks[..., 0]
    y = tracks[..., 1]
    boundary = np.minimum.reduce((x, y, 255.0 - x, 255.0 - y))
    shape = (q, t, k)

    def broadcast(value: np.ndarray) -> np.ndarray:
        return np.broadcast_to(np.asarray(value, dtype=np.float32), shape)

    components = (
        np.clip(occlusion_logits, -12.0, 12.0),
        np.clip(expected_logits, -12.0, 12.0),
        occ_probability,
        expected_probability,
        visibility,
        visibility - frame_median,
        visibility_rank,
        broadcast(frame_median),
        broadcast(frame_std),
        x / IMAGE_SIZE,
        y / IMAGE_SIZE,
        broadcast(query_xy[:, None, None, 0] / IMAGE_SIZE),
        broadcast(query_xy[:, None, None, 1] / IMAGE_SIZE),
        broadcast(frame_fraction[None, :, None]),
        broadcast(time_since_query[:, :, None]),
        query_delta[..., 0] / IMAGE_SIZE,
        query_delta[..., 1] / IMAGE_SIZE,
        query_distance / IMAGE_SIZE,
        step[..., 0] / 32.0,
        step[..., 1] / 32.0,
        speed / 32.0,
        acceleration_norm / 32.0,
        consensus_delta[..., 0] / 32.0,
        consensus_delta[..., 1] / 32.0,
        consensus_distance / 32.0,
        broadcast(consensus_scale / 32.0),
        support,
        np.clip(boundary / 128.0, -2.0, 2.0),
        broadcast(track_mean_visibility[:, None, :]),
        broadcast(track_std_visibility[:, None, :]),
        broadcast(track_mean_speed[:, None, :] / 32.0),
        broadcast(track_mean_consensus[:, None, :] / 32.0),
    )
    result = np.stack([broadcast(value) for value in components], axis=-1)
    if result.shape[-1] != 32 or not np.isfinite(result).all():
        raise ValueError("invalid base feature tensor")
    return np.asarray(result, dtype=np.float32)


def public_stage_features(public: Mapping[str, np.ndarray]) -> np.ndarray:
    core = build_features(public)
    tracks = np.asarray(public["candidate_tracks"], dtype=np.float32)
    visibility = native_visibility(public)
    q, t, k, _ = tracks.shape
    states = np.arange(k, dtype=np.int64)
    models = states // 5
    stages = states % 5
    state = np.column_stack(
        [
            np.eye(k, dtype=np.float32),
            models.astype(np.float32),
            stages.astype(np.float32) / 4.0,
            (stages == 4).astype(np.float32),
            (models == 1).astype(np.float32),
            (states == BOOT_FINAL_STATE).astype(np.float32),
        ]
    )
    state = np.broadcast_to(state[None, None], (q, t, k, 15))

    def relation(index: np.ndarray | int) -> tuple[np.ndarray, ...]:
        if np.isscalar(index):
            reference_tracks = tracks[:, :, int(index), None, :]
            reference_visibility = visibility[:, :, int(index), None]
        else:
            reference_tracks = tracks[:, :, np.asarray(index), :]
            reference_visibility = visibility[:, :, np.asarray(index)]
        delta = tracks - reference_tracks
        return (
            delta[..., 0] / 32.0,
            delta[..., 1] / 32.0,
            np.linalg.norm(delta, axis=-1) / 32.0,
            visibility - reference_visibility,
        )

    same_final = np.where(models == 0, 4, 9)
    other_same_stage = np.where(models == 0, states + 5, states - 5)
    relations = relation(BOOT_FINAL_STATE) + relation(same_final) + relation(other_same_stage)
    result = np.concatenate([core, state, np.stack(relations, axis=-1)], axis=-1)
    if result.shape[-1] != 59 or not np.isfinite(result).all():
        raise ValueError("invalid public-stage feature tensor")
    return np.asarray(result, dtype=np.float32)


def budget_path(
    scores: np.ndarray, queries: np.ndarray, budget: int = 4
) -> np.ndarray:
    q_count, frame_count, state_count = scores.shape
    chosen = np.argmax(scores, axis=2).astype(np.int16)
    negative = -1e100
    state_ids = np.arange(state_count, dtype=np.int16)
    for query in range(q_count):
        start = int(np.rint(queries[query, 0])) + 1
        if start >= frame_count:
            continue
        length = frame_count - start
        back_state = np.full(
            (length, budget + 1, state_count), -1, dtype=np.int16
        )
        back_budget = np.full_like(back_state, -1)
        dynamic = np.full((budget + 1, state_count), negative, np.float64)
        dynamic[0] = scores[query, start]
        back_state[0, 0] = state_ids
        back_budget[0, 0] = 0
        for offset, frame in enumerate(range(start + 1, frame_count), start=1):
            updated = np.full_like(dynamic, negative)
            for used in range(budget + 1):
                best_value = dynamic[used].copy()
                best_state = state_ids.copy()
                best_used = np.full(state_count, used, dtype=np.int16)
                if used > 0:
                    previous = dynamic[used - 1]
                    first = int(np.argmax(previous))
                    without_first = previous.copy()
                    without_first[first] = negative
                    second = int(np.argmax(without_first))
                    switch_state = np.full(state_count, first, dtype=np.int16)
                    switch_state[first] = second
                    switch_value = previous[switch_state]
                    use_switch = switch_value > best_value
                    best_value = np.where(use_switch, switch_value, best_value)
                    best_state = np.where(use_switch, switch_state, best_state)
                    best_used = np.where(use_switch, used - 1, best_used)
                reachable = best_value > negative / 2
                updated[used, reachable] = (
                    best_value[reachable] + scores[query, frame, reachable]
                )
                back_state[offset, used, reachable] = best_state[reachable]
                back_budget[offset, used, reachable] = best_used[reachable]
            dynamic = updated
        flat = int(np.argmax(dynamic))
        used, state = np.unravel_index(flat, dynamic.shape)
        for offset in range(length - 1, -1, -1):
            chosen[query, start + offset] = state
            state, used = (
                int(back_state[offset, used, state]),
                int(back_budget[offset, used, state]),
            )
    return chosen


def take_path(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    q = np.arange(values.shape[0])[:, None]
    t = np.arange(values.shape[1])[None, :]
    return values[q, t, indices]


def predict(artifact: Mapping[str, Any], public: Mapping[str, np.ndarray]):
    canonical = canonical_public(public)
    features = public_stage_features(canonical)
    flat = features.reshape(-1, features.shape[-1])
    shape = features.shape[:3]
    quality = np.clip(artifact["ranker"].predict(flat), 0, 1).reshape(shape)
    visibility_model = artifact["visibility_model"]
    quality = quality.astype(np.float16).astype(np.float32)
    positive = np.flatnonzero(np.asarray(visibility_model.classes_) == 1)
    if positive.size != 1:
        raise ValueError("trusted visibility estimator lacks its positive class")
    learned = visibility_model.predict_proba(flat)[:, int(positive[0])].reshape(shape)
    learned = learned.astype(np.float16).astype(np.float32)
    native = native_visibility(canonical).astype(np.float16).astype(np.float32)
    indices = budget_path(quality, canonical["query_points"], budget=4)
    pred_native = take_path(native, indices)
    pred_learned = take_path(learned, indices)
    mix = float(artifact["visibility_mix_learned"])
    probability = (1.0 - mix) * pred_native + mix * pred_learned
    return np.asarray(indices, dtype=np.uint8), np.asarray(
        probability <= float(artifact["visibility_threshold"]), dtype=bool
    )
