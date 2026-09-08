#!/usr/bin/env python3
"""Sealed verifier for four-switch semantic candidate routing."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np

from secure_session import (
    InfrastructureError,
    SessionError,
    preflight_staged_read,
    probe_release_support,
    run_case,
    stage_submission,
    tree_sha256,
)


SUBMISSION = Path("/app/methods/main/predict.py")
HELDOUT = Path("/tests/heldout")
TRUSTED_ROOT = Path("/tests/trusted")
TRUSTED_MODULE = TRUSTED_ROOT / "reference_inference.py"
TRUSTED_MODEL = TRUSTED_ROOT / "reference_model.joblib"
ANCHOR_FILE = Path("/tests/anchors.json")
REWARD_DIR = Path("/logs/verifier")
STAGE_PARENT = Path("/run/candidate-routing-grade")

SCHEMA_VERSION = "candidate_routing_bundle_v1"
INPUT_SCHEMA_NAME = "candidate-stage"
EXPECTED_PARTITION = "primary_sealed"
PUBLIC_FIELDS = (
    "query_points",
    "candidate_tracks",
    "occlusion_logits",
    "expected_dist_logits",
    "candidate_model_id",
    "candidate_stage",
)
LABEL_FIELDS = ("gt_tracks", "gt_occluded")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
CASE_ID_RE = re.compile(r"[0-9a-f]{20}")
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_QUERIES = 4096
MAX_FRAMES = 4096
MAX_QUERY_FRAMES = 2_000_000
NUM_CANDIDATES = 10
CONSISTENCY_ATOL = 1e-4
HELDOUT_INDEX_SHA256 = "9a57000667ac4fbb3be0732b8c07ec82f2e923baeddf6ceba44c378a9f912b11"
TRUSTED_REFERENCE_MODULE_SHA256 = "1dd6e2af00aa294bf3f6d98b0eb7b2badbe067ac0b855e4b8afdfc11588843fe"
TRUSTED_REFERENCE_MODEL_SHA256 = "7d95e4c7e4376417de3732401052c82527366f2b357b3fb9f9e791235752d93b"
CASE_TIMEOUT_SECONDS = 120.0
ANCHOR_MATCH_TOL = 0.000001
MAX_SWITCHES = 4
PUBLIC_SUMMARY_FIELDS = (
    "correctness",
    "reward",
    "mean_video_AJ",
    "case_count",
    "metric",
    "elapsed_seconds",
    "error_category",
    "error",
)


class GradeError(RuntimeError):
    pass


@dataclass(frozen=True)
class HiddenCase:
    case_id: str
    public: dict[str, np.ndarray]
    gt_tracks: np.ndarray
    gt_occluded: np.ndarray


@dataclass(frozen=True)
class Prediction:
    state_token: np.ndarray
    tracks: np.ndarray
    occluded: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()




def _safe_root(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise GradeError(f"{label} root is unavailable or unsafe")
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise GradeError(f"{label} root is not a real directory")


def _safe_regular(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or len(candidate.parts) != 2 or any(
        part in ("", ".", "..") for part in candidate.parts
    ):
        raise GradeError(f"unsafe manifest path: {relative!r}")
    parent = root / candidate.parts[0]
    path = root / candidate
    if parent.is_symlink() or not parent.is_dir() or path.is_symlink() or not path.is_file():
        raise GradeError(f"manifest artifact is missing or unsafe: {relative}")
    parent_info = parent.lstat()
    info = path.lstat()
    if not stat.S_ISDIR(parent_info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise GradeError(f"manifest artifact has an invalid type: {relative}")
    if info.st_nlink != 1 or not 1 <= info.st_size <= MAX_ARCHIVE_BYTES:
        raise GradeError(f"manifest artifact size/link count is unsafe: {relative}")
    return path


def _load_exact_npz(path: Path, fields: Sequence[str]) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if tuple(sorted(archive.files)) != tuple(sorted(fields)):
                raise GradeError(f"{path.name} does not have the exact field schema")
            return {name: np.asarray(archive[name]).copy() for name in fields}
    except GradeError:
        raise
    except Exception as exc:
        raise GradeError(f"could not safely load {path.name}") from exc


def _array_schema(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in arrays.items()
    }


def _validate_arrays(
    case_id: str,
    public: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
) -> None:
    queries = public["query_points"]
    tracks = public["candidate_tracks"]
    occlusion = public["occlusion_logits"]
    expected = public["expected_dist_logits"]
    model = public["candidate_model_id"]
    stage = public["candidate_stage"]
    gt_tracks = labels["gt_tracks"]
    gt_occluded = labels["gt_occluded"]
    if queries.dtype != np.float32 or queries.ndim != 2 or queries.shape[1] != 3:
        raise GradeError(f"{case_id}: query_points must be float32 [Q,3]")
    if tracks.dtype != np.float16 or tracks.ndim != 4 or tracks.shape[-1] != 2:
        raise GradeError(f"{case_id}: candidate_tracks must be float16 [Q,T,K,2]")
    q_count, frame_count, candidate_count, _ = tracks.shape
    if not (1 <= q_count <= MAX_QUERIES and 2 <= frame_count <= MAX_FRAMES):
        raise GradeError(f"{case_id}: query/frame count is outside limits")
    if q_count * frame_count > MAX_QUERY_FRAMES or candidate_count != NUM_CANDIDATES:
        raise GradeError(f"{case_id}: candidate tensor size is outside limits")
    if occlusion.dtype != np.float16 or expected.dtype != np.float16:
        raise GradeError(f"{case_id}: candidate logits must be float16")
    if occlusion.shape != tracks.shape[:3] or expected.shape != tracks.shape[:3]:
        raise GradeError(f"{case_id}: candidate logits do not match tracks")
    if model.dtype != np.uint8 or stage.dtype != np.uint8:
        raise GradeError(f"{case_id}: public candidate metadata must be uint8")
    if model.shape != (q_count, candidate_count) or stage.shape != model.shape:
        raise GradeError(f"{case_id}: public candidate metadata shape mismatch")
    if gt_tracks.dtype != np.float32 or gt_tracks.shape != (q_count, frame_count, 2):
        raise GradeError(f"{case_id}: gt_tracks must be float32 [Q,T,2]")
    if gt_occluded.dtype != np.bool_ or gt_occluded.shape != (q_count, frame_count):
        raise GradeError(f"{case_id}: gt_occluded must be bool [Q,T]")
    for name, value in public.items():
        if not np.isfinite(value).all():
            raise GradeError(f"{case_id}: {name} contains NaN or infinity")
    if not np.isfinite(gt_tracks).all():
        raise GradeError(f"{case_id}: gt_tracks contains NaN or infinity")
    frames = np.rint(queries[:, 0]).astype(np.int64)
    if np.any(frames < 0) or np.any(frames >= frame_count):
        raise GradeError(f"{case_id}: a query frame is outside the video")
    expected_pairs = {(m, s) for m in (0, 1) for s in range(5)}
    for query_index in range(q_count):
        pairs = {(int(m), int(s)) for m, s in zip(model[query_index], stage[query_index])}
        if pairs != expected_pairs:
            raise GradeError(f"{case_id}: query {query_index} lacks the ten public stages")


def load_hidden_bundle(root: Path, expected_sha256: str) -> tuple[list[HiddenCase], str, dict[str, Any]]:
    _safe_root(root, "held-out")
    manifest_path = root / "index.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise GradeError("held-out index.json is missing or unsafe")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != expected_sha256:
        raise GradeError(f"held-out manifest commitment mismatch: {manifest_sha}")
    manifest = json.loads(manifest_path.read_text())
    expected_partition = EXPECTED_PARTITION
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise GradeError("held-out manifest schema is invalid")
    if manifest.get("input_schema_name") != INPUT_SCHEMA_NAME:
        raise GradeError("held-out bundle has the wrong input schema")
    if manifest.get("partition") != expected_partition:
        raise GradeError("held-out partition role is invalid")
    if manifest.get("input_fields") != list(PUBLIC_FIELDS):
        raise GradeError("held-out public fields differ from the frozen interface")
    if manifest.get("label_fields") != list(LABEL_FIELDS):
        raise GradeError("held-out label fields differ from the frozen interface")
    rows = manifest.get("cases")
    if not isinstance(rows, list) or not rows or manifest.get("case_count") != len(rows):
        raise GradeError("held-out manifest case count is invalid")
    identifiers = [row.get("case_id") for row in rows if isinstance(row, dict)]
    if len(identifiers) != len(rows) or identifiers != sorted(identifiers):
        raise GradeError("held-out cases must be dictionaries sorted by opaque case ID")
    if len(set(identifiers)) != len(rows):
        raise GradeError("held-out case IDs are duplicated")

    allowed = {Path("index.json")}
    cases: list[HiddenCase] = []
    for row in rows:
        case_id = str(row["case_id"])
        if CASE_ID_RE.fullmatch(case_id) is None:
            raise GradeError("held-out case ID is not opaque")
        input_name = f"inputs/case_{case_id}.npz"
        label_name = f"labels/case_{case_id}.npz"
        if row.get("input_file") != input_name or row.get("label_file") != label_name:
            raise GradeError(f"{case_id}: manifest paths are not canonical")
        input_path = _safe_regular(root, input_name)
        label_path = _safe_regular(root, label_name)
        allowed.update((Path(input_name), Path(label_name)))
        if SHA256_RE.fullmatch(str(row.get("input_sha256", ""))) is None:
            raise GradeError(f"{case_id}: invalid input commitment")
        if SHA256_RE.fullmatch(str(row.get("label_sha256", ""))) is None:
            raise GradeError(f"{case_id}: invalid label commitment")
        if sha256_file(input_path) != row["input_sha256"]:
            raise GradeError(f"{case_id}: public input commitment mismatch")
        if sha256_file(label_path) != row["label_sha256"]:
            raise GradeError(f"{case_id}: label commitment mismatch")
        public = _load_exact_npz(input_path, PUBLIC_FIELDS)
        labels = _load_exact_npz(label_path, LABEL_FIELDS)
        if row.get("input_schema") != _array_schema(public):
            raise GradeError(f"{case_id}: public array schema commitment mismatch")
        if row.get("label_schema") != _array_schema(labels):
            raise GradeError(f"{case_id}: label array schema commitment mismatch")
        _validate_arrays(case_id, public, labels)
        cases.append(
            HiddenCase(
                case_id=case_id,
                public={name: np.ascontiguousarray(value) for name, value in public.items()},
                gt_tracks=np.ascontiguousarray(labels["gt_tracks"]),
                gt_occluded=np.ascontiguousarray(labels["gt_occluded"]),
            )
        )

    for path in root.rglob("*"):
        relative = path.relative_to(root)
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise GradeError(f"held-out tree contains symlink: {relative}")
        if path.is_dir():
            if relative not in {Path("inputs"), Path("labels")}:
                raise GradeError(f"held-out tree contains unexpected directory: {relative}")
        elif not stat.S_ISREG(info.st_mode) or relative not in allowed:
            raise GradeError(f"held-out tree contains unexpected artifact: {relative}")
    return cases, manifest_sha, manifest


def _load_trusted_reference() -> tuple[Any, Callable[[Any, Mapping[str, np.ndarray]], Any], dict[str, str]]:
    _safe_root(TRUSTED_ROOT, "trusted reference")
    for path in (TRUSTED_MODULE, TRUSTED_MODEL):
        if path.is_symlink() or not path.is_file() or path.lstat().st_nlink != 1:
            raise GradeError(f"trusted reference artifact is missing or unsafe: {path.name}")
    expected_module = TRUSTED_REFERENCE_MODULE_SHA256
    expected_model = TRUSTED_REFERENCE_MODEL_SHA256
    actual = {"module": sha256_file(TRUSTED_MODULE), "model": sha256_file(TRUSTED_MODEL)}
    if actual != {"module": expected_module, "model": expected_model}:
        raise GradeError("trusted reference commitment mismatch")
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("trusted_candidate_reference", TRUSTED_MODULE)
    if spec is None or spec.loader is None:
        raise GradeError("could not import trusted reference module")
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predict = getattr(module, "predict", None)
    if not callable(predict):
        raise GradeError("trusted reference module must define predict(model, public)")
    loader = getattr(module, "load_model", None)
    model = loader(TRUSTED_MODEL) if callable(loader) else joblib.load(TRUSTED_MODEL)
    return model, predict, actual


def stable_sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return np.where(
        value >= 0,
        1.0 / (1.0 + np.exp(-value)),
        np.exp(value) / (1.0 + np.exp(value)),
    )


def native_visibility(public: Mapping[str, np.ndarray]) -> np.ndarray:
    return (1.0 - stable_sigmoid(public["occlusion_logits"])) * (
        1.0 - stable_sigmoid(public["expected_dist_logits"])
    )


def take_per_frame(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    q_index = np.arange(values.shape[0])[:, None]
    t_index = np.arange(values.shape[1])[None, :]
    return values[q_index, t_index, index]


def canonical_order(public: Mapping[str, np.ndarray]) -> np.ndarray:
    token = (
        public["candidate_model_id"].astype(np.int16) * 5
        + public["candidate_stage"].astype(np.int16)
    )
    order = np.argsort(token, axis=1, kind="stable")
    ordered = np.take_along_axis(token, order, axis=1)
    if not np.array_equal(
        ordered, np.broadcast_to(np.arange(NUM_CANDIDATES), ordered.shape)
    ):
        raise GradeError("public candidate semantics are not states 0..9")
    return order


def take_candidate_axis(values: np.ndarray, order: np.ndarray) -> np.ndarray:
    if values.ndim == 3:
        gather = np.broadcast_to(order[:, None, :], values.shape)
    elif values.ndim == 4:
        gather = np.broadcast_to(order[:, None, :, None], values.shape)
    else:
        raise GradeError("candidate-axis tensor has unsupported rank")
    return np.take_along_axis(values, gather, axis=2)


def budget_path(
    scores: np.ndarray, query_points: np.ndarray, max_switches: int
) -> np.ndarray:
    """Maximize additive node utility with an exact per-query switch cap."""

    query_count, frame_count, state_count = scores.shape
    chosen = np.argmax(scores, axis=2).astype(np.int16)
    negative = -1e100
    state_ids = np.arange(state_count, dtype=np.int16)
    for query in range(query_count):
        start = int(np.rint(query_points[query, 0])) + 1
        if start >= frame_count:
            continue
        length = frame_count - start
        back_state = np.full(
            (length, max_switches + 1, state_count), -1, dtype=np.int16
        )
        back_budget = np.full_like(back_state, -1)
        dynamic = np.full((max_switches + 1, state_count), negative, np.float64)
        dynamic[0] = scores[query, start]
        back_state[0, 0] = state_ids
        back_budget[0, 0] = 0
        for offset, frame in enumerate(range(start + 1, frame_count), start=1):
            updated = np.full_like(dynamic, negative)
            for used in range(max_switches + 1):
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


def reconstruct_tracks(
    public: Mapping[str, np.ndarray], states: np.ndarray
) -> np.ndarray:
    semantic = (
        public["candidate_model_id"].astype(np.int16) * 5
        + public["candidate_stage"].astype(np.int16)
    )
    match = semantic[:, None, :] == states[:, :, None]
    if not np.all(np.sum(match, axis=2) == 1):
        raise GradeError("semantic state does not identify exactly one candidate")
    index = np.argmax(match, axis=2)
    tracks = np.asarray(public["candidate_tracks"], dtype=np.float32)
    return np.ascontiguousarray(take_per_frame(tracks, index), dtype=np.float32)


def weak_prediction(case: HiddenCase) -> Prediction:
    order = canonical_order(case.public)
    probability = take_candidate_axis(
        native_visibility(case.public), order
    ).astype(np.float16).astype(np.float32)
    states = budget_path(probability, case.public["query_points"], 0).astype(np.uint8)
    occluded = take_per_frame(probability, states.astype(np.int64)) <= 0.5
    return Prediction(
        states, reconstruct_tracks(case.public, states), np.asarray(occluded, dtype=bool)
    )


def constrained_diagnostic_prediction(case: HiddenCase) -> Prediction:
    order = canonical_order(case.public)
    tracks = take_candidate_axis(
        np.asarray(case.public["candidate_tracks"], np.float32), order
    )
    distance = np.linalg.norm(tracks - case.gt_tracks[:, :, None], axis=-1)
    thresholds = np.asarray((1.0, 2.0, 4.0, 8.0, 16.0), dtype=np.float32)
    utility = np.mean(distance[..., None] < thresholds, axis=-1).astype(np.float32)
    utility *= (~case.gt_occluded)[:, :, None]
    states = budget_path(
        utility, case.public["query_points"], MAX_SWITCHES
    ).astype(np.uint8)
    return Prediction(
        states, reconstruct_tracks(case.public, states), case.gt_occluded.copy()
    )


def validate_prediction(
    value: Any, public: Mapping[str, np.ndarray], label: str
) -> Prediction:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise GradeError(f"{label} must return (state_token, occluded)")
    q_count, frame_count = public["candidate_tracks"].shape[:2]
    states = np.asarray(value[0])
    occluded = np.asarray(value[1])
    if states.shape != (q_count, frame_count):
        raise GradeError(f"{label} state_token must have shape [Q,T]")
    if states.dtype == np.bool_ or not np.issubdtype(states.dtype, np.integer):
        raise GradeError(f"{label} state_token must have a non-boolean integer dtype")
    if np.any(states < 0) or np.any(states >= NUM_CANDIDATES):
        raise GradeError(f"{label} state_token values must lie in 0..9")
    if occluded.dtype != np.bool_ or occluded.shape != (q_count, frame_count):
        raise GradeError(f"{label} occluded must be bool [Q,T]")
    states = np.ascontiguousarray(states, dtype=np.uint8)
    query_frames = np.rint(public["query_points"][:, 0]).astype(np.int64)
    for query, query_frame in enumerate(query_frames):
        scored = states[query, int(query_frame) + 1 :]
        changes = int(np.sum(scored[1:] != scored[:-1])) if scored.size > 1 else 0
        if changes > MAX_SWITCHES:
            raise GradeError(
                f"{label} query {query} exceeds the {MAX_SWITCHES}-switch budget"
            )
    return Prediction(
        states,
        reconstruct_tracks(public, states),
        np.ascontiguousarray(occluded, dtype=bool),
    )


def average_jaccard(case: HiddenCase, prediction: Prediction) -> float:
    frame_count = case.gt_tracks.shape[1]
    eye = np.eye(frame_count, dtype=np.int8)
    evaluation = (np.cumsum(eye, axis=1) - eye)[
        np.rint(case.public["query_points"][:, 0]).astype(np.int64)
    ] > 0
    visible = ~case.gt_occluded
    pred_visible = ~prediction.occluded
    squared_distance = np.sum((prediction.tracks - case.gt_tracks) ** 2, axis=-1)
    scores = []
    for threshold in (1, 2, 4, 8, 16):
        within = squared_distance < threshold**2
        true_positive = np.sum(within & visible & pred_visible & evaluation)
        gt_positive = np.sum(visible & evaluation)
        false_positive = np.sum(
            ((((~visible) & pred_visible) | ((~within) & pred_visible)) & evaluation)
        )
        scores.append(true_positive / max(gt_positive + false_positive, 1))
    return float(np.mean(scores))


def score_calibration_diagnostics(
    cases: Sequence[HiddenCase], model: Any, predict: Callable[[Any, Mapping[str, np.ndarray]], Any]
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rows = []
    for case in cases:
        public_copy = {name: value.copy() for name, value in case.public.items()}
        reference = validate_prediction(
            predict(model, public_copy), public_copy, f"human reference {case.case_id}"
        )
        baseline = weak_prediction(case)
        constrained = constrained_diagnostic_prediction(case)
        rows.append(
            {
                "case_id": case.case_id,
                "baseline_AJ": average_jaccard(case, baseline),
                "reference_AJ": average_jaccard(case, reference),
                "constrained_dp_AJ": average_jaccard(case, constrained),
            }
        )
    diagnostics = {
        "baseline": float(np.mean([row["baseline_AJ"] for row in rows])),
        "reference": float(np.mean([row["reference_AJ"] for row in rows])),
        "constrained_dp": float(np.mean([row["constrained_dp_AJ"] for row in rows])),
    }
    if not 0.0 <= diagnostics["baseline"] < diagnostics["reference"] < diagnostics["constrained_dp"] <= 1.0:
        raise GradeError(f"calibration diagnostics are not strictly ordered: {diagnostics}")
    return diagnostics, rows


def read_configured_anchors() -> dict[str, float]:
    if ANCHOR_FILE.is_symlink() or not ANCHOR_FILE.is_file():
        raise GradeError("sealed anchor file is missing or unsafe")
    info = ANCHOR_FILE.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid != 0
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise GradeError("sealed anchor file must be root-owned mode 0400")
    try:
        payload = json.loads(ANCHOR_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GradeError("sealed anchor file is unreadable or invalid") from exc
    expected = {"BASELINE", "UPPER_BOUND"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise GradeError("sealed anchor file has unexpected keys")
    values: dict[str, float] = {}
    for name in sorted(expected):
        value = payload[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GradeError(f"{name} must be numeric")
        values[name.lower()] = float(value)
    if not all(math.isfinite(value) for value in values.values()):
        raise GradeError("configured anchors must be finite")
    if not 0.0 <= values["baseline"] < values["upper_bound"] == 1.0:
        raise GradeError("configured anchors must satisfy 0 <= baseline < upper = 1")
    return values


def map_reward(metric: float, weak: float, upper: float) -> float:
    """baseline->0, upper->1, linear in between."""
    if not math.isfinite(metric) or metric <= weak:
        return 0.0
    if metric >= upper:
        return 1.0
    return (metric - weak) / (upper - weak)


def _nonidentity_permutation(rng: np.random.Generator, count: int) -> np.ndarray:
    order = rng.permutation(count)
    if count > 1 and np.array_equal(order, np.arange(count)):
        order = np.roll(order, 1)
    return order


def permute_public(
    public: Mapping[str, np.ndarray], seed: int, *, permute_queries: bool = True
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    q_count, _, candidate_count, _ = public["candidate_tracks"].shape
    query_order = _nonidentity_permutation(rng, q_count) if permute_queries else np.arange(q_count)
    result = {name: np.asarray(value)[query_order].copy() for name, value in public.items()}
    for query in range(q_count):
        order = _nonidentity_permutation(rng, candidate_count)
        result["candidate_tracks"][query] = np.take(
            result["candidate_tracks"][query], order, axis=1
        )
        result["occlusion_logits"][query] = np.take(
            result["occlusion_logits"][query], order, axis=1
        )
        result["expected_dist_logits"][query] = np.take(
            result["expected_dist_logits"][query], order, axis=1
        )
        result["candidate_model_id"][query] = np.take(
            result["candidate_model_id"][query], order, axis=0
        )
        result["candidate_stage"][query] = np.take(
            result["candidate_stage"][query], order, axis=0
        )
    return result, query_order


def _case_seed(manifest_sha: str, case_id: str, purpose: str) -> int:
    digest = hashlib.sha256(f"{manifest_sha}:{case_id}:{purpose}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _run_submission(
    cases: Sequence[HiddenCase], manifest_sha: str, staged: Path, timeout: float
) -> tuple[float, list[dict[str, Any]]]:
    rows = []
    for case_index, case in enumerate(cases):
        q_count, _frame_count = case.gt_occluded.shape
        batch_public, query_order = permute_public(
            case.public, _case_seed(manifest_sha, case.case_id, "batch")
        )
        batch_result = run_case(
            staged, batch_public, stage_parent=STAGE_PARENT,
            timeout_seconds=timeout, release_mode=True,
        )
        batch = validate_prediction(
            (batch_result.state_token, batch_result.occluded), batch_public,
            f"submission batch {case.case_id}",
        )

        repeat_elapsed: float | None = None
        if case_index == 0:
            repeat_result = run_case(
                staged, batch_public, stage_parent=STAGE_PARENT,
                timeout_seconds=timeout, release_mode=True,
            )
            repeat = validate_prediction(
                (repeat_result.state_token, repeat_result.occluded), batch_public,
                f"submission deterministic repeat {case.case_id}",
            )
            if not np.array_equal(
                batch.state_token, repeat.state_token
            ) or not np.array_equal(batch.occluded, repeat.occluded):
                raise GradeError(f"{case.case_id}: deterministic repeat check failed")
            repeat_elapsed = repeat_result.elapsed_seconds

        selected_original = _case_seed(manifest_sha, case.case_id, "singleton-index") % q_count
        position = int(np.flatnonzero(query_order == selected_original)[0])
        singleton_source = {
            name: np.asarray(value)[selected_original : selected_original + 1].copy()
            for name, value in case.public.items()
        }
        singleton_public, _ = permute_public(
            singleton_source,
            _case_seed(manifest_sha, case.case_id, "singleton-candidates"),
            permute_queries=False,
        )
        singleton_result = run_case(
            staged, singleton_public, stage_parent=STAGE_PARENT,
            timeout_seconds=timeout, release_mode=True,
        )
        singleton = validate_prediction(
            (singleton_result.state_token, singleton_result.occluded), singleton_public,
            f"submission singleton {case.case_id}",
        )
        if not np.array_equal(
            batch.state_token[position], singleton.state_token[0]
        ) or not np.array_equal(batch.occluded[position], singleton.occluded[0]):
            raise GradeError(
                f"{case.case_id}: query/candidate permutation or singleton consistency failed"
            )

        original_state = np.empty_like(batch.state_token)
        original_tracks = np.empty_like(batch.tracks)
        original_occluded = np.empty_like(batch.occluded)
        original_state[query_order] = batch.state_token
        original_tracks[query_order] = batch.tracks
        original_occluded[query_order] = batch.occluded
        prediction = Prediction(original_state, original_tracks, original_occluded)
        value = average_jaccard(case, prediction)
        rows.append(
            {
                "case_id": case.case_id,
                "average_jaccard": value,
                "batch_elapsed_seconds": batch_result.elapsed_seconds,
                "singleton_elapsed_seconds": singleton_result.elapsed_seconds,
                "consistency_query_index": int(selected_original),
                "repeat_elapsed_seconds": repeat_elapsed,
            }
        )
    return float(np.mean([row["average_jaccard"] for row in rows])), rows


def _secure_reward_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise GradeError("verifier reward path is unsafe")
    # Harbor creates the host-side verifier-log directory as the invoking host
    # user before bind-mounting it into this root-run verifier. Adopt that
    # already type-checked mount while grading, then retain the owner check.
    if os.geteuid() == 0 and info.st_uid != 0:
        os.chown(path, 0, 0, follow_symlinks=False)
        info = path.lstat()
    if info.st_uid != os.geteuid():
        raise GradeError("verifier reward directory has the wrong owner")
    os.chmod(path, 0o700)


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _public_summary(outcome: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: outcome[name] for name in PUBLIC_SUMMARY_FIELDS if name in outcome
    }


def _write_results(outcome: Mapping[str, Any]) -> None:
    _secure_reward_dir(REWARD_DIR)
    try:
        reward = float(outcome.get("reward", 0.0))
    except (TypeError, ValueError):
        reward = 0.0
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        reward = 0.0
    public_summary = _public_summary({**outcome, "reward": reward})
    score_details = {
        "correctness": bool(outcome.get("correctness", False)),
        "metric": outcome.get("metric", "mean_video_average_jaccard_under_four_switch_budget"),
        "raw_metric": outcome.get("mean_video_AJ"),
        "reward": reward,
    }
    if "error" in public_summary:
        score_details["error"] = public_summary["error"]
    _atomic_write(
        REWARD_DIR / "reward.json",
        (json.dumps({"reward": reward}, allow_nan=False, sort_keys=True) + "\n").encode(),
    )
    _atomic_write(REWARD_DIR / "reward.txt", f"{reward:.9f}\n".encode())
    _atomic_write(
        REWARD_DIR / "grade_debug.json",
        (json.dumps(public_summary, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(),
    )
    _atomic_write(
        REWARD_DIR / "score_details.json",
        (json.dumps(score_details, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(),
    )


def _destroy_secret_tree(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        shutil.rmtree(path)
    if path.exists() or path.is_symlink():
        raise GradeError(f"failed to destroy {label} tree before candidate launch")


def main() -> None:
    started = time.monotonic()
    outcome: dict[str, Any]
    submission_root: Path | None = None
    submission_fault = False
    try:
        timeout = CASE_TIMEOUT_SECONDS
        if not 1.0 <= timeout <= 300.0:
            raise GradeError("CASE_TIMEOUT_SEC is outside limits")
        # Probe the release namespace before reading or scoring any sealed data.
        probe_release_support()
        cases, manifest_sha, _ = load_hidden_bundle(
            HELDOUT, HELDOUT_INDEX_SHA256
        )
        reference_model, reference_predict, reference_hashes = _load_trusted_reference()
        diagnostics, _diagnostic_rows = score_calibration_diagnostics(
            cases, reference_model, reference_predict
        )
        configured = read_configured_anchors()
        if abs(configured["baseline"] - diagnostics["baseline"]) > ANCHOR_MATCH_TOL:
            raise GradeError("configured baseline differs from sealed diagnostics")


        # Candidate code cannot start until both labels and trusted reference artifacts
        # have been consumed into parent memory, scored, and removed from the filesystem.
        del reference_model, reference_predict
        gc.collect()
        _destroy_secret_tree(HELDOUT, "held-out")
        _destroy_secret_tree(TRUSTED_ROOT, "trusted reference")

        try:
            submission_root, staged, submission_sha = stage_submission(SUBMISSION, STAGE_PARENT)
            preflight_staged_read(staged)
            metric, _case_rows = _run_submission(cases, manifest_sha, staged, timeout)
            if tree_sha256(submission_root) != submission_sha:
                raise GradeError("staged submission changed during verification")
        except InfrastructureError:
            raise
        except (GradeError, SessionError):
            submission_fault = True
            raise
        reward = map_reward(metric, configured["baseline"], configured["upper_bound"])
        outcome = {
            "correctness": True,
            "reward": reward,
            "mean_video_AJ": metric,
            "calibration_diagnostics": diagnostics,
            "configured_anchors": configured,
            "theoretical_upper_bound": 1.0,
            "manifest_sha256": manifest_sha,
            "trusted_reference_hashes": reference_hashes,
            "submission_sha256": submission_sha,
            "case_count": len(cases),
            "metric": "mean_video_average_jaccard_under_four_switch_budget",
            "elapsed_seconds": time.monotonic() - started,
        }
    except Exception as exc:
        prefix = "submission failed:" if submission_fault else "grader failed:"
        outcome = {
            "correctness": False,
            "reward": 0.0,
            "mean_video_AJ": None,
            "elapsed_seconds": time.monotonic() - started,
            "error_category": type(exc).__name__,
            "error": f"{prefix} {type(exc).__name__}",
        }
    finally:
        if submission_root is not None:
            shutil.rmtree(submission_root, ignore_errors=True)
    _write_results(outcome)
    print(json.dumps(_public_summary(outcome), allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
