#!/usr/bin/env python3
"""Evaluate semantic candidate routing on the visible cases."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np


DATA_ROOT = Path("/app/data/visible")
DEFAULT_PREDICTOR = Path("/app/methods/main/predict.py")
PUBLIC_FIELDS = (
    "query_points",
    "candidate_tracks",
    "occlusion_logits",
    "expected_dist_logits",
    "candidate_model_id",
    "candidate_stage",
)
LABEL_FIELDS = ("gt_tracks", "gt_occluded")
DISTANCE_THRESHOLDS = (1.0, 2.0, 4.0, 8.0, 16.0)
NUM_CANDIDATES = 10
MAX_SWITCHES = 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_data_path(relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("index path must be a nonempty string")
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"unsafe index path: {relative!r}")
    path = (DATA_ROOT / raw).resolve()
    if not path.is_relative_to(DATA_ROOT.resolve()):
        raise ValueError(f"index path escapes visible data: {relative!r}")
    return path


def _load_index() -> list[dict[str, Any]]:
    path = DATA_ROOT / "index.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("visible index has no cases")
    if payload.get("case_count") != len(cases):
        raise ValueError("visible index case_count mismatch")
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for row in cases:
        if not isinstance(row, dict):
            raise ValueError("visible index case row is not an object")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
            raise ValueError("visible index has an invalid or duplicate case_id")
        seen_ids.add(case_id)
        for path_key, hash_key in (
            ("input_file", "input_sha256"),
            ("label_file", "label_sha256"),
        ):
            data_path = _safe_data_path(row.get(path_key))
            relative = str(data_path.relative_to(DATA_ROOT.resolve()))
            if relative in seen_paths:
                raise ValueError("visible index reuses a case file")
            seen_paths.add(relative)
            expected_hash = row.get(hash_key)
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise ValueError(f"invalid {hash_key} for {case_id}")
            if not data_path.is_file() or _sha256(data_path) != expected_hash:
                raise ValueError(f"visible case failed integrity check: {relative}")
    return cases


def _load_npz(path: Path, required: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(required):
            raise ValueError(
                f"{path.name}: expected fields {sorted(required)}, got {sorted(archive.files)}"
            )
        result = {name: np.asarray(archive[name]) for name in required}
    return result


def _load_case(row: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    public = _load_npz(_safe_data_path(row["input_file"]), PUBLIC_FIELDS)
    labels = _load_npz(_safe_data_path(row["label_file"]), LABEL_FIELDS)
    tracks = public["candidate_tracks"]
    if tracks.ndim != 4 or tracks.shape[-1] != 2:
        raise ValueError("candidate_tracks must have shape [Q,T,K,2]")
    query_count, frame_count, candidate_count, _ = tracks.shape
    if query_count < 1 or frame_count < 1 or candidate_count != NUM_CANDIDATES:
        raise ValueError("visible case requires Q>0, T>0, and K=10")
    expected_shapes = {
        "query_points": (query_count, 3),
        "occlusion_logits": (query_count, frame_count, candidate_count),
        "expected_dist_logits": (query_count, frame_count, candidate_count),
        "candidate_model_id": (query_count, candidate_count),
        "candidate_stage": (query_count, candidate_count),
    }
    for name, shape in expected_shapes.items():
        if public[name].shape != shape:
            raise ValueError(f"{name} shape mismatch")
    expected_dtypes = {
        "query_points": np.dtype(np.float32),
        "candidate_tracks": np.dtype(np.float16),
        "occlusion_logits": np.dtype(np.float16),
        "expected_dist_logits": np.dtype(np.float16),
        "candidate_model_id": np.dtype(np.uint8),
        "candidate_stage": np.dtype(np.uint8),
    }
    for name, dtype in expected_dtypes.items():
        if public[name].dtype != dtype:
            raise ValueError(f"{name} dtype mismatch")
    if labels["gt_tracks"].shape != (query_count, frame_count, 2):
        raise ValueError("gt_tracks shape mismatch")
    if labels["gt_tracks"].dtype != np.float32:
        raise ValueError("gt_tracks dtype mismatch")
    if labels["gt_occluded"].shape != (query_count, frame_count):
        raise ValueError("gt_occluded shape mismatch")
    if labels["gt_occluded"].dtype != np.bool_:
        raise ValueError("gt_occluded dtype mismatch")
    for name in PUBLIC_FIELDS[:4]:
        if not np.isfinite(public[name]).all():
            raise ValueError(f"{name} contains NaN or infinity")
    if not np.isfinite(labels["gt_tracks"]).all():
        raise ValueError("gt_tracks contains NaN or infinity")
    query_frames = np.rint(public["query_points"][:, 0]).astype(np.int64)
    if np.any(query_frames < 0) or np.any(query_frames >= frame_count):
        raise ValueError("query frame is outside the video")
    tokens = public["candidate_model_id"].astype(np.int16) * 5 + public[
        "candidate_stage"
    ].astype(np.int16)
    if not np.array_equal(
        np.sort(tokens, axis=1),
        np.broadcast_to(np.arange(NUM_CANDIDATES), tokens.shape),
    ):
        raise ValueError("public model/stage pairs are not complete per query")
    for array in public.values():
        array.flags.writeable = False
    return public, labels


def _load_predict(path: Path) -> Callable[..., Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing submission predictor: {path}")
    module_dir = str(path.resolve().parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("candidate_routing_submission", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import predictor: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not isinstance(module, ModuleType) or not callable(getattr(module, "predict", None)):
        raise TypeError("submission must expose callable predict(...)")
    return module.predict


def _invoke(
    predict: Callable[..., Any], public: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = predict(*(public[name] for name in PUBLIC_FIELDS))
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise TypeError("predict must return exactly (state_token, occluded)")
    states = np.asarray(result[0])
    occluded = np.asarray(result[1])
    query_count, frame_count = public["candidate_tracks"].shape[:2]
    if states.shape != (query_count, frame_count):
        raise ValueError("predicted state_token must have shape [Q,T]")
    if states.dtype == np.bool_ or not np.issubdtype(states.dtype, np.integer):
        raise TypeError("predicted state_token must have a non-boolean integer dtype")
    if np.any(states < 0) or np.any(states >= NUM_CANDIDATES):
        raise ValueError("predicted state_token values must lie in 0..9")
    if occluded.shape != (query_count, frame_count) or occluded.dtype != np.bool_:
        raise TypeError("predicted occluded must be a boolean [Q,T] array")
    states = np.asarray(states, dtype=np.uint8)
    query_frames = np.rint(public["query_points"][:, 0]).astype(np.int64)
    for query, query_frame in enumerate(query_frames):
        scored = states[query, int(query_frame) + 1 :]
        changes = int(np.sum(scored[1:] != scored[:-1])) if scored.size > 1 else 0
        if changes > MAX_SWITCHES:
            raise ValueError(
                f"query {query} uses {changes} post-query switches; maximum is {MAX_SWITCHES}"
            )

    semantic = public["candidate_model_id"].astype(np.int16) * 5 + public[
        "candidate_stage"
    ].astype(np.int16)
    match = semantic[:, None, :] == states[:, :, None]
    if not np.all(np.sum(match, axis=2) == 1):
        raise ValueError("a predicted semantic state does not match exactly one candidate")
    candidate_index = np.argmax(match, axis=2)
    q_index = np.arange(query_count)[:, None]
    t_index = np.arange(frame_count)[None, :]
    tracks = np.asarray(public["candidate_tracks"], dtype=np.float32)[
        q_index, t_index, candidate_index
    ]
    return states, np.asarray(occluded, dtype=bool), np.asarray(tracks, np.float32)


def _average_jaccard(
    query_points: np.ndarray,
    gt_tracks: np.ndarray,
    gt_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    pred_occluded: np.ndarray,
) -> float:
    frame_count = gt_tracks.shape[1]
    eye = np.eye(frame_count, dtype=np.int32)
    first_query_mode = np.cumsum(eye, axis=1) - eye
    query_frame = np.rint(query_points[:, 0]).astype(np.int32)
    evaluation = first_query_mode[query_frame] > 0
    visible = ~gt_occluded
    pred_visible = ~pred_occluded
    squared_distance = np.sum(np.square(pred_tracks - gt_tracks), axis=-1)
    jaccards: list[float] = []
    for threshold in DISTANCE_THRESHOLDS:
        within = squared_distance < threshold**2
        true_positive = np.sum(within & visible & pred_visible & evaluation)
        gt_positive = np.sum(visible & evaluation)
        false_positive = ((~visible) & pred_visible) | ((~within) & pred_visible)
        false_positive_count = np.sum(false_positive & evaluation)
        jaccards.append(
            float(true_positive) / max(int(gt_positive + false_positive_count), 1)
        )
    return float(np.mean(jaccards))


def _assert_same(
    expected: tuple[np.ndarray, np.ndarray, np.ndarray],
    actual: tuple[np.ndarray, np.ndarray, np.ndarray],
    label: str,
) -> None:
    if not np.array_equal(expected[0], actual[0]):
        raise AssertionError(f"{label}: semantic state outputs changed")
    if not np.array_equal(expected[1], actual[1]):
        raise AssertionError(f"{label}: occlusion outputs changed")
    if not np.allclose(expected[2], actual[2], rtol=0.0, atol=1e-5):
        raise AssertionError(f"{label}: reconstructed tracks changed")


def _audit_contract(
    predict: Callable[..., Any],
    public: dict[str, np.ndarray],
    original: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    _assert_same(original, _invoke(predict, public), "deterministic repeat")
    query_count, frame_count, candidate_count = public["candidate_tracks"].shape[:3]
    query_order = np.arange(query_count - 1, -1, -1)
    query_public = {name: values[query_order] for name, values in public.items()}
    query_prediction = _invoke(predict, query_public)
    _assert_same(
        (original[0][query_order], original[1][query_order], original[2][query_order]),
        query_prediction,
        "query permutation",
    )
    singleton_public = {name: values[:1] for name, values in public.items()}
    _assert_same(
        (original[0][:1], original[1][:1], original[2][:1]),
        _invoke(predict, singleton_public),
        "singleton consistency",
    )

    rng = np.random.default_rng(20260708)
    order = np.stack([rng.permutation(candidate_count) for _ in range(query_count)])
    gather_3d = np.broadcast_to(order[:, None, :], (query_count, frame_count, candidate_count))
    gather_4d = np.broadcast_to(
        order[:, None, :, None], (query_count, frame_count, candidate_count, 2)
    )
    candidate_public = {
        "query_points": public["query_points"],
        "candidate_tracks": np.take_along_axis(public["candidate_tracks"], gather_4d, axis=2),
        "occlusion_logits": np.take_along_axis(public["occlusion_logits"], gather_3d, axis=2),
        "expected_dist_logits": np.take_along_axis(public["expected_dist_logits"], gather_3d, axis=2),
        "candidate_model_id": np.take_along_axis(public["candidate_model_id"], order, axis=1),
        "candidate_stage": np.take_along_axis(public["candidate_stage"], order, axis=1),
    }
    _assert_same(original, _invoke(predict, candidate_public), "candidate permutation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--audit-contract", action="store_true")
    parser.add_argument("--predictor", type=Path, default=DEFAULT_PREDICTOR)
    args = parser.parse_args()
    if args.case_limit is not None and args.case_limit < 1:
        parser.error("--case-limit must be positive")

    rows = _load_index()
    complete_count = len(rows)
    if args.case_limit is not None:
        rows = rows[: min(args.case_limit, complete_count)]
    predict = _load_predict(args.predictor)
    started = time.monotonic()
    scores: list[float] = []
    for index, row in enumerate(rows):
        public, labels = _load_case(row)
        prediction = _invoke(predict, public)
        if args.audit_contract and index == 0:
            _audit_contract(predict, public, prediction)
        score = _average_jaccard(
            public["query_points"],
            labels["gt_tracks"],
            labels["gt_occluded"],
            prediction[2],
            prediction[1],
        )
        scores.append(score)
        print(f"case {index + 1:02d}/{len(rows):02d}  AJ={score:.6f}", flush=True)

    mean_score = float(np.mean(scores))
    scope = "COMPLETE visible split" if len(rows) == complete_count else "NONCOMPARABLE subset"
    elapsed = time.monotonic() - started
    print(
        f"[selfcheck: {scope}] mean_video_average_jaccard_under_four_switch_budget={mean_score:.12f} "
        f"cases={len(rows)} elapsed_sec={elapsed:.3f}",
        flush=True,
    )
    if args.audit_contract:
        print("[selfcheck] deterministic/query/candidate/singleton contract audit passed")

    # Report optional run-budget checkpoints after the visible score.
    import os as _os, subprocess as _sp, sys as _sys
    if _os.path.exists("/app/budget.py"):
        _sys.stdout.flush()
        _sp.run([_sys.executable, "/app/budget.py"], check=False)


if __name__ == "__main__":
    main()
