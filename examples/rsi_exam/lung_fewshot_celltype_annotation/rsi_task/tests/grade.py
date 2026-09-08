#!/usr/bin/env python3
"""Replay one submitted annotator in a sealed child and score aggregate macro-F1."""

from __future__ import annotations

import io
import json
import math
import sys
import tempfile
from pathlib import Path

import anndata as ad
import pandas as pd
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fail_closed import LOG_DIR as REWARD_DIR, atomic_write, write_failure_outputs  # noqa: E402
from reward_mapping import reward_for  # noqa: E402
from secure_runtime import (  # noqa: E402
    INPUT_DIR,
    SEALED_DIR,
    GraderError,
    SubmissionError,
    _remove_tree,
    allocate_run_identity,
    assert_sealed_denied,
    capture_and_unlink_sealed,
    make_parent_nondumpable_subreaper,
    remove_experiment_log,
    run_submission,
    seal_external_write_surfaces,
    staged_submission,
    validate_child_inputs,
)

ANCHOR_PATH = SEALED_DIR / "config.json"
TRUTH_PATH = SEALED_DIR / "truth.csv"
ANCHOR_SHA256 = "962e08fa3b30b7adb8c49db7194c9261441c45266c06f12287079820a8381054"
TRUTH_SHA256 = "4af62b3a9302583dea5a562401dc1f0c3eb8eb6488d5d966fe7374e1d74a666b"
INPUT_PATHS = (
    INPUT_DIR / "labeled.h5ad",
    INPUT_DIR / "unlabeled.h5ad",
    INPUT_DIR / "query.h5ad",
    INPUT_DIR / "classes.txt",
    INPUT_DIR / "reference_model.pkl",
)


def _strict_object(payload: bytes, name: str) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise GraderError(f"grader failed: duplicate key in {name}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise GraderError(f"grader failed: malformed {name}") from exc
    if not isinstance(value, dict):
        raise GraderError(f"grader failed: {name} must contain one object")
    return value


def _load_anchors(payload: bytes) -> dict[str, float]:
    raw = _strict_object(payload, "sealed anchor configuration")
    names = ("BASELINE", "REFERENCE", "UPPER_BOUND")
    if set(raw) != set(names):
        raise GraderError("grader failed: sealed anchor configuration has the wrong keys")
    anchors: dict[str, float] = {}
    for name in names:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise GraderError("grader failed: sealed anchor is not numeric")
        anchors[name] = float(value)
        if not math.isfinite(anchors[name]):
            raise GraderError("grader failed: sealed anchor is not finite")
    if not anchors["BASELINE"] < anchors["REFERENCE"] < anchors["UPPER_BOUND"]:
        raise GraderError("grader failed: sealed anchors are not strictly ordered")
    if not math.isclose(anchors["UPPER_BOUND"], 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise GraderError("grader failed: theoretical macro-F1 upper must be one")
    return anchors


def _load_classes() -> list[str]:
    try:
        classes = [line.strip() for line in (INPUT_DIR / "classes.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        raise GraderError("grader failed: class vocabulary is unreadable") from exc
    if len(classes) != 15 or len(set(classes)) != 15:
        raise GraderError("grader failed: class vocabulary is invalid")
    return classes


def _load_query_ids() -> pd.Index:
    try:
        query = ad.read_h5ad(INPUT_DIR / "query.h5ad", backed="r")
    except Exception as exc:  # noqa: BLE001 - trusted image input
        raise GraderError("grader failed: sealed query could not be loaded") from exc
    try:
        if any("label" in str(column).lower() or str(column) == "ann_finest_level" for column in query.obs.columns):
            raise GraderError("grader failed: sealed query contains a label-like column")
        query_ids = pd.Index(query.obs_names.astype(str), name="cell_id")
    finally:
        query.file.close()
    if len(query_ids) != len(set(query_ids)):
        raise GraderError("grader failed: sealed query IDs are not unique")
    return query_ids


def _load_truth(payload: bytes, query_ids: pd.Index, classes: list[str]) -> pd.Series:
    try:
        frame = pd.read_csv(io.BytesIO(payload), dtype=str)
    except Exception as exc:  # noqa: BLE001 - trusted sealed input
        raise GraderError("grader failed: sealed truth could not be parsed") from exc
    if list(frame.columns) != ["cell_id", "label"] or frame.isna().any().any() or frame["cell_id"].duplicated().any():
        raise GraderError("grader failed: sealed truth schema is invalid")
    if len(frame) != len(query_ids) or set(frame["cell_id"]) != set(query_ids):
        raise GraderError("grader failed: sealed truth IDs do not match the query")
    truth = frame.set_index("cell_id")["label"].reindex(query_ids)
    if truth.isna().any() or set(truth) - set(classes):
        raise GraderError("grader failed: sealed truth contains invalid labels")
    return truth


def _validate_predictions(payload: bytes, query_ids: pd.Index, classes: list[str]) -> pd.Series:
    try:
        frame = pd.read_csv(io.BytesIO(payload), dtype=str)
    except Exception as exc:  # noqa: BLE001 - untrusted output
        raise SubmissionError("submission failed: predictions CSV could not be parsed") from exc
    if list(frame.columns) != ["cell_id", "pred_label"]:
        raise SubmissionError("submission failed: predictions must have exactly cell_id,pred_label columns")
    if frame.isna().any().any() or frame["cell_id"].duplicated().any():
        raise SubmissionError("submission failed: predictions contain missing or duplicate IDs")
    if len(frame) != len(query_ids) or set(frame["cell_id"]) != set(query_ids):
        raise SubmissionError("submission failed: prediction IDs do not match the sealed query")
    if set(frame["pred_label"]) - set(classes):
        raise SubmissionError("submission failed: predictions contain an unknown label")
    return frame.set_index("cell_id")["pred_label"].reindex(query_ids)


def grade() -> dict[str, object]:
    make_parent_nondumpable_subreaper()
    run_uid, run_gid = allocate_run_identity()
    sealed_paths = (ANCHOR_PATH, TRUTH_PATH)
    assert_sealed_denied(sealed_paths, run_uid, run_gid)
    payloads = capture_and_unlink_sealed(
        {
            ANCHOR_PATH: (4 * 1024, ANCHOR_SHA256),
            TRUTH_PATH: (1024 * 1024, TRUTH_SHA256),
        }
    )
    anchors = _load_anchors(payloads[ANCHOR_PATH])
    classes = _load_classes()
    query_ids = _load_query_ids()
    truth = _load_truth(payloads[TRUTH_PATH], query_ids, classes)
    validate_child_inputs(INPUT_PATHS, run_uid, run_gid)
    remove_experiment_log()

    workspace = Path(tempfile.mkdtemp(prefix="lung-verify-", dir="/tmp"))
    workspace.chmod(0o755)
    try:
        with staged_submission(workspace, run_uid, run_gid) as solver:
            seal_external_write_surfaces()
            run = run_submission(solver, workspace, run_uid, run_gid)
        predictions = _validate_predictions(run.predictions, query_ids, classes)
        per_class = f1_score(
            truth.to_numpy(),
            predictions.to_numpy(),
            labels=classes,
            average=None,
            zero_division=0,
        )
        metric = float(per_class.mean())
        if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
            raise GraderError("grader failed: macro-F1 is invalid")
        return {
            "correctness": True,
            "metric": "macro_f1",
            "raw_metric": metric,
            "reward": reward_for(metric, anchors),
            "error": None,
            "status": "ok",
            "isolation": {
                "run_uid": run.run_uid,
                "run_gid": run.run_gid,
                "scratch_bytes": run.scratch_bytes,
                "scratch_entries": run.scratch_entries,
                "processes_killed": run.processes_killed,
                "sysv_ipc_removed": run.sysv_ipc_removed,
                "shared_host_uid_concurrency": "unmeasured",
            },
            "method_stdout_tail": run.stdout_tail,
            "method_stderr_tail": run.stderr_tail,
        }
    finally:
        _remove_tree(workspace)


def _zero_result(marker: str, status: str) -> dict[str, object]:
    return {
        "correctness": False,
        "metric": "macro_f1",
        "raw_metric": 0.0,
        "reward": 0.0,
        "error": marker,
        "status": status,
    }


def _write_outputs(result: dict[str, object]) -> None:
    try:
        reward = float(result["reward"])
        raw_metric = float(result["raw_metric"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GraderError("grader failed: verifier result is malformed") from exc
    if not math.isfinite(reward) or not math.isfinite(raw_metric) or not 0.0 <= reward <= 1.0 or not 0.0 <= raw_metric <= 1.0:
        raise GraderError("grader failed: verifier result is outside its numeric domain")
    details: dict[str, object] = {
        "correctness": bool(result.get("correctness", False)),
        "metric": "macro_f1",
        "raw_metric": raw_metric,
        "reward": reward,
    }
    marker = result.get("error")
    if isinstance(marker, str) and marker:
        details["error"] = marker[:4096]
    debug = {**result, **details}
    # Harbor gives reward.json precedence, so publish it last. Until every
    # diagnostic is durable, the initializer's zero-valued JSON remains active.
    atomic_write(REWARD_DIR / "reward.txt", f"{reward!r}\n")
    atomic_write(REWARD_DIR / "score_details.json", json.dumps(details, allow_nan=False, sort_keys=True) + "\n")
    atomic_write(REWARD_DIR / "grade_debug.json", json.dumps(debug, allow_nan=False, sort_keys=True) + "\n")
    atomic_write(REWARD_DIR / "reward.json", json.dumps({"reward": reward}, allow_nan=False, sort_keys=True) + "\n")


def main() -> int:
    # Publish a complete zero result before touching any scorer asset. If this
    # process is killed, the shell/harness still sees an exact fail-closed set.
    write_failure_outputs("grader failed: verifier grading did not complete")
    try:
        result = grade()
        return_code = 0
    except SubmissionError as exc:
        result = _zero_result(f"submission failed: {type(exc).__name__}", "submission_failure")
        return_code = 0
    except BaseException as exc:  # noqa: BLE001 - trusted faults fail closed and demand rerun
        result = _zero_result(f"grader failed: {type(exc).__name__}", "grader_failure")
        return_code = 2
    _write_outputs(result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"method_stdout_tail", "method_stderr_tail"}}, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
