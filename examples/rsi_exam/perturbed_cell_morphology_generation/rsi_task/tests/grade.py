#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import copy_check
from fid_metric import overall_fid


TEST_ROOT = pathlib.Path("/tests")
SUBMISSION_METHOD = pathlib.Path("/app/methods/main")
SUBMISSION_MODEL = pathlib.Path("/app/submission")
ANCHOR_FILE = TEST_ROOT / "anchors.json"
INPUT_FILE = TEST_ROOT / "sealed" / "inputs.npz"
TARGET_FILE = TEST_ROOT / "sealed" / "targets.npz"
TRAIN_REFERENCE_FILE = TEST_ROOT / "sealed" / "train_reference.npz"
LOG_ROOT = pathlib.Path("/logs/verifier")
SOLVER_UID = 65534
SOLVER_GID = 65534
CHILD_TIMEOUT_SECONDS = 1200


def reward_of(fid: float, anchors: dict) -> float:
    baseline = float(anchors["baseline"])
    upper = float(anchors["upper"])
    if not math.isfinite(fid) or not (baseline > upper >= 0.0):
        raise ValueError("invalid FID or anchor ordering")
    if fid >= baseline:
        return 0.0
    if fid <= upper:
        return 1.0
    worse = math.log1p(baseline)
    better = math.log1p(upper)
    value = (worse - math.log1p(fid)) / (worse - better)
    return float(min(1.0, max(0.0, value)))


def demote() -> None:
    os.setgroups([])
    os.setgid(SOLVER_GID)
    os.setuid(SOLVER_UID)


def assert_child_can_read(path: pathlib.Path) -> None:
    command = [sys.executable, "-c", "open(__import__('sys').argv[1], 'rb').read(1)", str(path)]
    result = subprocess.run(
        command, preexec_fn=demote, capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        raise PermissionError(f"submission unreadable after privilege drop: {path}")


def write_failure(error: Exception) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    message = f"grader failed: {type(error).__name__}: {error}"
    (LOG_ROOT / "grade_debug.json").write_text(
        json.dumps({"error": message}, indent=2) + "\n", encoding="utf-8"
    )
    (LOG_ROOT / "reward.json").write_text(
        json.dumps(
            {
                "reward": 0.0,
                "fid": 0.0,
                "overall_fid": 0.0,
                "copy_violation": 0.0,
                "duplicate_share": 0.0,
                "grader_failed": 1.0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    anchors = json.loads(ANCHOR_FILE.read_text(encoding="utf-8"))
    with np.load(TARGET_FILE, allow_pickle=False) as target_file:
        target = target_file["target"].copy()
    with np.load(INPUT_FILE, allow_pickle=False) as input_file:
        expected_ids = input_file["sample_id"].copy()
        conditions = input_file["condition"].copy()
    with np.load(TRAIN_REFERENCE_FILE, allow_pickle=False) as reference_file:
        train_reference = reference_file["treated"].copy()
    TARGET_FILE.unlink()
    TRAIN_REFERENCE_FILE.unlink()
    ANCHOR_FILE.unlink()

    solver = SUBMISSION_METHOD / "solver.py"
    checkpoint = SUBMISSION_MODEL / "checkpoint.pt"
    assert_child_can_read(solver)
    assert_child_can_read(checkpoint)

    child_root = pathlib.Path(tempfile.mkdtemp(prefix="morphology-", dir="/tmp"))
    os.chmod(child_root, 0o777)
    output = child_root / "predictions.npz"
    child_env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": str(SUBMISSION_METHOD),
        "HOME": "/tmp",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
        "CUDA_VISIBLE_DEVICES": "0",
        "TORCH_HOME": "/opt/torch",
    }
    command = [
        sys.executable,
        str(TEST_ROOT / "run_predict.py"),
        "--solver",
        str(solver),
        "--inputs",
        str(INPUT_FILE),
        "--checkpoint",
        str(SUBMISSION_MODEL),
        "--output",
        str(output),
    ]
    result = subprocess.run(
        command,
        env=child_env,
        preexec_fn=demote,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=CHILD_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise RuntimeError(f"prediction child failed with return code {result.returncode}")
    with np.load(output, allow_pickle=False) as prediction_file:
        generated = prediction_file["prediction"]
        sample_ids = prediction_file["sample_id"]
    if not np.array_equal(sample_ids, expected_ids):
        raise ValueError("sample_id order mismatch")
    if generated.shape != target.shape or generated.dtype != np.uint8:
        raise ValueError(
            f"expected uint8 predictions with shape {target.shape}, got {generated.shape} {generated.dtype}"
        )

    copied, copy_details = copy_check.violates(generated, train_reference)
    fid = overall_fid(target, generated)
    reward = 0.0 if copied else reward_of(fid, anchors)
    score_details = {
        "metric": "overall_fid_inception_2048d_after_fixed_six_channel_composite",
        "direction": "lower_is_better",
        "samples": int(len(target)),
        "conditions": int(len(np.unique(conditions))),
        "overall_fid": fid,
        "anchors": anchors,
        "copy_violation": bool(copied),
        **copy_details,
        "reward": reward,
    }
    (LOG_ROOT / "score_details.json").write_text(
        json.dumps(score_details, indent=2) + "\n", encoding="utf-8"
    )
    (LOG_ROOT / "reward.json").write_text(
        json.dumps(
            {
                "reward": reward,
                "fid": fid,
                "overall_fid": fid,
                "copy_violation": float(copied),
                "duplicate_share": float(copy_details["duplicate_share"]),
                "closest_reference_distance": float(copy_details["closest_reference_distance"]),
                "grader_failed": 0.0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.rmtree(child_root, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_failure(error)
        raise
