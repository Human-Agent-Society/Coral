#!/usr/bin/env python3
"""Root-owned atomic verifier output publication for the active v3 task."""

from __future__ import annotations

import json
import math
import os
import stat
import tempfile
from pathlib import Path


REWARD_DIR = Path("/logs/verifier")
OUTPUTS = ("reward.txt", "reward.json", "score_details.json", "grade_debug.json")
MAX_BYTES = 2 * 1024 * 1024
UNTRUSTED_UID = 4242


class OutputError(RuntimeError):
    pass


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def prepare_outputs(log_dir: Path = REWARD_DIR) -> None:
    """Harden the harness-owned output directory without assuming it is uid 0."""
    log_dir.mkdir(parents=True, exist_ok=True)
    info = os.lstat(log_dir)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid == UNTRUSTED_UID:
        raise OutputError("unsafe verifier output directory")
    os.chmod(log_dir, 0o700)
    hardened = os.lstat(log_dir)
    if (
        not stat.S_ISDIR(hardened.st_mode)
        or hardened.st_uid != info.st_uid
        or hardened.st_uid == UNTRUSTED_UID
        or stat.S_IMODE(hardened.st_mode) != 0o700
    ):
        raise OutputError("could not harden verifier output directory")
    for name in OUTPUTS:
        path = log_dir / name
        if _lexists(path):
            if stat.S_ISDIR(os.lstat(path).st_mode):
                raise OutputError(f"refusing planted output directory: {name}")
            path.unlink()


def secure_reward_dir(log_dir: Path = REWARD_DIR) -> None:
    """Compatibility name used by the trusted grader."""
    prepare_outputs(log_dir)


def _atomic(name: str, payload: bytes) -> None:
    if name not in OUTPUTS or not payload or len(payload) > MAX_BYTES:
        raise OutputError("invalid verifier output payload")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=REWARD_DIR)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, REWARD_DIR / name)
        os.chown(REWARD_DIR / name, 0, 0, follow_symlinks=False)
        os.chmod(REWARD_DIR / name, 0o600, follow_symlinks=False)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def validate_outputs(*, published: bool) -> dict[str, object]:
    expected_mode = 0o644 if published else 0o600
    payloads: dict[str, bytes] = {}
    for name in OUTPUTS:
        path = REWARD_DIR / name
        info = os.lstat(path)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != 0
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != expected_mode
            or info.st_size <= 0
            or info.st_size > MAX_BYTES
        ):
            raise OutputError(f"unsafe verifier output: {name}")
        payloads[name] = path.read_bytes()
    reward = float(payloads["reward.txt"].decode("utf-8").strip())
    reward_json = json.loads(payloads["reward.json"])
    details = json.loads(payloads["score_details.json"])
    debug = json.loads(payloads["grade_debug.json"])
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        raise OutputError("reward is outside [0,1]")
    if reward_json != {"reward": reward} or details != debug:
        raise OutputError("verifier output values disagree")
    if not math.isclose(float(details.get("reward", -1.0)), reward, rel_tol=0.0, abs_tol=1e-12):
        raise OutputError("detail reward disagrees")
    if (
        details.get("schema_version") != "dense-score-details-v4"
        or details.get("metric") != "mean_multithreshold_symmetric_fscore"
        or details.get("direction") != "higher"
        or details.get("calibration_status") != "active"
        or details.get("mapping_order") != "mean_then_map"
        or not isinstance(details.get("correctness"), bool)
        or not isinstance(details.get("per_case"), dict)
    ):
        raise OutputError("score details schema is invalid")
    raw = details.get("raw_metric")
    if raw is not None and (not math.isfinite(float(raw)) or not 0.0 <= float(raw) <= 1.0):
        raise OutputError("raw metric is invalid")
    return details


def publish_existing_outputs() -> None:
    validate_outputs(published=False)
    for name in OUTPUTS:
        os.chmod(REWARD_DIR / name, 0o644, follow_symlinks=False)
    os.chmod(REWARD_DIR, 0o755)
    validate_outputs(published=True)


def write_result(details: dict[str, object]) -> None:
    reward = float(details["reward"])
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        raise OutputError("invalid reward")
    encoded = (json.dumps(details, sort_keys=True, separators=(",", ":")) + "\n").encode()
    _atomic("reward.txt", f"{reward:.17g}\n".encode())
    _atomic("reward.json", (json.dumps({"reward": reward}, separators=(",", ":")) + "\n").encode())
    _atomic("score_details.json", encoded)
    _atomic("grade_debug.json", encoded)
    publish_existing_outputs()


def write_failure_outputs(
    marker: str | None = None, *, preserve_existing: bool = True
) -> None:
    del preserve_existing
    if not isinstance(marker, str) or not marker.startswith("grader failed:"):
        marker = "grader failed: verifier entrypoint did not produce valid outputs"
    prepare_outputs()
    write_result(
        {
            "schema_version": "dense-score-details-v4",
            "correctness": False,
            "metric": "mean_multithreshold_symmetric_fscore",
            "direction": "higher",
            "raw_metric": None,
            "reward": 0.0,
            "calibration_status": "active",
            "mapping_order": "mean_then_map",
            "per_case": {},
            "error": marker[:1000],
        }
    )


def finalize_grade_process(status: int) -> None:
    """Publish a valid result; otherwise publish an explicit grader-failure zero."""
    try:
        if status == 0:
            publish_existing_outputs()
            return
    except Exception:
        pass
    try:
        details = validate_outputs(published=True)
    except Exception:
        details = None
    if (
        status != 0
        and isinstance(details, dict)
        and details.get("reward") == 0.0
        and str(details.get("error", "")).startswith("grader failed:")
    ):
        return
    write_failure_outputs(f"grader failed: verifier process exited with status {status}")


if __name__ == "__main__":
    if __import__("sys").argv[1:] == ["--prepare"]:
        prepare_outputs()
    elif __import__("sys").argv[1:]:
        raise SystemExit("unexpected fail-closed helper argument")
    else:
        write_failure_outputs()
