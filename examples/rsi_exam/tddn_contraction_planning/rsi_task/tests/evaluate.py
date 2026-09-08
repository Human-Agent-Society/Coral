"""Trusted hidden evaluator with per-case solver and TDD process isolation."""
from __future__ import annotations

import json
import math
import os
import secrets
import select
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from tdd_backend import make_environment


# Sealed scoring band. Read at import; 0400 root:root in the verifier image.
ANCHORS = json.loads((Path(__file__).resolve().parent / "anchors.json").read_text())
PEAK_WEIGHT = float(ANCHORS["peak_weight"])
TIME_WEIGHT = float(ANCHORS["time_weight"])
# per-case planning budget. Planning time does not score, but a submission that
# never finishes must fail as the submission, not as a grader timeout.
PLANNING_BUDGET_SEC = 30.0

WORKER = Path("/usr/local/libexec/tddn_solver_worker.py")
NOBODY_UID = 65534
NOBODY_GID = 65534


def enabled(observation: dict[str, Any], edge_id: int) -> bool:
    mask = observation["action_mask"]
    return bool(mask.get(str(edge_id), mask.get(edge_id, False)))


class SolverWorker:
    def __init__(
        self,
        solver_path: Path,
        *,
        unprivileged: bool,
        action_timeout: float = 5.0,
    ) -> None:
        self._timeout = action_timeout
        self._pending = b""
        self._sandbox = Path(tempfile.mkdtemp(prefix="tddn-solver-"))
        if unprivileged:
            copied_methods = self._sandbox / "methods"
            shutil.copytree(solver_path.parent, copied_methods)
            solver_path = copied_methods / solver_path.name
            for path in [self._sandbox, *self._sandbox.rglob("*")]:
                os.chown(path, NOBODY_UID, NOBODY_GID)
            os.chown(self._sandbox, NOBODY_UID, NOBODY_GID)
            self._sandbox.chmod(0o700)

        def drop_privileges() -> None:
            if not unprivileged:
                return
            os.setgroups([])
            os.setgid(NOBODY_GID)
            os.setuid(NOBODY_UID)

        env = {
            "HOME": str(self._sandbox),
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        self._process = subprocess.Popen(
            ["python3", str(WORKER), str(solver_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=self._sandbox,
            env=env,
            preexec_fn=drop_privileges,
            bufsize=0,
        )
        message = self._read()
        if message.get("type") != "ready":
            self.close()
            raise RuntimeError(f"solver worker did not become ready: {message}")

    def _read(self) -> dict[str, Any]:
        assert self._process.stdout is not None
        fd = self._process.stdout.fileno()
        deadline = time.monotonic() + self._timeout
        while True:
            while b"\n" in self._pending:
                raw, self._pending = self._pending.split(b"\n", 1)
                try:
                    value = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict):
                    return value
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("select_edge timed out")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                raise TimeoutError("select_edge timed out")
            chunk = os.read(fd, 65536)
            if not chunk:
                raise RuntimeError(
                    f"solver worker exited with status {self._process.poll()}"
                )
            self._pending += chunk

    def select(self, observation: dict[str, Any]) -> int:
        assert self._process.stdin is not None
        nonce = secrets.token_hex(16)
        request = {"nonce": nonce, "observation": observation}
        self._process.stdin.write(
            (json.dumps(request, separators=(",", ":")) + "\n").encode()
        )
        self._process.stdin.flush()
        response = self._read()
        if response.get("type") == "error":
            raise RuntimeError(str(response.get("error", "solver error")))
        if response.get("type") != "action" or response.get("nonce") != nonce:
            raise RuntimeError("invalid solver-worker response")
        return int(response["action"])

    def close(self) -> None:
        if getattr(self, "_process", None) is not None:
            if self._process.poll() is None:
                self._process.kill()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        if getattr(self, "_sandbox", None) is not None:
            shutil.rmtree(self._sandbox, ignore_errors=True)

    def __enter__(self) -> "SolverWorker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def run_case(
    solver_path: Path,
    case: dict[str, Any],
    *,
    unprivileged: bool,
) -> dict[str, Any]:
    planning_seconds = 0.0
    with make_environment(case) as environment, SolverWorker(
        solver_path, unprivileged=unprivileged
    ) as solver:
        while not environment.is_terminal:
            observation = environment.observation()
            started = time.perf_counter()
            action = solver.select(observation)
            planning_seconds += time.perf_counter() - started
            if planning_seconds > PLANNING_BUDGET_SEC:
                raise TimeoutError(
                    f"planning budget exhausted: {planning_seconds:.1f}s "
                    f"> {PLANNING_BUDGET_SEC:.0f}s"
                )
            if not enabled(observation, action):
                raise RuntimeError(f"invalid masked edge_id {action}")
            environment.contract(action)
        result = environment.result()

    contraction_seconds = float(result["contraction_time_seconds"])
    return {
        "case_id": case["case_id"],
        "family": case["family"],
        "category": case["category"],
        "correct": bool(result["correct"]),
        "steps": int(result["steps"]),
        "peak_tdd_nodes": int(result["peak_tdd_nodes"]),
        "planning_time_seconds": planning_seconds,
        "contraction_time_seconds": contraction_seconds,
        "total_time_seconds": planning_seconds + contraction_seconds,
    }


def evaluate_suite(
    solver_path: Path,
    cases: list[dict[str, Any]],
    *,
    unprivileged: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        try:
            rows.append(run_case(
                solver_path,
                case,
                unprivileged=unprivileged,
            ))
        except Exception as error:
            rows.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "category": case["category"],
                "correct": False,
                "error": f"{type(error).__name__}: {error}",
            })
    return {
        "correct": all(row.get("correct") is True for row in rows),
        "cases_passed": sum(row.get("correct") is True for row in rows),
        "cases_total": len(rows),
        "cases": rows,
    }


def mean_log_ratio(
    numerator: dict[str, dict[str, Any]],
    denominator: dict[str, dict[str, Any]],
    field: str,
) -> float:
    values = []
    for case_id, top in numerator.items():
        bottom = denominator[case_id]
        a = float(top[field])
        b = float(bottom[field])
        if a <= 0 or b <= 0:
            raise ValueError(f"non-positive {field} for {case_id}")
        values.append(math.log(a / b))
    return sum(values) / len(values)


def score_suites(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    """baseline->0, upper->1; linear in combined log gain."""
    if not candidate["correct"]:
        return {
            "reward": 0.0,
            "correct": False,
            "error": "candidate failed correctness or execution",
        }
    if not baseline["correct"] or not reference["correct"]:
        return {
            "reward": 0.0,
            "correct": False,
            "error": (
                "grader failed: trusted baseline/reference did not complete; "
                "task configuration invalid"
            ),
        }

    def by_id(suite: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {row["case_id"]: row for row in suite["cases"]}

    cand = by_id(candidate)
    base = by_id(baseline)
    ref = by_id(reference)
    candidate_peak_gain = mean_log_ratio(base, cand, "peak_tdd_nodes")
    candidate_time_gain = mean_log_ratio(base, cand, "contraction_time_seconds")
    reference_peak_gain = mean_log_ratio(base, ref, "peak_tdd_nodes")
    reference_time_gain = mean_log_ratio(base, ref, "contraction_time_seconds")
    candidate_gain = PEAK_WEIGHT * candidate_peak_gain + TIME_WEIGHT * candidate_time_gain
    reference_gain = PEAK_WEIGHT * reference_peak_gain + TIME_WEIGHT * reference_time_gain

    upper_gain = float(ANCHORS["upper_combined_log_gain"])
    if upper_gain <= 0:
        return {
            "reward": 0.0,
            "correct": False,
            "error": (
                "grader failed: upper anchor is not above the baseline "
                f"(upper={upper_gain:.6g})"
            ),
        }

    if candidate_gain <= 0:
        reward = 0.0
    else:
        reward = candidate_gain / upper_gain
    reward = max(0.0, min(1.0, reward))

    return {
        "reward": round(float(reward), 6),
        "correct": True,
        "combined_log_gain": candidate_gain,
        "reference_combined_log_gain": reference_gain,
        "upper_combined_log_gain": upper_gain,
        "peak_tdd_nodes_speedup_geomean": math.exp(candidate_peak_gain),
        "total_time_speedup_geomean": math.exp(candidate_time_gain),
        "reference_peak_speedup_geomean": math.exp(reference_peak_gain),
        "reference_time_speedup_geomean": math.exp(reference_time_gain),
    }
