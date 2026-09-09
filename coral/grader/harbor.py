"""CORAL ``TaskGrader`` adapter for a pinned local Harbor task."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from coral.grader.task_grader import TaskGrader
from coral.harbor_task import (
    HARBOR_ADAPTER_MARKER,
    HARBOR_PRIVATE_TASK_DIR,
    HARBOR_RUNTIME_VERSION,
    inspect_local_harbor_task,
)
from coral.types import Score, ScoreBundle

_IGNORED_CANDIDATE_NAMES = {".git", ".coral", "__pycache__"}
_CORAL_SHARED_STATE_NAMES = {
    ".claude",
    ".codex",
    ".cursor",
    ".dsh",
    ".kiro",
    ".opencode",
    ".pi",
}


class _HarborRunnerTimeoutError(RuntimeError):
    def __init__(self, timeout: float, stdout: str, stderr: str) -> None:
        super().__init__(f"Harbor runner timed out after {timeout}s")
        self.timeout = timeout
        self.stdout = stdout
        self.stderr = stderr


def _run_harbor_runner(
    command: list[str],
    *,
    environment: dict[str, str],
    timeout: float | None,
    cleanup_grace: float,
) -> subprocess.CompletedProcess[str]:
    """Run Harbor and give asyncio cancellation time to clean Docker resources."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            process.send_signal(signal.SIGINT)
        else:
            process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=cleanup_grace)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        raise _HarborRunnerTimeoutError(timeout or 0, stdout, stderr) from None

    if process.returncode is None:
        raise RuntimeError("Harbor runner exited without a return code")
    return subprocess.CompletedProcess(
        args=command,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _parse_last_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.strip().splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("Harbor runner did not return a structured result")


def _copy_candidate_workspace(source: Path, destination: Path) -> None:
    """Copy project files without Git/CORAL state or escaping symlinks."""
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Candidate workspace not found: {source}")

    def copy_symlink(path: Path, target: Path) -> None:
        link_value = Path(os.readlink(path))
        relative_link = path.relative_to(source)
        inside_shared_state = bool(
            relative_link.parts and relative_link.parts[0] in _CORAL_SHARED_STATE_NAMES
        )
        if link_value.is_absolute():
            if inside_shared_state:
                return
            raise ValueError(f"Candidate workspace contains an absolute symlink: {path}")
        resolved = (path.parent / link_value).resolve()
        try:
            relative_target = resolved.relative_to(source)
        except ValueError as exc:
            if inside_shared_state:
                return
            raise ValueError(f"Candidate workspace symlink escapes the repository: {path}") from exc
        if relative_target.parts and relative_target.parts[0] in _IGNORED_CANDIDATE_NAMES:
            if inside_shared_state:
                return
            raise ValueError(f"Candidate workspace symlink targets private runtime state: {path}")
        target.symlink_to(link_value, target_is_directory=path.is_dir())

    destination.mkdir(parents=True)
    for root, dirs, files in os.walk(source, followlinks=False):
        root_path = Path(root)
        relative_root = root_path.relative_to(source)
        target_root = destination / relative_root
        target_root.mkdir(parents=True, exist_ok=True)

        retained_dirs: list[str] = []
        for name in dirs:
            path = root_path / name
            if name in _IGNORED_CANDIDATE_NAMES:
                continue
            if path.is_symlink():
                if name not in _CORAL_SHARED_STATE_NAMES:
                    copy_symlink(path, target_root / name)
                continue
            retained_dirs.append(name)
        dirs[:] = retained_dirs

        for name in files:
            path = root_path / name
            if name in _IGNORED_CANDIDATE_NAMES or name.endswith(".pyc"):
                continue
            if path.is_symlink():
                if name not in _CORAL_SHARED_STATE_NAMES:
                    copy_symlink(path, target_root / name)
                continue
            shutil.copy2(path, target_root / name)


def _score_bundle_from_result(
    result: dict[str, Any],
    *,
    primary_reward: str,
    direction: str,
    expected_digest: str,
    summary_path: str,
) -> ScoreBundle:
    rewards = result.get("rewards")
    if not isinstance(rewards, dict) or not rewards:
        raise RuntimeError("Harbor verifier returned no rewards")
    if primary_reward not in rewards:
        available = ", ".join(sorted(str(key) for key in rewards))
        raise RuntimeError(
            f"Harbor verifier did not return primary reward {primary_reward!r}; "
            f"available rewards: {available or '(none)'}"
        )

    scores: dict[str, Score] = {}
    for name, value in rewards.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"Harbor reward {name!r} is not numeric")
        scores[str(name)] = Score(value=value, name=str(name))

    runtime_version = result.get("runtime_version")
    if runtime_version != HARBOR_RUNTIME_VERSION:
        raise RuntimeError(
            f"Harbor runtime drift: expected {HARBOR_RUNTIME_VERSION}, got {runtime_version!r}"
        )

    metadata = {
        "harbor": {
            "adapter": HARBOR_ADAPTER_MARKER,
            "runtime_version": runtime_version,
            "schema_version": result.get("schema_version"),
            "task_name": result.get("task_name"),
            "task_package_version": result.get("task_package_version"),
            "source_digest": expected_digest,
            "harbor_task_checksum": result.get("task_digest"),
            "primary_reward": primary_reward,
            "direction": direction,
            "summary": summary_path,
        }
    }
    reward_value = scores[primary_reward].value
    return ScoreBundle(
        scores=scores,
        aggregated=float(reward_value) if reward_value is not None else None,
        feedback=f"Harbor verifier completed. Sanitized summary: `{summary_path}`",
        metadata=metadata,
    )


class HarborTaskGrader(TaskGrader):
    """Evaluate a CORAL candidate with Harbor v0.22.0 in an isolated runtime."""

    def evaluate(self) -> ScoreBundle:
        args = self.args
        if args.get("harbor_adapter") != HARBOR_ADAPTER_MARKER:
            raise ValueError("Invalid or missing Harbor adapter marker")
        if args.get("harbor_runtime_version") != HARBOR_RUNTIME_VERSION:
            raise ValueError(
                "Harbor adapter runtime must remain pinned to "
                f"{HARBOR_RUNTIME_VERSION}, got {args.get('harbor_runtime_version')!r}"
            )

        task_subdir = str(args.get("harbor_task_subdir", ""))
        expected_digest = str(args.get("harbor_task_digest", ""))
        primary_reward = str(args.get("primary_reward", ""))
        if not task_subdir or not expected_digest or not primary_reward:
            raise ValueError("Incomplete Harbor adapter configuration")
        if task_subdir != HARBOR_PRIVATE_TASK_DIR:
            raise ValueError(
                f"Harbor task must be staged at {HARBOR_PRIVATE_TASK_DIR!r}, got {task_subdir!r}"
            )

        private_dir = Path(self.private_dir).resolve()
        task_dir = private_dir / task_subdir
        descriptor = inspect_local_harbor_task(task_dir)
        if descriptor.digest != expected_digest:
            raise RuntimeError(
                "Staged Harbor task digest changed: "
                f"expected {expected_digest}, got {descriptor.digest}"
            )
        expected_schema = str(args.get("harbor_schema_version", ""))
        expected_name = str(args.get("harbor_task_name", ""))
        if descriptor.schema_version != expected_schema:
            raise RuntimeError(
                "Staged Harbor schema differs from the resolved run config: "
                f"expected {expected_schema!r}, got {descriptor.schema_version!r}"
            )
        if descriptor.name != expected_name:
            raise RuntimeError(
                "Staged Harbor task name differs from the resolved run config: "
                f"expected {expected_name!r}, got {descriptor.name!r}"
            )

        private_run = private_dir / "harbor_runs" / Path(self.codebase_path).name / uuid4().hex
        private_run.mkdir(parents=True)
        request_path = private_run / "request.json"
        candidate_path = private_run / "candidate"
        try:
            _copy_candidate_workspace(Path(self.codebase_path).resolve(), candidate_path)
        except Exception:
            shutil.rmtree(candidate_path, ignore_errors=True)
            raise
        request_path.write_text(
            json.dumps(
                {
                    "task_path": str(task_dir),
                    "candidate_path": str(candidate_path),
                    "trials_dir": str(private_run / "trials"),
                    "trial_name": f"coral-eval-{private_run.name[:12]}",
                }
            ),
            encoding="utf-8",
        )

        runner = Path(__file__).parent.parent / "_runners" / "harbor_task.py"
        command = [
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--no-config",
            "--python",
            "3.12",
            "--with",
            f"harbor=={HARBOR_RUNTIME_VERSION}",
            "python",
            str(runner),
            str(request_path),
        ]
        environment = os.environ.copy()
        environment.pop("VIRTUAL_ENV", None)
        environment.pop("UV_PROJECT_ENVIRONMENT", None)
        environment.setdefault(
            "UV_CACHE_DIR",
            str(Path(tempfile.gettempdir()) / "coral-harbor-uv-cache"),
        )
        environment.setdefault("UV_NO_PROGRESS", "1")

        if self.timeout is None:
            runner_timeout = None
            cleanup_grace = 20.0
        else:
            cleanup_grace = float(min(20, max(1, self.timeout // 10)))
            runner_timeout = float(max(1, self.timeout - cleanup_grace - 1))
        stdout = ""
        stderr = ""
        try:
            completed = _run_harbor_runner(
                command,
                environment=environment,
                timeout=runner_timeout,
                cleanup_grace=cleanup_grace,
            )
            stdout = completed.stdout
            stderr = completed.stderr
        except _HarborRunnerTimeoutError as exc:
            stdout = exc.stdout
            stderr = exc.stderr
            raise RuntimeError(f"Harbor evaluation timed out after {runner_timeout}s") from None
        finally:
            # The immutable attempt commit remains in CORAL's repo; retaining
            # another full candidate copy beside Harbor's private logs would
            # multiply disk use on every evaluation.
            shutil.rmtree(candidate_path, ignore_errors=True)
            (private_run / "runner.stdout.log").write_text(stdout, encoding="utf-8")
            (private_run / "runner.stderr.log").write_text(stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"Harbor runner exited {completed.returncode}; "
                "details are retained in manager-only Harbor run logs"
            )

        payload = _parse_last_json(stdout)
        if "error" in payload:
            error_name = str(payload["error"]).split(":", 1)[0]
            if not error_name.replace("_", "").isalnum():
                error_name = "HarborError"
            raise RuntimeError(
                f"Harbor evaluation failed ({error_name}); "
                "details are retained in manager-only Harbor run logs"
            )
        result = payload.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Harbor runner response is missing result data")
        if result.get("schema_version") != descriptor.schema_version:
            raise RuntimeError(
                "Harbor runner returned an unexpected schema version: "
                f"{result.get('schema_version')!r}"
            )
        if result.get("task_name") != descriptor.name:
            raise RuntimeError(
                f"Harbor runner returned an unexpected task name: {result.get('task_name')!r}"
            )

        public_summary = self.eval_logs_dir / "harbor-summary.json"
        summary_path = self.eval_logs_worktree_path(public_summary).as_posix()
        bundle = _score_bundle_from_result(
            result,
            primary_reward=primary_reward,
            direction=self.config.direction,
            expected_digest=expected_digest,
            summary_path=summary_path,
        )
        summary = {
            "runtime_version": result.get("runtime_version"),
            "schema_version": result.get("schema_version"),
            "task_name": result.get("task_name"),
            "task_package_version": result.get("task_package_version"),
            "source_digest": expected_digest,
            "harbor_task_checksum": result.get("task_digest"),
            "primary_reward": primary_reward,
            "direction": self.config.direction,
            "rewards": result.get("rewards"),
        }
        temporary_summary = public_summary.with_suffix(".json.tmp")
        temporary_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        temporary_summary.replace(public_summary)
        return bundle
