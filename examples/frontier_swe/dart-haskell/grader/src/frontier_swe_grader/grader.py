"""Evaluate a CORAL source bundle with a pinned official Frontier-SWE task."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from coral.grader import TaskGrader
from coral.types import ScoreBundle

from .bundle import BundleError, parse_bundle

UPSTREAM_URL = "https://github.com/Proximal-Labs/frontier-swe.git"
UPSTREAM_COMMIT = "111464af7933002a9192240798cbcf65d2790296"
HARBOR_VERSION = "0.22.0"
MAX_BUNDLE_BYTES = 10 * 1024 * 1024
TASK_TARGETS = {
    "git-to-zig": "/app/zig-port",
    "lua-native-compiler": "/app/lua-native-compiler",
    "libexpat-to-x86asm": "/app/asm-port",
    "dart-style-haskell": "/app/dart-style",
}
TASK_IMAGES = {
    "git-to-zig": "ghcr.io/proximal-labs/frontier-swe/git-to-zig@sha256:04a78190d62e5ab7b396d8cb094f964149531aad5277aa3d98ec833f4ea7eb00",
    "lua-native-compiler": "ghcr.io/proximal-labs/frontier-swe/lua-native-compiler@sha256:20d39c61f936bec8a6f7e1681a3c78f978a7fd4968913e31fb06a0929a4a9561",
    "libexpat-to-x86asm": "ghcr.io/proximal-labs/frontier-swe/libexpat-to-x86asm@sha256:c2b36a040add352f7f19ddb1efcb10e2259e4540835b7b480e166b00d41f61bd",
    "dart-style-haskell": "ghcr.io/proximal-labs/frontier-swe/dart-style-haskell@sha256:c0d816853d0bdbda0761c000edca9e7f10a2c3926e7d7ff66e96ecf38903eb8d",
}
TASK_PREPARE_COMMANDS = {
    "libexpat-to-x86asm": "cd /app/asm-port && make",
}


class Grader(TaskGrader):
    """Run one official Harbor verifier and return its scalar reward."""

    def evaluate(self) -> ScoreBundle:
        try:
            return self._evaluate()
        except Exception as error:
            return self.fail(f"Frontier-SWE grader failed: {type(error).__name__}: {error}")

    def _evaluate(self) -> ScoreBundle:
        task_id = str(self.args.get("task_id", ""))
        if task_id not in TASK_TARGETS:
            return self.fail(f"unsupported Frontier-SWE task_id: {task_id!r}")

        program_file = str(self.args.get("program_file", "candidate.bundle"))
        candidate = Path(self.codebase_path) / program_file
        bundle_error = _validate_candidate(candidate)
        if bundle_error is not None:
            return self.fail(bundle_error)

        task_dir = self._official_task_dir(task_id)
        harbor_logs = Path(self.eval_logs_dir) / "harbor_logs"
        harbor_logs.mkdir(parents=True, exist_ok=True)
        job_name = f"frontier_swe_{task_id}_{int(time.time())}"
        environment = os.environ.get(
            "CORAL_FRONTIER_SWE_HARBOR_ENVIRONMENT",
            str(self.args.get("environment", "docker")),
        )

        command = _harbor_command(
            task_dir=task_dir,
            candidate=candidate,
            target_dir=TASK_TARGETS[task_id],
            prepare_command=TASK_PREPARE_COMMANDS.get(task_id, ""),
            logs_dir=harbor_logs,
            job_name=job_name,
            environment=environment,
        )
        env = os.environ.copy()
        grader_src = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = os.pathsep.join(
            path for path in (grader_src, env.get("PYTHONPATH", "")) if path
        )

        try:
            completed = subprocess.run(
                command,
                cwd=task_dir,
                env=env,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return self.fail(f"Harbor evaluation timed out after {self.timeout}s: {error}")
        except OSError as error:
            return self.fail(f"could not launch Harbor: {error}")

        job_dir = harbor_logs / job_name
        parsed = _read_trial_score(job_dir)
        if isinstance(parsed, str):
            return self.fail(
                f"Harbor produced no official score (exit {completed.returncode}): {parsed}. "
                f"{_diagnostic(completed)}"
            )

        reward, trial = parsed
        detail = _read_reward_detail(job_dir)
        feedback = _feedback(
            task_id,
            reward,
            environment,
            self.eval_logs_worktree_path(job_dir),
            detail,
            trial,
        )
        explanation = f"official frontier_reward={reward:.12g} ({task_id}, Harbor {environment})"
        return self.score(
            reward,
            explanation=explanation,
            feedback=feedback,
            metadata={
                "task_id": task_id,
                "upstream_commit": UPSTREAM_COMMIT,
                "container_image": TASK_IMAGES[task_id],
                "harbor_environment": environment,
            },
        )

    def _official_task_dir(self, task_id: str) -> Path:
        private_dir = Path(self.private_dir)
        private_dir.mkdir(parents=True, exist_ok=True)
        checkout = private_dir / f"frontier-swe-{UPSTREAM_COMMIT[:12]}"
        task_dir = checkout / "tasks" / task_id
        if _checkout_matches(checkout, task_dir):
            _pin_task_image(task_dir, task_id)
            return task_dir

        if checkout.exists():
            shutil.rmtree(checkout)
        with tempfile.TemporaryDirectory(
            prefix="frontier-swe-checkout-", dir=private_dir
        ) as temporary:
            staged = Path(temporary) / "repo"
            commands = [
                ["git", "init", str(staged)],
                ["git", "-C", str(staged), "remote", "add", "origin", UPSTREAM_URL],
                [
                    "git",
                    "-C",
                    str(staged),
                    "sparse-checkout",
                    "set",
                    "--no-cone",
                    f"/tasks/{task_id}/",
                    f"!/tasks/{task_id}/solution/",
                    f"!/tasks/{task_id}/environment/*",
                    f"/tasks/{task_id}/environment/Dockerfile",
                ],
                [
                    "git",
                    "-C",
                    str(staged),
                    "fetch",
                    "--filter=blob:none",
                    "--depth",
                    "1",
                    "origin",
                    UPSTREAM_COMMIT,
                ],
                ["git", "-C", str(staged), "checkout", "--detach", "FETCH_HEAD"],
            ]
            for command in commands:
                result = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    timeout=600,
                    check=False,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"upstream checkout command failed: {' '.join(command)}; "
                        f"{_diagnostic(result)}"
                    )
            staged.replace(checkout)

        if not _checkout_matches(checkout, task_dir):
            raise RuntimeError("pinned upstream checkout is incomplete or at the wrong commit")
        _pin_task_image(task_dir, task_id)
        return task_dir


def _validate_candidate(candidate: Path) -> str | None:
    try:
        size = candidate.stat().st_size
    except OSError as error:
        return f"candidate bundle is unavailable: {error}"
    if size > MAX_BUNDLE_BYTES:
        return f"candidate bundle exceeds {MAX_BUNDLE_BYTES} bytes"
    try:
        files = parse_bundle(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, BundleError) as error:
        return f"invalid candidate bundle: {error}"
    return None if files else "candidate bundle contains no files"


def _harbor_command(
    *,
    task_dir: Path,
    candidate: Path,
    target_dir: str,
    prepare_command: str,
    logs_dir: Path,
    job_name: str,
    environment: str,
) -> list[str]:
    package = (
        f"harbor[modal]=={HARBOR_VERSION}"
        if environment == "modal"
        else f"harbor=={HARBOR_VERSION}"
    )
    agent_args = [
        "--agent-kwarg",
        f"candidate_path={candidate}",
        "--agent-kwarg",
        f"target_dir={target_dir}",
    ]
    if prepare_command:
        agent_args.extend(["--agent-kwarg", f"prepare_command={prepare_command}"])

    return [
        "uvx",
        "--python",
        "3.12",
        "--from",
        package,
        "harbor",
        "run",
        "--path",
        str(task_dir),
        "--agent",
        "frontier_swe_grader.bundle_agent:BundleAgent",
        *agent_args,
        "--verifier",
        "frontier_swe_grader.reward_verifier:FrontierSWEVerifier",
        "--env",
        environment,
        "--jobs-dir",
        str(logs_dir),
        "--job-name",
        job_name,
        "--n-attempts",
        "1",
        "--n-concurrent",
        "1",
        "--yes",
        "--quiet",
    ]


def _checkout_matches(checkout: Path, task_dir: Path) -> bool:
    if not task_dir.is_dir():
        return False
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == UPSTREAM_COMMIT


def _pin_task_image(task_dir: Path, task_id: str) -> None:
    task_config = task_dir / "task.toml"
    text = task_config.read_text(encoding="utf-8")
    pinned_line = f'docker_image = "{TASK_IMAGES[task_id]}"'
    if pinned_line in text:
        return
    tagged_line = f'docker_image = "ghcr.io/proximal-labs/frontier-swe/{task_id}:v4"'
    if text.count(tagged_line) != 1:
        raise RuntimeError(f"could not locate the expected image tag in {task_config}")
    task_config.write_text(text.replace(tagged_line, pinned_line), encoding="utf-8")


def _read_trial_score(job_dir: Path) -> tuple[float, dict[str, Any]] | str:
    if not job_dir.is_dir():
        return f"job directory is missing: {job_dir}"
    errors: list[str] = []
    for result_path in sorted(job_dir.rglob("result.json")):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{result_path.name}: {error}")
            continue
        verifier = payload.get("verifier_result")
        if not isinstance(verifier, dict):
            continue
        rewards = verifier.get("rewards")
        if not isinstance(rewards, dict):
            continue
        try:
            reward = float(rewards["reward"])
        except (KeyError, TypeError, ValueError):
            continue
        return reward, payload
    return "; ".join(errors) or "no trial result contained verifier_result.rewards.reward"


def _read_reward_detail(job_dir: Path) -> dict[str, Any]:
    for reward_path in sorted(job_dir.rglob("reward.json")):
        try:
            payload = json.loads(reward_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _feedback(
    task_id: str,
    reward: float,
    environment: str,
    job_dir: Path,
    detail: dict[str, Any],
    trial: dict[str, Any],
) -> str:
    lines = [
        f"Official Frontier-SWE result for `{task_id}`: {reward:.12g}",
        f"Harbor environment: `{environment}`",
        f"Upstream commit: `{UPSTREAM_COMMIT}`",
        f"Container image: `{TASK_IMAGES[task_id]}`",
        f"Logs: `{job_dir}`",
    ]
    reason = detail.get("reason")
    if reason:
        lines.append(f"Reason: {reason}")
    for key in (
        "tests_passed",
        "tests_total",
        "total_passed",
        "total_failed",
        "total_skipped",
        "score",
        "reward",
    ):
        if key in detail:
            lines.append(f"{key}: {detail[key]}")
    if trial.get("exception_info"):
        lines.append(f"Trial exception: {trial['exception_info']}")
    return "\n".join(lines)


def _diagnostic(completed: subprocess.CompletedProcess[str]) -> str:
    pieces = []
    for label, value in (("stdout", completed.stdout), ("stderr", completed.stderr)):
        text = (value or "").strip()
        if text:
            pieces.append(f"{label}: {text[-2000:]}")
    return " | ".join(pieces) or "no subprocess output"
