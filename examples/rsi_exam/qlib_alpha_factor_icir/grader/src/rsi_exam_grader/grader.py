"""Replay only the submission into RSI-Exam's original two-container protocol."""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import tomllib
import uuid
from pathlib import Path

from coral.grader import TaskGrader
from rsi_exam_grader.contract import artifact_sources, submission_path


def read_reward(job_dir: Path, key: str = "reward") -> float:
    """Read one completed trial, never a job mean or a candidate-written file."""
    results = list(job_dir.glob("*/result.json"))
    if len(results) != 1:
        raise ValueError(f"Expected one Harbor trial result, found {len(results)}")
    result = json.loads(results[0].read_text())
    if result.get("exception_info"):
        raise ValueError("Harbor trial failed; inspect the private trial logs")
    if not result.get("finished_at"):
        raise ValueError("Harbor trial has not finished")
    if result.get("verifier_environment_mode") != "separate":
        raise ValueError("Harbor did not run a separate verifier")
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    value = rewards.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing or non-finite Harbor reward {key!r}")
    # Some GPU tasks intentionally emit raw speedups. Do not clip or renormalize.
    return float(value)


class Grader(TaskGrader):
    def describe_tune(self) -> str:
        return (
            "Hidden grading is disabled for --tune. Run `uv run rsi_runtime.py run "
            "'python /app/selfcheck.py'` in your worktree for visible feedback. "
            "The visible metric and hidden reward need not have the same scale."
        )

    def evaluate(self):
        if self.tune:
            return self.fail(self.describe_tune())
        task_dir = Path(self.private_dir) / "rsi_task"
        assets = task_dir / "assets.json"
        if assets.exists():
            missing = [
                entry["path"]
                for entry in json.loads(assets.read_text())
                if not (task_dir / entry["path"]).is_file()
            ]
            if missing:
                raise RuntimeError(
                    "Task data is not prepared. Before starting CORAL, run "
                    "`uv run examples/rsi_exam/prepare.py <task_name>`. "
                    f"Missing {len(missing)} upstream assets."
                )
        config = tomllib.loads((task_dir / "task.toml").read_text())
        if config.get("verifier", {}).get("environment_mode") != "separate":
            raise RuntimeError("RSI-Exam requires verifier.environment_mode = separate")
        sources = artifact_sources(config)
        try:
            for source in sources:
                submission_path(Path(self.codebase_path).resolve(), source)
        except ValueError as exc:
            return self.fail(str(exc))

        # Harbor logs/configs may contain hidden cases and resolved API credentials.
        # Keep the entire trial private; publish only the selected numeric reward.
        jobs = Path(self.private_dir) / "rsi_jobs"
        jobs.mkdir(exist_ok=True)
        job = "eval-" + uuid.uuid4().hex
        log_path = jobs / f"{job}.log"
        runtime = jobs / f"{job}-runtime"
        runtime.mkdir()
        package = Path(__file__).parent
        shutil.copy2(package / "replay.py", runtime / "rsi_runtime.py")
        shutil.copy2(package / "contract.py", runtime / "rsi_contract.py")
        command = [
            # Harbor's LiteLLM requirement conflicts with CORAL's pinned version.
            # uv caches a separate Python 3.12 environment for this command.
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--python",
            "3.12",
            "--with",
            "harbor==0.22.0",
            "python",
            "-m",
            "harbor.cli.main",
            "run",
            "--path",
            str(task_dir),
            "--agent-import-path",
            "rsi_runtime:ReplayAgent",
            "--n-attempts",
            "1",
            "--n-concurrent",
            "1",
            "--max-retries",
            "0",
            "--jobs-dir",
            str(jobs),
            "--job-name",
            job,
            "--agent-kwarg",
            "source_dir=" + json.dumps(self.codebase_path),
            "--agent-kwarg",
            "sources=" + json.dumps(sources),
        ]
        # Do not import modules from the candidate checkout or its PYTHONPATH.
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        try:
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    command,
                    cwd=runtime,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    # Give Harbor its own timeout/cleanup first. The CORAL limit
                    # includes both image builds and the upstream verifier budget.
                    code = process.wait(timeout=max(1, self.timeout - 15) if self.timeout else None)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise RuntimeError(f"Harbor timed out; private diagnostics: rsi_jobs/{job}.log")
            if code:
                raise RuntimeError(f"Harbor exited {code}; private diagnostics: rsi_jobs/{job}.log")
            value = read_reward(jobs / job, self.args.get("reward_key", "reward"))
        except (OSError, ValueError) as exc:
            # Exception messages from Harbor result.json are deliberately not echoed.
            raise RuntimeError(f"{exc}; private diagnostics: rsi_jobs/{job}.log") from exc
        return self.score(
            value,
            explanation=f"RSI-Exam hidden reward: {value:.6g}",
            metadata={
                "harbor_job": job,
                "protocol": "coral_repeated_hidden_evaluation",
                "reward_key": self.args.get("reward_key", "reward"),
            },
        )
