"""Run qlib's unchanged verifier in a fresh Docker container without networking.

This narrow backend avoids Harbor's nftables requirement on older Docker VMs.
Only /app/methods crosses into the container. No host paths or Docker sockets
are mounted, and verifier output remains private, as with the Harbor backend.
"""

import json
import math
import subprocess
import tomllib
import uuid
from pathlib import Path

from coral.grader import TaskGrader
from rsi_exam_grader.contract import artifact_sources, submission_path
from rsi_exam_grader.qlib_docker import PLATFORM, ensure_image


class Grader(TaskGrader):
    def describe_tune(self):
        return "Hidden grading is disabled for --tune. Run `.venv/bin/python selfcheck.py` locally."

    def evaluate(self):
        if self.tune:
            return self.fail(self.describe_tune())
        task = Path(self.private_dir) / "rsi_task"
        config = tomllib.loads((task / "task.toml").read_text())
        verifier = config["verifier"]
        if (
            config["task"]["name"] != "autoresearch-arena/qlib_alpha_factor_icir"
            or artifact_sources(config) != ["/app/methods"]
            or verifier["environment_mode"] != "separate"
            or verifier["environment"] != {"network_mode": "no-network"}
            or verifier.get("env")
        ):
            raise RuntimeError("Unsupported qlib Docker contract; use the Harbor backend")
        try:
            methods = submission_path(Path(self.codebase_path).resolve(), "/app/methods")
        except ValueError as exc:
            return self.score(0.0, str(exc))
        if not (methods / "main/solver.py").is_file():
            return self.score(0.0, "Missing methods/main/solver.py")

        job = "docker-" + uuid.uuid4().hex
        job_dir = Path(self.private_dir) / "rsi_jobs" / job
        job_dir.mkdir(parents=True)
        container = "coral-rsi-" + uuid.uuid4().hex
        stage = "build"
        diagnostics = f"rsi_jobs/{job}/docker.log"
        try:
            with (job_dir / "docker.log").open("w") as log:

                def run(*args, timeout=60):
                    return subprocess.run(
                        ["docker", *args],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=timeout,
                        cwd=job_dir,
                    )

                image = ensure_image(task / "tests", log)
                stage = "create"
                try:
                    # The explicit verifier environment has no CPU/memory overrides.
                    run(
                        "create",
                        "--platform",
                        PLATFORM,
                        "--network",
                        "none",
                        "--name",
                        container,
                        image,
                        "bash",
                        "-c",
                        "mkdir -p /app/methods && cp -a /submission/. /app/methods/ "
                        "&& exec bash /tests/test.sh",
                    )
                    run("cp", str(methods), f"{container}:/submission")
                    stage = "verify"
                    try:
                        run("start", "-a", container, timeout=verifier["timeout_sec"])
                    except subprocess.TimeoutExpired:
                        return self.score(0.0, "Submission exceeded the upstream 1800-second limit")
                    stage = "read_reward"
                    output = job_dir / "verifier"
                    output.mkdir()
                    run("cp", f"{container}:/logs/verifier/.", str(output))
                    reward = json.loads((output / "reward.json").read_text())["reward"]
                    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
                        raise ValueError("Invalid verifier reward")
                    if not math.isfinite(reward):
                        raise ValueError("Non-finite verifier reward")
                finally:
                    # Also remove the container after upload errors and verifier timeouts.
                    subprocess.run(
                        ["docker", "rm", "-f", container],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=30,
                        check=False,
                    )
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            message = (
                f"Qlib Docker infrastructure failed at {stage}; private diagnostics: {diagnostics}"
            )
            (self.eval_logs_dir / "diagnostics.txt").write_text(message + "\n")
            # A raised exception is classified as grader_error by the daemon.
            # Do not publish raw verifier output or exceptions containing hidden data.
            raise RuntimeError(message) from exc
        return self.score(
            float(reward),
            explanation=f"RSI-Exam hidden reward: {reward:.6g}",
            metadata={
                "backend": "docker_none",
                "private_job": job,
                "protocol": "coral_repeated_hidden_evaluation",
            },
        )
