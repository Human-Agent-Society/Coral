# /// script
# requires-python = ">=3.12"
# dependencies = ["harbor==0.22.0"]
# ///
"""Upload committed artifacts without running candidate code on the host.

prepare.py also copies this file to seed/rsi_runtime.py for visible experiments.
The grader imports its own installed copy, never the candidate's copy.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
import uuid
from pathlib import Path, PurePosixPath

from harbor.agents.base import BaseAgent

if __package__:
    from .contract import artifact_sources, submission_path
else:
    from rsi_contract import artifact_sources, submission_path


class ReplayAgent(BaseAgent):
    """A Harbor transfer adapter. CORAL performs optimization outside this trial."""

    @staticmethod
    def name() -> str:
        return "coral-rsi-replay"

    def version(self) -> str:
        return "0.1.0"

    def __init__(self, *args, source_dir: str, sources: list[str], **kwargs):
        super().__init__(*args, **kwargs)
        self.source_dir = Path(source_dir).resolve()
        self.sources = artifact_sources({"artifacts": sources})

    async def setup(self, environment) -> None:
        pass

    async def upload(self, environment) -> None:
        # Validate all files before modifying the container. No harness/config/log
        # files from the checkout cross this boundary.
        paths = [(s, submission_path(self.source_dir, s)) for s in self.sources]
        for source, path in paths:
            parent = str(PurePosixPath(source).parent)
            result = await environment.exec(
                command=f"rm -rf -- {shlex.quote(source)} && mkdir -p -- {shlex.quote(parent)}",
                user="root",
            )
            if result.return_code:
                raise RuntimeError("Could not reset the submission in the container")
            # Absence is a deletion, never permission to reuse the baked-in baseline.
            if path.is_dir():
                await environment.upload_dir(source_dir=path, target_dir=source)
            elif path.is_file():
                await environment.upload_file(source_path=path, target_path=source)

    async def run(self, instruction, environment, context) -> None:
        await self.upload(environment)


class VisibleAgent(ReplayAgent):
    """Run a user-specified visible experiment and retrieve its declared artifacts."""

    def __init__(self, *args, command: str = "", bootstrap: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.command = command
        self.bootstrap = bootstrap

    async def run(self, instruction, environment, context) -> None:
        if not self.bootstrap:
            await self.upload(environment)
        result = None
        if self.command:
            result = await environment.exec(command=self.command, cwd="/app")
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            (self.logs_dir / "visible.txt").write_text(
                (result.stdout or "") + (result.stderr or "")
            )
        for source in self.sources:
            target = submission_path(self.source_dir, source)
            is_dir = await environment.exec(command=f"test -d {shlex.quote(source)}")
            exists = await environment.exec(command=f"test -e {shlex.quote(source)}")
            if exists.return_code:
                if self.bootstrap:
                    continue
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
                downloaded = Path(tmp) / target.name
                if is_dir.return_code == 0:
                    await environment.download_dir(source_dir=source, target_dir=downloaded)
                else:
                    await environment.download_file(source_path=source, target_path=downloaded)
                if self.bootstrap:
                    # Image builds can add files inside an existing methods/
                    # directory. Merge missing files without replacing local edits.
                    for entry in (
                        [downloaded, *downloaded.rglob("*")]
                        if downloaded.is_dir()
                        else [downloaded]
                    ):
                        destination = target / entry.relative_to(downloaded)
                        if entry.is_symlink():
                            raise ValueError("Image baseline contains a symlink")
                        if entry.is_dir():
                            destination.mkdir(parents=True, exist_ok=True)
                        elif not destination.exists():
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(entry, destination)
                    continue
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
                shutil.move(str(downloaded), target)
        if result is not None and result.return_code:
            raise RuntimeError(f"Visible command exited with status {result.return_code}")


def visible_main() -> None:
    parser = argparse.ArgumentParser(description="Run inside the public RSI-Exam environment")
    parser.add_argument("mode", choices=["bootstrap", "run"])
    parser.add_argument(
        "--allow-host",
        action="append",
        default=[],
        help="Task-side model API host allowed during a visible experiment (repeatable)",
    )
    parser.add_argument(
        "command", nargs="?", help="Shell command inside /app (quote as one argument)"
    )
    args = parser.parse_args()
    if args.mode == "run" and not args.command:
        parser.error("run requires a command, e.g. 'python /app/selfcheck.py'")
    root = Path(__file__).resolve().parent
    config = tomllib.loads((root / "harbor/task.toml").read_text())
    job = "visible-" + uuid.uuid4().hex[:12]
    command = [
        sys.executable,
        "-m",
        "harbor.cli.main",
        "run",
        "--path",
        str(root / "harbor"),
        "--agent-import-path",
        "rsi_runtime:VisibleAgent",
        "--disable-verification",
        "--n-attempts",
        "1",
        "--n-concurrent",
        "1",
        "--max-retries",
        "0",
        "--jobs-dir",
        str(root / ".rsi_runs"),
        "--job-name",
        job,
        "--agent-kwarg",
        "source_dir=" + json.dumps(str(root)),
        "--agent-kwarg",
        "sources=" + json.dumps(artifact_sources(config)),
        "--agent-kwarg",
        "bootstrap=" + json.dumps(args.mode == "bootstrap"),
        "--agent-kwarg",
        "command=" + json.dumps(args.command or ""),
    ]
    for host in args.allow_host:
        command.extend(["--allow-agent-host", host])
    completed = subprocess.run(command, cwd=root)
    job_dir = root / ".rsi_runs" / job
    for log in job_dir.glob("*/agent/visible.txt"):
        print(log.read_text())
    results = list(job_dir.glob("*/result.json"))
    failed = len(results) != 1 or any(
        json.loads(p.read_text()).get("exception_info") for p in results
    )
    raise SystemExit(completed.returncode or int(failed))


if __name__ == "__main__":
    visible_main()
