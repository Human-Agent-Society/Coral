"""Pinned Harbor v0.22.0 trial runner for the CORAL local-task adapter.

This file is executed by ``uv run --isolated`` under Python 3.12, so importing
it from CORAL's normal Python environment is neither required nor supported.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import shlex
from pathlib import Path, PurePosixPath
from typing import override

from harbor import (
    AgentContext,
    BaseAgent,
    BaseEnvironment,
    EnvironmentType,
    Trial,
    TrialAgentConfig,
    TrialConfig,
    TrialEnvironmentConfig,
    TrialTaskConfig,
)
from harbor import (
    Task as HarborTask,
)

HARBOR_RUNTIME_VERSION = "0.22.0"
HARBOR_SCHEMA_VERSION = "1.4"
_PROTECTED_WORKDIRS = {
    PurePosixPath("/"),
    PurePosixPath("/logs"),
    PurePosixPath("/tests"),
    PurePosixPath("/solution"),
}


class CandidateWorkspaceAgent(BaseAgent):
    """Transfer a CORAL candidate into Harbor's agent workdir without optimizing it."""

    SUPPORTS_WINDOWS = False

    def __init__(self, *args: object, codebase_path: str, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._codebase_path = Path(codebase_path).resolve()

    @staticmethod
    @override
    def name() -> str:
        return "coral_candidate_workspace"

    @override
    def version(self) -> str:
        return "1"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        del instruction, context
        result = await environment.exec("pwd", user="root")
        if result.return_code != 0:
            raise RuntimeError(result.stderr or "Harbor environment could not report its workdir")
        workdir_text = (result.stdout or "").strip()
        workdir = PurePosixPath(workdir_text)
        protected = workdir == PurePosixPath("/") or any(
            workdir == protected_dir or protected_dir in workdir.parents
            for protected_dir in _PROTECTED_WORKDIRS
            if protected_dir != PurePosixPath("/")
        )
        if not workdir.is_absolute() or protected:
            raise ValueError(
                "Harbor task must use a dedicated absolute workdir other than "
                f"/, /logs, /tests, or /solution; got {workdir_text!r}"
            )

        quoted = shlex.quote(workdir.as_posix())
        prepare = await environment.exec(
            f"mkdir -p {quoted}",
            cwd="/",
            user="root",
        )
        if prepare.return_code != 0:
            raise RuntimeError(prepare.stderr or f"Could not prepare Harbor workdir {workdir}")
        contents = await environment.exec(
            f"find {quoted} -mindepth 1 -maxdepth 1 -print -quit",
            cwd="/",
            user="root",
        )
        if contents.return_code != 0:
            raise RuntimeError(contents.stderr or f"Could not inspect Harbor workdir {workdir}")
        if (contents.stdout or "").strip():
            raise ValueError(
                "Initial CORAL Harbor compatibility requires a dedicated empty task workdir; "
                f"{workdir} already contains files"
            )
        await environment.upload_dir(self._codebase_path, workdir.as_posix())


async def _run(request: dict[str, object]) -> dict[str, object]:
    runtime_version = importlib.metadata.version("harbor")
    if runtime_version != HARBOR_RUNTIME_VERSION:
        raise RuntimeError(f"Expected Harbor {HARBOR_RUNTIME_VERSION}, running {runtime_version}")

    task_path = Path(str(request["task_path"])).resolve()
    candidate_path = Path(str(request["candidate_path"])).resolve()
    trials_dir = Path(str(request["trials_dir"])).resolve()
    trial_name = str(request["trial_name"])

    task = HarborTask(task_path)
    if task.config.schema_version != HARBOR_SCHEMA_VERSION:
        raise ValueError(
            f"Expected Harbor task schema {HARBOR_SCHEMA_VERSION}, got {task.config.schema_version}"
        )
    if task.has_steps:
        raise ValueError("Initial CORAL Harbor compatibility supports single-step tasks only")
    if task.config.environment.os.value != "linux":
        raise ValueError("Initial CORAL Harbor compatibility supports Linux tasks only")

    config = TrialConfig(
        task=TrialTaskConfig(path=task_path),
        agent=TrialAgentConfig(
            import_path="__main__:CandidateWorkspaceAgent",
            kwargs={"codebase_path": str(candidate_path)},
        ),
        environment=TrialEnvironmentConfig(
            type=EnvironmentType.DOCKER,
            force_build=False,
            delete=True,
        ),
        trials_dir=trials_dir,
        trial_name=trial_name,
    )
    trial = await Trial.create(config=config)
    result = await trial.run()

    if result.exception_info is not None:
        raise RuntimeError(
            f"{result.exception_info.exception_type}: {result.exception_info.exception_message}"
        )
    if result.verifier_result is None or result.verifier_result.rewards is None:
        raise RuntimeError("Harbor verifier returned no rewards")

    return {
        "runtime_version": runtime_version,
        "schema_version": task.config.schema_version,
        "task_name": task.name,
        "task_package_version": (
            task.config.task.version if task.config.task is not None else None
        ),
        "task_digest": result.task_checksum,
        "rewards": result.verifier_result.rewards,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        payload = asyncio.run(_run(request))
        print(json.dumps({"result": payload}, separators=(",", ":")))
    except Exception as exc:  # noqa: BLE001 - serialized across a subprocess boundary
        print(
            json.dumps(
                {"error": f"{type(exc).__name__}: {exc}"},
                separators=(",", ":"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
