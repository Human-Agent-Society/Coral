"""Structured validation for CORAL task directories."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from coral.config import CoralConfig
from coral.types import ScoreBundle, Task


@dataclass(frozen=True)
class ValidationDiagnostic:
    """A machine-readable problem found in a task directory."""

    code: str
    message: str
    path: str | None = None
    severity: Literal["error", "warning"] = "error"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.path is not None:
            data["path"] = self.path
        return data


@dataclass(frozen=True)
class ValidationReport:
    """Structured result of validating one task directory."""

    task_dir: Path
    diagnostics: tuple[ValidationDiagnostic, ...]

    @property
    def valid(self) -> bool:
        return all(diagnostic.severity != "error" for diagnostic in self.diagnostics)

    @property
    def error_messages(self) -> list[str]:
        """Return the legacy error representation used by the CLI."""
        return [
            diagnostic.message for diagnostic in self.diagnostics if diagnostic.severity == "error"
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_dir": str(self.task_dir),
            "valid": self.valid,
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
        }


ValidationStage = Literal[
    "structure",
    "workspace",
    "grader_environment",
    "grader_load",
    "baseline",
]
ValidationProgressStatus = Literal["started", "completed", "failed"]


@dataclass(frozen=True)
class ValidationProgressEvent:
    """One frontend-neutral progress update from a validation run."""

    stage: ValidationStage
    status: ValidationProgressStatus
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "stage": self.stage,
            "status": self.status,
            "message": self.message,
        }


@dataclass(frozen=True)
class ValidationFailure:
    """A structured failure that stopped a validation run."""

    stage: ValidationStage
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class ValidationRunResult:
    """Structural diagnostics, progress, and baseline output from one run."""

    report: ValidationReport
    events: tuple[ValidationProgressEvent, ...]
    baseline: ScoreBundle | None = None
    failure: ValidationFailure | None = None

    @property
    def successful(self) -> bool:
        return self.report.valid and self.failure is None and self.baseline is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "successful": self.successful,
            "report": self.report.to_dict(),
            "events": [event.to_dict() for event in self.events],
            "baseline": self.baseline.to_dict() if self.baseline is not None else None,
            "failure": self.failure.to_dict() if self.failure is not None else None,
        }


def validate_task(task_dir: Path) -> ValidationReport:
    """Validate a task directory and return structured diagnostics."""
    diagnostics: list[ValidationDiagnostic] = []

    task_yaml = task_dir / "task.yaml"
    if not task_yaml.exists():
        diagnostics.append(
            ValidationDiagnostic(
                code="task.config.missing",
                message=f"task.yaml not found in {task_dir}",
                path="task.yaml",
            )
        )
        return ValidationReport(task_dir, tuple(diagnostics))

    try:
        config = CoralConfig.from_yaml(task_yaml)
    except Exception as exc:
        diagnostics.append(
            ValidationDiagnostic(
                code="task.config.invalid",
                message=f"task.yaml parse error: {exc}",
                path="task.yaml",
            )
        )
        return ValidationReport(task_dir, tuple(diagnostics))

    if not config.grader.entrypoint:
        diagnostics.append(
            ValidationDiagnostic(
                code="grader.entrypoint.missing",
                message=(
                    "No grader configured. Set grader.entrypoint = "
                    "'your_pkg.module:Grader' in task.yaml and grader.setup to "
                    "install the package."
                ),
                path="task.yaml",
            )
        )
    elif ":" not in config.grader.entrypoint:
        diagnostics.append(
            ValidationDiagnostic(
                code="grader.entrypoint.invalid",
                message=(
                    "grader.entrypoint must be 'module.path:ClassName', "
                    f"got {config.grader.entrypoint!r}"
                ),
                path="task.yaml",
            )
        )

    if config.grader.direction not in ("maximize", "minimize"):
        diagnostics.append(
            ValidationDiagnostic(
                code="grader.direction.invalid",
                message=(
                    "grader.direction must be 'maximize' or 'minimize', "
                    f"got '{config.grader.direction}'"
                ),
                path="task.yaml",
            )
        )

    # The grader package is surfaced read-only to agents at <shared_dir>/grader/.
    # Private paths inside it would therefore be copied into .coral/private/ and
    # exposed through the surfaced source at the same time.
    grader_dir = (task_dir / "grader").resolve()
    for private_path in config.grader.private:
        path = Path(private_path)
        if not path.is_absolute():
            path = task_dir / path
        if not path.exists():
            diagnostics.append(
                ValidationDiagnostic(
                    code="grader.private.missing",
                    message=f"Private file not found: {private_path}",
                    path=str(private_path),
                )
            )
            continue
        try:
            path.resolve().relative_to(grader_dir)
        except ValueError:
            pass
        else:
            diagnostics.append(
                ValidationDiagnostic(
                    code="grader.private.exposed",
                    message=(
                        f"grader.private path '{private_path}' is inside the grader package "
                        "(grader/), which is surfaced read-only to agents at "
                        "<shared_dir>/grader/ — this would leak it. Move it outside grader/ "
                        "(e.g. a sibling 'taskdata/')."
                    ),
                    path=str(private_path),
                )
            )

    return ValidationReport(task_dir, tuple(diagnostics))


async def run_validation_async(
    task_dir: Path,
    *,
    on_event: Callable[[ValidationProgressEvent], None] | None = None,
) -> ValidationRunResult:
    """Run structural checks and asynchronously grade a task baseline."""
    events: list[ValidationProgressEvent] = []

    def emit(stage: ValidationStage, status: ValidationProgressStatus, message: str) -> None:
        event = ValidationProgressEvent(stage=stage, status=status, message=message)
        events.append(event)
        if on_event is not None:
            on_event(event)

    emit("structure", "started", "Checking task structure")
    try:
        report = validate_task(task_dir)
    except Exception as exc:
        message = f"Failed to validate task structure: {exc}"
        report = ValidationReport(
            task_dir=task_dir,
            diagnostics=(
                ValidationDiagnostic(
                    code="task.structure.failed",
                    message=message,
                ),
            ),
        )
        failure = ValidationFailure(
            stage="structure",
            code="task.structure.failed",
            message=message,
        )
        emit(failure.stage, "failed", failure.message)
        return ValidationRunResult(report=report, events=tuple(events), failure=failure)

    def stop(stage: ValidationStage, code: str, message: str) -> ValidationRunResult:
        failure = ValidationFailure(stage=stage, code=code, message=message)
        emit(stage, "failed", message)
        return ValidationRunResult(report=report, events=tuple(events), failure=failure)

    if not report.valid:
        return stop(
            "structure",
            "task.structure.invalid",
            "Task structure validation failed",
        )
    try:
        config = CoralConfig.from_yaml(task_dir / "task.yaml")
    except Exception as exc:
        return stop(
            "structure",
            "task.config.load_failed",
            f"Failed to load task configuration: {exc}",
        )
    emit("structure", "completed", "Task structure is valid")

    emit("workspace", "started", "Preparing validation workspace")
    try:
        tempdir = tempfile.TemporaryDirectory(prefix="coral_test_eval_")
    except Exception as exc:
        return stop(
            "workspace",
            "task.workspace.failed",
            f"Failed to prepare validation workspace: {exc}",
        )

    with tempdir as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        workspace = tmpdir / "workspace"
        try:
            workspace.mkdir()
            seed_dir = task_dir / "seed"
            # Mirror the copy loop below: a seed/ that exists but contributes
            # nothing to the workspace (empty, or only __pycache__) is treated
            # as absent so the progress messages describe what actually runs.
            has_seed = seed_dir.is_dir() and any(
                item.name != "__pycache__" for item in seed_dir.iterdir()
            )
            if has_seed:
                for item in seed_dir.iterdir():
                    if item.name == "__pycache__":
                        continue
                    dst = workspace / item.name
                    if item.is_dir():
                        shutil.copytree(item, dst)
                    else:
                        shutil.copy2(item, dst)
                workspace_message = f"Seed: copied {seed_dir.name}/ into workspace"
            else:
                workspace_message = (
                    "Warning: No seed/ directory — grader will run against an empty workspace.\n"
                    "  This is fine if your task expects agents to build from scratch."
                )

            coral_dir = tmpdir / ".coral"
            private_dir = coral_dir / "private"
            private_dir.mkdir(parents=True)

            for private_path_str in config.grader.private:
                src = Path(private_path_str)
                if not src.is_absolute():
                    src = (task_dir / src).resolve()
                if src.exists():
                    dst = private_dir / src.name
                    if src.is_dir():
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
        except Exception as exc:
            return stop(
                "workspace",
                "task.workspace.failed",
                f"Failed to prepare validation workspace: {exc}",
            )
        emit("workspace", "completed", workspace_message)

        from coral.workspace.grader_env import setup_grader_env

        emit(
            "grader_environment",
            "started",
            "Setting up grader venv (.coral/private/grader_venv)...",
        )
        try:
            setup_grader_env(coral_dir, config.grader, task_dir)
        except Exception as exc:
            return stop(
                "grader_environment",
                "grader.environment.failed",
                f"Failed to set up grader environment: {exc}",
            )
        emit("grader_environment", "completed", "Grader environment is ready")

        from coral.grader.loader import load_grader

        emit("grader_load", "started", "Loading grader entrypoint")
        try:
            grader = load_grader(config, coral_dir)
        except Exception as exc:
            return stop("grader_load", "grader.load.failed", f"Error loading grader: {exc}")
        emit("grader_load", "completed", "Grader entrypoint loaded")

        task = Task(
            id=config.task.name,
            name=config.task.name,
            description=config.task.description,
        )
        target = "seed code" if has_seed else "empty workspace"
        emit("baseline", "started", f"Running grader against {target}...")
        try:
            baseline = await grader.grade(str(workspace), [task])
        except Exception as exc:
            return stop("baseline", "grader.baseline.failed", f"Grader crashed: {exc}")
        emit("baseline", "completed", "Baseline grading completed")

    return ValidationRunResult(report=report, events=tuple(events), baseline=baseline)


def run_validation(
    task_dir: Path,
    *,
    on_event: Callable[[ValidationProgressEvent], None] | None = None,
) -> ValidationRunResult:
    """Run validation synchronously for CLI and other non-async callers."""
    return asyncio.run(run_validation_async(task_dir, on_event=on_event))
