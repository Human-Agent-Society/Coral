"""Tests for task-directory validation (coral validate / coral start)."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from coral.cli.author import cmd_validate
from coral.cli.validation import validate_task
from coral.task import validation as task_validation
from coral.task.validation import ValidationDiagnostic, ValidationReport
from coral.types import Score, ScoreBundle

_TASK_YAML = """\
task:
  name: t
  description: d
grader:
{grader_body}
agents:
  count: 1
"""


def _make_task(base: Path, grader_body: str) -> Path:
    task_dir = base / "task"
    task_dir.mkdir()
    (task_dir / "task.yaml").write_text(_TASK_YAML.format(grader_body=grader_body))
    return task_dir


def test_validate_accepts_entrypoint():
    with tempfile.TemporaryDirectory() as d:
        task_dir = _make_task(Path(d), '  entrypoint: "my_pkg.grader:Grader"')
        assert validate_task(task_dir) == []


def test_validate_rejects_missing_entrypoint():
    with tempfile.TemporaryDirectory() as d:
        task_dir = _make_task(Path(d), "  timeout: 60")
        errors = validate_task(task_dir)
        assert any("No grader configured" in e for e in errors)


def test_structured_validation_report_is_serializable():
    with tempfile.TemporaryDirectory() as d:
        task_dir = _make_task(Path(d), "  timeout: 60")

        report = task_validation.validate_task(task_dir)

        assert not report.valid
        assert report.error_messages == validate_task(task_dir)
        assert report.to_dict() == {
            "task_dir": str(task_dir),
            "valid": False,
            "diagnostics": [
                {
                    "code": "grader.entrypoint.missing",
                    "message": report.error_messages[0],
                    "path": "task.yaml",
                    "severity": "error",
                }
            ],
        }


def test_warning_diagnostic_does_not_fail_report():
    report = ValidationReport(
        task_dir=Path("task"),
        diagnostics=(
            ValidationDiagnostic(
                code="task.example.warning",
                message="Example warning",
                severity="warning",
            ),
        ),
    )

    assert report.valid
    assert report.error_messages == []


def test_validate_rejects_malformed_entrypoint():
    with tempfile.TemporaryDirectory() as d:
        task_dir = _make_task(Path(d), "  entrypoint: my_pkg.grader.Grader")
        errors = validate_task(task_dir)
        assert any("module.path:ClassName" in e for e in errors)


def _make_task_with_dirs(base: Path, grader_body: str, dirs: list[str]) -> Path:
    task_dir = base / "task"
    task_dir.mkdir()
    (task_dir / "task.yaml").write_text(_TASK_YAML.format(grader_body=grader_body))
    (task_dir / "grader").mkdir()
    for rel in dirs:
        (task_dir / rel).mkdir(parents=True, exist_ok=True)
    return task_dir


def test_structured_validation_reports_private_path():
    body = '  entrypoint: "p.g:G"\n  private:\n    - "missing-data"'
    with tempfile.TemporaryDirectory() as d:
        task_dir = _make_task_with_dirs(Path(d), body, [])

        report = task_validation.validate_task(task_dir)

        assert [diagnostic.code for diagnostic in report.diagnostics] == ["grader.private.missing"]
        assert report.diagnostics[0].path == "missing-data"


def test_validate_accepts_private_sibling_of_grader():
    """The common, safe layout: hidden data beside grader/ (e.g. taskdata/)."""
    body = '  entrypoint: "p.g:G"\n  private:\n    - "taskdata"'
    with tempfile.TemporaryDirectory() as d:
        task_dir = _make_task_with_dirs(Path(d), body, ["taskdata"])
        assert validate_task(task_dir) == []


def test_validate_rejects_private_inside_grader_package():
    """A grader.private path inside grader/ would be surfaced to agents via
    <shared_dir>/grader/ — validate must flag it as a leak."""
    body = '  entrypoint: "p.g:G"\n  private:\n    - "grader/taskdata"'
    with tempfile.TemporaryDirectory() as d:
        task_dir = _make_task_with_dirs(Path(d), body, ["grader/taskdata"])
        errors = validate_task(task_dir)
        assert any("inside the grader package" in e for e in errors)
        assert any("grader/taskdata" in e for e in errors)


def test_run_validation_returns_baseline_and_progress_events(tmp_path, monkeypatch):
    task_dir = _make_task(tmp_path, '  entrypoint: "p.g:G"')
    seed_dir = task_dir / "seed"
    seed_dir.mkdir()
    (seed_dir / "solution.txt").write_text("baseline", encoding="utf-8")

    def fake_setup_grader_env(coral_dir, grader_config, config_dir):
        assert (coral_dir / "private").is_dir()
        assert grader_config.entrypoint == "p.g:G"
        assert config_dir == task_dir

    class FakeGrader:
        async def grade(self, codebase_path, tasks):
            assert (Path(codebase_path) / "solution.txt").read_text(encoding="utf-8") == "baseline"
            assert [(task.id, task.name, task.description) for task in tasks] == [("t", "t", "d")]
            return ScoreBundle(
                scores={"eval": Score(value=1.25, name="eval", explanation="baseline ok")},
                aggregated=1.25,
            )

    monkeypatch.setattr(
        "coral.workspace.grader_env.setup_grader_env",
        fake_setup_grader_env,
    )
    monkeypatch.setattr(
        "coral.grader.loader.load_grader",
        lambda config, coral_dir: FakeGrader(),
    )

    observed_events = []
    result = task_validation.run_validation(task_dir, on_event=observed_events.append)

    assert result.successful
    assert result.report.valid
    assert result.baseline.to_dict() == {
        "scores": {
            "eval": {
                "value": 1.25,
                "name": "eval",
                "explanation": "baseline ok",
                "metadata": {},
            }
        },
        "aggregated": 1.25,
        "is_public": True,
    }
    assert result.failure is None
    assert observed_events == list(result.events)
    assert [(event.stage, event.status) for event in result.events] == [
        ("structure", "started"),
        ("structure", "completed"),
        ("workspace", "started"),
        ("workspace", "completed"),
        ("grader_environment", "started"),
        ("grader_environment", "completed"),
        ("grader_load", "started"),
        ("grader_load", "completed"),
        ("baseline", "started"),
        ("baseline", "completed"),
    ]


@pytest.mark.parametrize(
    "seed_contents",
    [
        pytest.param(None, id="no-seed-dir"),
        pytest.param([], id="empty-seed-dir"),
        pytest.param(["__pycache__"], id="pycache-only-seed-dir"),
    ],
)
def test_run_validation_reports_empty_workspace_consistently(tmp_path, monkeypatch, seed_contents):
    """When nothing lands in the workspace, both the workspace warning and the
    baseline-target message must say so — previously a seed/ dir that existed
    but contributed nothing produced the 'No seed/' warning and then claimed
    to run the grader 'against seed code'."""
    task_dir = _make_task(tmp_path, '  entrypoint: "p.g:G"')
    if seed_contents is not None:
        seed_dir = task_dir / "seed"
        seed_dir.mkdir()
        for name in seed_contents:
            (seed_dir / name).mkdir()

    class FakeGrader:
        async def grade(self, codebase_path, tasks):
            assert list(Path(codebase_path).iterdir()) == []
            return ScoreBundle(
                scores={"eval": Score(value=0.0, name="eval", explanation="empty")},
                aggregated=0.0,
            )

    monkeypatch.setattr(
        "coral.workspace.grader_env.setup_grader_env",
        lambda coral_dir, grader_config, config_dir: None,
    )
    monkeypatch.setattr(
        "coral.grader.loader.load_grader",
        lambda config, coral_dir: FakeGrader(),
    )

    result = task_validation.run_validation(task_dir)

    assert result.successful
    events = {(event.stage, event.status): event.message for event in result.events}
    assert "No seed/ directory" in events[("workspace", "completed")]
    assert events[("baseline", "started")] == "Running grader against empty workspace..."


async def test_run_validation_async_runs_inside_existing_event_loop(tmp_path, monkeypatch):
    task_dir = _make_task(tmp_path, '  entrypoint: "p.g:G"')

    class FakeGrader:
        async def grade(self, codebase_path, tasks):
            assert asyncio.get_running_loop().is_running()
            return ScoreBundle(
                scores={"eval": Score(value=2.0, name="eval", explanation="async baseline")},
                aggregated=2.0,
            )

    monkeypatch.setattr(
        "coral.workspace.grader_env.setup_grader_env",
        lambda coral_dir, grader_config, config_dir: None,
    )
    monkeypatch.setattr(
        "coral.grader.loader.load_grader",
        lambda config, coral_dir: FakeGrader(),
    )

    observed_events = []
    result = await task_validation.run_validation_async(
        task_dir,
        on_event=observed_events.append,
    )

    assert result.successful
    assert result.baseline.aggregated == 2.0
    assert observed_events == list(result.events)
    assert (result.events[-1].stage, result.events[-1].status) == ("baseline", "completed")


def test_run_validation_returns_structured_grader_environment_failure(tmp_path, monkeypatch):
    task_dir = _make_task(tmp_path, '  entrypoint: "p.g:G"')

    def fail_setup(coral_dir, grader_config, config_dir):
        raise RuntimeError("install failed")

    monkeypatch.setattr("coral.workspace.grader_env.setup_grader_env", fail_setup)
    monkeypatch.setattr(
        "coral.grader.loader.load_grader",
        lambda config, coral_dir: pytest.fail("grader loading must not run after setup fails"),
    )

    observed_events = []
    result = task_validation.run_validation(task_dir, on_event=observed_events.append)

    assert not result.successful
    assert result.baseline is None
    assert result.failure.stage == "grader_environment"
    assert result.failure.code == "grader.environment.failed"
    assert result.failure.message == "Failed to set up grader environment: install failed"
    assert observed_events[-1] == result.events[-1]
    assert (result.events[-1].stage, result.events[-1].status) == (
        "grader_environment",
        "failed",
    )


def test_run_validation_returns_structured_baseline_failure(tmp_path, monkeypatch):
    task_dir = _make_task(tmp_path, '  entrypoint: "p.g:G"')

    class FailingGrader:
        async def grade(self, codebase_path, tasks):
            raise ValueError("invalid baseline output")

    monkeypatch.setattr(
        "coral.workspace.grader_env.setup_grader_env",
        lambda coral_dir, grader_config, config_dir: None,
    )
    monkeypatch.setattr(
        "coral.grader.loader.load_grader",
        lambda config, coral_dir: FailingGrader(),
    )

    result = task_validation.run_validation(task_dir)

    assert not result.successful
    assert result.baseline is None
    assert result.failure.stage == "baseline"
    assert result.failure.code == "grader.baseline.failed"
    assert result.failure.message == "Grader crashed: invalid baseline output"
    assert (result.events[-1].stage, result.events[-1].status) == ("baseline", "failed")


def test_run_validation_returns_structured_grader_load_failure(tmp_path, monkeypatch):
    task_dir = _make_task(tmp_path, '  entrypoint: "p.g:G"')

    def fail_load(config, coral_dir):
        raise ImportError("grader package missing")

    monkeypatch.setattr(
        "coral.workspace.grader_env.setup_grader_env",
        lambda coral_dir, grader_config, config_dir: None,
    )
    monkeypatch.setattr("coral.grader.loader.load_grader", fail_load)

    result = task_validation.run_validation(task_dir)

    assert not result.successful
    assert result.failure.stage == "grader_load"
    assert result.failure.code == "grader.load.failed"
    assert result.failure.message == "Error loading grader: grader package missing"
    assert (result.events[-1].stage, result.events[-1].status) == ("grader_load", "failed")


def test_validation_run_result_is_serializable():
    report = ValidationReport(task_dir=Path("task"), diagnostics=())
    event = task_validation.ValidationProgressEvent(
        stage="grader_load",
        status="failed",
        message="Error loading grader: missing",
    )
    failure = task_validation.ValidationFailure(
        stage="grader_load",
        code="grader.load.failed",
        message="Error loading grader: missing",
    )
    result = task_validation.ValidationRunResult(
        report=report,
        events=(event,),
        failure=failure,
    )

    assert result.to_dict() == {
        "successful": False,
        "report": {
            "task_dir": "task",
            "valid": True,
            "diagnostics": [],
        },
        "events": [
            {
                "stage": "grader_load",
                "status": "failed",
                "message": "Error loading grader: missing",
            }
        ],
        "baseline": None,
        "failure": {
            "stage": "grader_load",
            "code": "grader.load.failed",
            "message": "Error loading grader: missing",
        },
    }


def test_run_validation_returns_structured_workspace_failure(tmp_path, monkeypatch):
    task_dir = _make_task(tmp_path, '  entrypoint: "p.g:G"')
    seed_dir = task_dir / "seed"
    seed_dir.mkdir()
    (seed_dir / "solution.txt").write_text("baseline", encoding="utf-8")

    def fail_copy(source, destination):
        raise OSError("disk full")

    monkeypatch.setattr(task_validation.shutil, "copy2", fail_copy)
    monkeypatch.setattr(
        "coral.workspace.grader_env.setup_grader_env",
        lambda coral_dir, grader_config, config_dir: pytest.fail(
            "grader setup must not run after workspace preparation fails"
        ),
    )

    result = task_validation.run_validation(task_dir)

    assert not result.successful
    assert result.failure.stage == "workspace"
    assert result.failure.code == "task.workspace.failed"
    assert result.failure.message == "Failed to prepare validation workspace: disk full"
    assert (result.events[-1].stage, result.events[-1].status) == ("workspace", "failed")


def test_run_validation_structures_temp_workspace_failure(tmp_path, monkeypatch):
    task_dir = _make_task(tmp_path, '  entrypoint: "p.g:G"')

    def fail_tempdir(*args, **kwargs):
        raise OSError("no temporary space")

    monkeypatch.setattr(task_validation.tempfile, "TemporaryDirectory", fail_tempdir)

    result = task_validation.run_validation(task_dir)

    assert not result.successful
    assert result.failure.stage == "workspace"
    assert result.failure.code == "task.workspace.failed"
    assert result.failure.message == ("Failed to prepare validation workspace: no temporary space")
    assert [(event.stage, event.status) for event in result.events[-2:]] == [
        ("workspace", "started"),
        ("workspace", "failed"),
    ]


def test_task_package_exports_validation_runner_api():
    import coral.task as task_api

    assert task_api.run_validation is task_validation.run_validation
    assert task_api.run_validation_async is task_validation.run_validation_async
    assert task_api.ValidationProgressEvent is task_validation.ValidationProgressEvent
    assert task_api.ValidationFailure is task_validation.ValidationFailure
    assert task_api.ValidationRunResult is task_validation.ValidationRunResult


def test_run_validation_stops_after_structural_failure(tmp_path, monkeypatch):
    task_dir = _make_task(tmp_path, "  timeout: 60")
    monkeypatch.setattr(
        "coral.workspace.grader_env.setup_grader_env",
        lambda coral_dir, grader_config, config_dir: pytest.fail(
            "grader setup must not run when structural validation fails"
        ),
    )

    result = task_validation.run_validation(task_dir)

    assert not result.successful
    assert not result.report.valid
    assert result.report.diagnostics[0].code == "grader.entrypoint.missing"
    assert result.failure.stage == "structure"
    assert result.failure.code == "task.structure.invalid"
    assert [(event.stage, event.status) for event in result.events] == [
        ("structure", "started"),
        ("structure", "failed"),
    ]


def test_run_validation_structures_config_reload_failure(tmp_path, monkeypatch):
    task_dir = tmp_path / "task"
    report = ValidationReport(task_dir=task_dir, diagnostics=())

    class FailingConfig:
        @classmethod
        def from_yaml(cls, path):
            raise OSError("changed during validation")

    monkeypatch.setattr(task_validation, "validate_task", lambda path: report)
    monkeypatch.setattr(task_validation, "CoralConfig", FailingConfig)

    result = task_validation.run_validation(task_dir)

    assert not result.successful
    assert result.failure.stage == "structure"
    assert result.failure.code == "task.config.load_failed"
    assert result.failure.message == (
        "Failed to load task configuration: changed during validation"
    )
    assert [(event.stage, event.status) for event in result.events] == [
        ("structure", "started"),
        ("structure", "failed"),
    ]


def test_run_validation_structures_unexpected_static_check_failure(tmp_path, monkeypatch):
    task_dir = tmp_path / "task"

    def fail_validation(path):
        raise OSError("permission denied")

    monkeypatch.setattr(task_validation, "validate_task", fail_validation)

    result = task_validation.run_validation(task_dir)

    assert not result.successful
    assert result.report.diagnostics[0].code == "task.structure.failed"
    assert result.report.diagnostics[0].message == (
        "Failed to validate task structure: permission denied"
    )
    assert result.failure.stage == "structure"
    assert result.failure.code == "task.structure.failed"
    assert [(event.stage, event.status) for event in result.events] == [
        ("structure", "started"),
        ("structure", "failed"),
    ]


def test_cmd_validate_renders_shared_runner_result(tmp_path, monkeypatch, capsys):
    task_dir = tmp_path / "task"
    report = ValidationReport(task_dir=task_dir.resolve(), diagnostics=())
    events = (
        task_validation.ValidationProgressEvent(
            stage="structure",
            status="completed",
            message="Task structure is valid",
        ),
        task_validation.ValidationProgressEvent(
            stage="workspace",
            status="completed",
            message="Seed: copied seed/ into workspace",
        ),
        task_validation.ValidationProgressEvent(
            stage="grader_environment",
            status="started",
            message="Setting up grader venv (.coral/private/grader_venv)...",
        ),
        task_validation.ValidationProgressEvent(
            stage="baseline",
            status="started",
            message="Running grader against seed code...",
        ),
    )
    baseline = ScoreBundle(
        scores={"eval": Score(value=1.25, name="eval", explanation="baseline ok")},
        aggregated=1.25,
    )

    def fake_run_validation(path, *, on_event=None):
        assert path == task_dir.resolve()
        for event in events:
            on_event(event)
        return task_validation.ValidationRunResult(
            report=report,
            events=events,
            baseline=baseline,
        )

    monkeypatch.setattr(task_validation, "run_validation", fake_run_validation)

    cmd_validate(argparse.Namespace(path=str(task_dir)))

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.splitlines() == [
        "Validation: OK",
        "Seed: copied seed/ into workspace",
        "Setting up grader venv (.coral/private/grader_venv)...",
        "",
        "Running grader against seed code...",
        "",
        "==================================================",
        "Score: 1.25",
        "  eval: baseline ok",
        "==================================================",
    ]


def test_validate_help_documents_json_output() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "coral.cli", "validate", "--help"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert "--json" in result.stdout


def test_cmd_validate_json_outputs_one_machine_readable_document(tmp_path, monkeypatch, capsys):
    task_dir = tmp_path / "task"
    result = task_validation.ValidationRunResult(
        report=ValidationReport(task_dir=task_dir.resolve(), diagnostics=()),
        events=(),
        baseline=ScoreBundle(
            scores={"eval": Score(value=1.25, name="eval", explanation="baseline ok")},
            aggregated=1.25,
        ),
    )

    def fake_run_validation(path, *, on_event=None):
        assert path == task_dir.resolve()
        assert on_event is None
        return result

    monkeypatch.setattr(task_validation, "run_validation", fake_run_validation)

    cmd_validate(argparse.Namespace(path=str(task_dir), json=True))

    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["successful"] is True
    assert payload["report"] == {
        "task_dir": str(task_dir.resolve()),
        "valid": True,
        "diagnostics": [],
    }
    assert payload["events"] == []
    assert payload["baseline"]["aggregated"] == 1.25
    assert payload["failure"] is None


def test_cmd_validate_json_failure_is_structured_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    task_dir = tmp_path / "task"
    failure = task_validation.ValidationFailure(
        stage="baseline",
        code="grader.baseline.failed",
        message="Grader crashed: boom",
    )
    result = task_validation.ValidationRunResult(
        report=ValidationReport(task_dir=task_dir.resolve(), diagnostics=()),
        events=(),
        failure=failure,
    )

    def fake_run_validation(path, *, on_event=None):
        assert path == task_dir.resolve()
        assert on_event is None
        return result

    monkeypatch.setattr(task_validation, "run_validation", fake_run_validation)

    with pytest.raises(SystemExit) as exc_info:
        cmd_validate(argparse.Namespace(path=str(task_dir), json=True))

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["successful"] is False
    assert payload["baseline"] is None
    assert payload["failure"] == {
        "stage": "baseline",
        "code": "grader.baseline.failed",
        "message": "Grader crashed: boom",
    }


def test_cmd_validate_prints_structure_failure_without_diagnostics(tmp_path, monkeypatch, capsys):
    task_dir = tmp_path / "task"
    failure = task_validation.ValidationFailure(
        stage="structure",
        code="task.config.load_failed",
        message="Failed to load task configuration: changed during validation",
    )
    result = task_validation.ValidationRunResult(
        report=ValidationReport(task_dir=task_dir.resolve(), diagnostics=()),
        events=(),
        failure=failure,
    )
    monkeypatch.setattr(
        task_validation,
        "run_validation",
        lambda path, on_event=None: result,
    )

    with pytest.raises(SystemExit) as exc_info:
        cmd_validate(argparse.Namespace(path=str(task_dir)))

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Failed to load task configuration: changed during validation\n"


def test_cmd_validate_preserves_static_validation_error_output(tmp_path, capsys):
    task_dir = _make_task(tmp_path, "  timeout: 60")

    with pytest.raises(SystemExit) as exc_info:
        cmd_validate(argparse.Namespace(path=str(task_dir)))

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("Validation errors:\n  - No grader configured.")
