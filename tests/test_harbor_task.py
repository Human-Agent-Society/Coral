"""Tests for the initial local Harbor task compatibility profile."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path

import pytest

from coral.config import CoralConfig, GraderConfig, TaskConfig
from coral.grader.harbor import (
    HarborTaskGrader,
    _copy_candidate_workspace,
    _HarborRunnerTimeoutError,
    _run_harbor_runner,
    _score_bundle_from_result,
)
from coral.harbor_task import (
    HARBOR_ADAPTER_MARKER,
    HARBOR_GRADER_ENTRYPOINT,
    HARBOR_RUNTIME_VERSION,
    inspect_local_harbor_task,
    stage_local_harbor_task,
)
from coral.task.validation import run_validation, validate_task
from coral.types import Score, ScoreBundle, Task
from coral.workspace.project import create_project


def _write_harbor_task(root: Path, *, schema: str = "1.4", steps: bool = False) -> Path:
    task_dir = root / "harbor-task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "solution").mkdir()
    task_toml = [
        f'schema_version = "{schema}"',
        "",
        "[task]",
        'name = "example/hello"',
        'version = "1.0.0"',
        "",
        "[environment]",
    ]
    if steps:
        task_toml.extend(["", "[[steps]]", 'name = "first"'])
    (task_dir / "task.toml").write_text("\n".join(task_toml), encoding="utf-8")
    (task_dir / "instruction.md").write_text("Create hello.txt.\n", encoding="utf-8")
    (task_dir / "environment" / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nWORKDIR /app\n",
        encoding="utf-8",
    )
    (task_dir / "tests" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n",
        encoding="utf-8",
    )
    (task_dir / "solution" / "answer.txt").write_text("secret\n", encoding="utf-8")
    return task_dir


def _write_coral_config(root: Path, source: str = "./harbor-task") -> Path:
    config_path = root / "task.yaml"
    config_path.write_text(
        "\n".join(
            [
                "task:",
                f"  source: {source}",
                "  reward:",
                "    primary: reward",
                "    direction: maximize",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


def test_inspect_local_harbor_task_records_identity_and_digest(tmp_path: Path) -> None:
    task_dir = _write_harbor_task(tmp_path)

    descriptor = inspect_local_harbor_task(task_dir)

    assert descriptor.name == "example/hello"
    assert descriptor.instruction == "Create hello.txt."
    assert descriptor.schema_version == "1.4"
    assert descriptor.package_version == "1.0.0"
    assert descriptor.digest.startswith("sha256:")


def test_harbor_instruction_strips_leading_canary_like_harbor_runtime(
    tmp_path: Path,
) -> None:
    task_dir = _write_harbor_task(tmp_path)
    (task_dir / "instruction.md").write_text(
        "<!-- benchmark canary 123 -->\n# second CANARY marker\n\nCreate hello.txt.\n",
        encoding="utf-8",
    )

    descriptor = inspect_local_harbor_task(task_dir)

    assert descriptor.instruction == "Create hello.txt."


def test_harbor_source_hydrates_runtime_config_without_user_grader_fields(
    tmp_path: Path,
) -> None:
    _write_harbor_task(tmp_path)
    config_path = _write_coral_config(tmp_path)

    config = CoralConfig.from_yaml(config_path)

    assert config.task.name == "example/hello"
    assert config.task.description == "Create hello.txt."
    assert config.task.source == "./harbor-task"
    assert config.task.reward is not None
    assert config.task.reward.primary == "reward"
    assert config.grader.entrypoint == HARBOR_GRADER_ENTRYPOINT
    assert config.grader.direction == "maximize"
    assert config.grader.args["harbor_adapter"] == HARBOR_ADAPTER_MARKER
    assert config.grader.args["harbor_runtime_version"] == HARBOR_RUNTIME_VERSION
    assert config.grader.args["harbor_task_digest"].startswith("sha256:")


def test_harbor_config_hydration_does_not_mutate_input_mapping(tmp_path: Path) -> None:
    task_dir = _write_harbor_task(tmp_path)
    data = {
        "task": {
            "source": str(task_dir),
            "reward": {"primary": "reward", "direction": "maximize"},
        },
        "grader": {"timeout": 600},
    }
    original = json.loads(json.dumps(data))

    CoralConfig.from_dict(data)

    assert data == original


def test_legacy_task_serialization_is_unchanged() -> None:
    config = CoralConfig(task=TaskConfig(name="legacy", description="Keep existing behavior"))

    task_data = config.to_dict()["task"]

    assert "source" not in task_data
    assert "reward" not in task_data


def test_resolved_harbor_run_config_reloads_without_original_relative_source(
    tmp_path: Path,
) -> None:
    _write_harbor_task(tmp_path)
    config = CoralConfig.from_yaml(_write_coral_config(tmp_path))
    snapshot_dir = tmp_path / "run" / ".coral"
    snapshot_dir.mkdir(parents=True)
    snapshot = snapshot_dir / "config.yaml"
    config.to_yaml(snapshot)

    restored = CoralConfig.from_yaml(snapshot)

    assert restored.task.name == "example/hello"
    assert restored.task.source == "./harbor-task"
    assert restored.grader.args["harbor_task_digest"] == config.grader.args["harbor_task_digest"]


def test_harbor_reward_dotlist_override_keeps_internal_grader_in_sync(
    tmp_path: Path,
) -> None:
    _write_harbor_task(tmp_path)
    config = CoralConfig.from_yaml(_write_coral_config(tmp_path))

    merged = CoralConfig.merge_dotlist(
        config,
        ["task.reward.primary=quality", "task.reward.direction=minimize"],
    )

    assert merged.task.reward is not None
    assert merged.task.reward.primary == "quality"
    assert merged.grader.args["primary_reward"] == "quality"
    assert merged.grader.direction == "minimize"


def test_harbor_task_rejects_docker_in_docker_session(tmp_path: Path) -> None:
    _write_harbor_task(tmp_path)
    config_path = _write_coral_config(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "run:\n  session: docker\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported Docker-in-Docker"):
        CoralConfig.from_yaml(config_path)

    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "run:\n  session: docker\n",
            "run:\n  session: local\n",
        ),
        encoding="utf-8",
    )
    config = CoralConfig.from_yaml(config_path)
    with pytest.raises(ValueError, match="unsupported Docker-in-Docker"):
        CoralConfig.merge_dotlist(config, ["run.session=docker"])


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("  name: duplicate\n", "must not duplicate"),
        ("grader:\n  entrypoint: custom:Grader\n", "cannot define portable grader"),
        ("workspace:\n  repo_path: ./seed-repo\n", "creates an empty CORAL workspace"),
    ],
)
def test_harbor_source_rejects_mixed_legacy_ownership(
    tmp_path: Path,
    extra: str,
    message: str,
) -> None:
    _write_harbor_task(tmp_path)
    base = _write_coral_config(tmp_path).read_text(encoding="utf-8")
    (tmp_path / "task.yaml").write_text(base + extra, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        CoralConfig.from_yaml(tmp_path / "task.yaml")


def test_harbor_source_rejects_registry_and_unsupported_schema(tmp_path: Path) -> None:
    _write_coral_config(tmp_path, source="example/hello@1.0.0")
    with pytest.raises(ValueError, match="Registry Harbor task sources"):
        CoralConfig.from_yaml(tmp_path / "task.yaml")

    _write_harbor_task(tmp_path, schema="1.5")
    _write_coral_config(tmp_path)
    with pytest.raises(ValueError, match="requires task schema 1.4"):
        CoralConfig.from_yaml(tmp_path / "task.yaml")


def test_harbor_source_rejects_multi_step_task(tmp_path: Path) -> None:
    _write_harbor_task(tmp_path, steps=True)
    _write_coral_config(tmp_path)

    with pytest.raises(ValueError, match="single-step"):
        CoralConfig.from_yaml(tmp_path / "task.yaml")


def test_harbor_source_rejects_unpublished_declared_artifacts(tmp_path: Path) -> None:
    task_dir = _write_harbor_task(tmp_path)
    config_path = task_dir / "task.toml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            'schema_version = "1.4"\n',
            'schema_version = "1.4"\nartifacts = ["/tmp/result.json"]\n',
        ),
        encoding="utf-8",
    )
    _write_coral_config(tmp_path)

    with pytest.raises(ValueError, match="does not yet publish declared task artifacts"):
        CoralConfig.from_yaml(tmp_path / "task.yaml")


def test_stage_harbor_task_checks_digest_and_keeps_solution_private(tmp_path: Path) -> None:
    task_dir = _write_harbor_task(tmp_path)
    descriptor = inspect_local_harbor_task(task_dir)
    private_dir = tmp_path / "private"
    private_dir.mkdir()

    staged = stage_local_harbor_task(
        "./harbor-task",
        base_dir=tmp_path,
        private_dir=private_dir,
        expected_digest=descriptor.digest,
    )

    assert (staged / "solution" / "answer.txt").read_text(encoding="utf-8") == "secret\n"
    with pytest.raises(ValueError, match="changed after configuration"):
        other_private = tmp_path / "other-private"
        other_private.mkdir()
        stage_local_harbor_task(
            task_dir,
            base_dir=tmp_path,
            private_dir=other_private,
            expected_digest="sha256:wrong",
        )


def test_validate_rejects_seed_mixed_with_harbor_source(tmp_path: Path) -> None:
    _write_harbor_task(tmp_path)
    _write_coral_config(tmp_path)
    (tmp_path / "seed").mkdir()

    report = validate_task(tmp_path)

    assert not report.valid
    assert {diagnostic.code for diagnostic in report.diagnostics} == {"task.harbor.seed_mixed"}


def test_run_validation_stages_harbor_privately_and_reports_empty_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_harbor_task(tmp_path)
    _write_coral_config(tmp_path)

    class FakeGrader:
        async def grade(self, codebase_path: str, tasks: list[Task]) -> ScoreBundle:
            workspace = Path(codebase_path)
            assert list(workspace.iterdir()) == []
            private_task = workspace.parent / ".coral" / "private" / "harbor_task"
            assert (private_task / "tests" / "test.sh").is_file()
            assert (private_task / "solution" / "answer.txt").read_text() == "secret\n"
            assert tasks[0].name == "example/hello"
            return ScoreBundle(scores={"reward": Score(value=0.0, name="reward")}, aggregated=0.0)

    monkeypatch.setattr(
        "coral.workspace.grader_env.setup_grader_env",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("coral.grader.loader.load_grader", lambda *args: FakeGrader())

    result = run_validation(tmp_path)

    assert result.successful
    assert result.baseline is not None
    assert result.baseline.aggregated == 0.0
    events = {(event.stage, event.status): event.message for event in result.events}
    assert "Harbor source staged privately" in events[("workspace", "completed")]
    assert events[("baseline", "started")] == "Running grader against empty workspace..."


def test_create_project_stages_harbor_task_without_exposing_it_to_agents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_harbor_task(tmp_path)
    config = CoralConfig.from_yaml(_write_coral_config(tmp_path))
    config.task_dir = tmp_path
    config.workspace.run_dir = str(tmp_path / "run")
    monkeypatch.setattr(
        "coral.workspace.grader_env.setup_grader_env",
        lambda *args, **kwargs: Path("unused"),
    )

    paths = create_project(config, config_dir=tmp_path)

    assert (paths.coral_dir / "private" / "harbor_task" / "tests" / "test.sh").is_file()
    assert not (paths.repo_dir / "task.toml").exists()
    assert not (paths.repo_dir / "solution").exists()
    assert not (paths.repo_dir / "harbor-task").exists()


def test_score_bundle_preserves_named_rewards_and_primary_selection() -> None:
    result = {
        "runtime_version": HARBOR_RUNTIME_VERSION,
        "schema_version": "1.4",
        "task_name": "example/hello",
        "task_package_version": "1.0.0",
        "task_digest": "harbor-checksum",
        "rewards": {"reward": 1, "quality": 0.75},
    }

    bundle = _score_bundle_from_result(
        result,
        primary_reward="quality",
        direction="maximize",
        expected_digest="sha256:source",
        summary_path="eval_logs/abc/harbor-summary.json",
    )

    assert bundle.aggregated == 0.75
    assert bundle.scores["reward"].value == 1
    assert bundle.scores["quality"].value == 0.75
    assert bundle.metadata["harbor"]["source_digest"] == "sha256:source"
    assert bundle.metadata["harbor"]["direction"] == "maximize"
    assert "eval_logs/abc/harbor-summary.json" in (bundle.feedback or "")


def test_score_bundle_rejects_missing_primary_and_non_numeric_reward() -> None:
    base = {
        "runtime_version": HARBOR_RUNTIME_VERSION,
        "rewards": {"reward": 1},
    }
    with pytest.raises(RuntimeError, match="did not return primary reward"):
        _score_bundle_from_result(
            base,
            primary_reward="quality",
            direction="maximize",
            expected_digest="sha256:x",
            summary_path="summary.json",
        )

    with pytest.raises(RuntimeError, match="not numeric"):
        _score_bundle_from_result(
            {**base, "rewards": {"reward": True}},
            primary_reward="reward",
            direction="maximize",
            expected_digest="sha256:x",
            summary_path="summary.json",
        )


def test_score_bundle_preserves_zero_primary_reward() -> None:
    bundle = _score_bundle_from_result(
        {
            "runtime_version": HARBOR_RUNTIME_VERSION,
            "rewards": {"reward": 0},
        },
        primary_reward="reward",
        direction="minimize",
        expected_digest="sha256:x",
        summary_path="summary.json",
    )

    assert bundle.aggregated == 0.0
    assert bundle.scores["reward"].value == 0


def test_harbor_runner_timeout_is_a_grader_error_not_a_null_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_dir = _write_harbor_task(tmp_path)
    descriptor = inspect_local_harbor_task(task_dir)
    private_dir = tmp_path / "run" / ".coral" / "private"
    private_dir.mkdir(parents=True)
    stage_local_harbor_task(
        task_dir,
        base_dir=tmp_path,
        private_dir=private_dir,
        expected_digest=descriptor.digest,
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    grader = HarborTaskGrader(
        GraderConfig(
            timeout=10,
            args={
                "harbor_adapter": HARBOR_ADAPTER_MARKER,
                "harbor_runtime_version": HARBOR_RUNTIME_VERSION,
                "harbor_schema_version": descriptor.schema_version,
                "harbor_task_name": descriptor.name,
                "harbor_task_subdir": "harbor_task",
                "harbor_task_digest": descriptor.digest,
                "primary_reward": "reward",
            },
        )
    )
    grader.private_dir = str(private_dir)
    grader.codebase_path = str(candidate)

    def time_out(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise _HarborRunnerTimeoutError(8, "private stdout", "private stderr")

    monkeypatch.setattr("coral.grader.harbor._run_harbor_runner", time_out)

    with pytest.raises(RuntimeError, match="timed out after 8.0s"):
        grader.evaluate()
    attempt_runs = list((private_dir / "harbor_runs" / "candidate").iterdir())
    assert all(not (run / "candidate").exists() for run in attempt_runs)
    assert (attempt_runs[0] / "runner.stdout.log").read_text(encoding="utf-8") == ("private stdout")
    request = json.loads((attempt_runs[0] / "request.json").read_text(encoding="utf-8"))
    assert request["trial_name"].startswith("coral-eval-")


def test_runner_timeout_interrupts_before_force_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        returncode = 130

        def __init__(self) -> None:
            self.communicate_timeouts: list[float | None] = []
            self.signals: list[int] = []
            self.terminated = False
            self.killed = False

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            self.communicate_timeouts.append(timeout)
            if len(self.communicate_timeouts) == 1:
                raise subprocess.TimeoutExpired(cmd="runner", timeout=timeout or 0)
            return "cancelled stdout", "cancelled stderr"

        def send_signal(self, sent_signal: int) -> None:
            self.signals.append(sent_signal)

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    process = FakeProcess()
    monkeypatch.setattr("coral.grader.harbor.subprocess.Popen", lambda *a, **kw: process)

    with pytest.raises(_HarborRunnerTimeoutError) as captured:
        _run_harbor_runner(
            ["runner"],
            environment={},
            timeout=8,
            cleanup_grace=2,
        )

    assert process.communicate_timeouts == [8, 2]
    if os.name == "posix":
        assert process.signals == [signal.SIGINT]
        assert not process.terminated
    else:
        assert process.signals == []
        assert process.terminated
    assert not process.killed
    assert captured.value.stdout == "cancelled stdout"


def test_candidate_copy_excludes_runtime_state_and_preserves_safe_symlinks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "keep.txt").write_text("ok", encoding="utf-8")
    (source / ".git").write_text("gitdir: elsewhere", encoding="utf-8")
    (source / "cache.pyc").write_bytes(b"bytecode")
    (source / "private.txt").write_text("private", encoding="utf-8")
    (source / "link").symlink_to("private.txt")
    (source / ".codex").symlink_to(tmp_path / "shared-state")
    destination = tmp_path / "destination"

    _copy_candidate_workspace(source, destination)

    assert (destination / "keep.txt").read_text(encoding="utf-8") == "ok"
    assert not (destination / ".git").exists()
    assert not (destination / "cache.pyc").exists()
    assert not (destination / ".codex").exists()
    assert (destination / "link").is_symlink()
    assert (destination / "link").readlink() == Path("private.txt")


def test_candidate_copy_keeps_runtime_config_but_skips_shared_state_links(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    runtime_dir = source / ".codex"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "config.toml").write_text("model = 'test'\n", encoding="utf-8")
    shared = tmp_path / "shared" / "notes"
    shared.mkdir(parents=True)
    (runtime_dir / "notes").symlink_to(shared)

    destination = tmp_path / "destination"
    _copy_candidate_workspace(source, destination)

    assert (destination / ".codex" / "config.toml").is_file()
    assert not (destination / ".codex" / "notes").exists()


def test_candidate_copy_rejects_symlink_that_escapes_repository(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "escape").symlink_to("../outside")

    with pytest.raises(ValueError, match="symlink escapes"):
        _copy_candidate_workspace(source, tmp_path / "destination")


def test_public_summary_shape_does_not_include_private_trial_path(tmp_path: Path) -> None:
    summary = {
        "runtime_version": HARBOR_RUNTIME_VERSION,
        "schema_version": "1.4",
        "task_name": "example/hello",
        "rewards": {"reward": 1},
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    assert "private" not in path.read_text(encoding="utf-8")


@pytest.mark.skipif(
    os.environ.get("CORAL_HARBOR_E2E") != "1",
    reason="set CORAL_HARBOR_E2E=1 when Docker and network are available",
)
def test_harbor_task_grader_end_to_end(tmp_path: Path) -> None:
    task_dir = _write_harbor_task(tmp_path)
    (task_dir / "tests" / "test.sh").write_text(
        "#!/bin/sh\n"
        'if [ "$(cat /app/hello.txt)" = "Hello, world!" ]; then\n'
        "  echo 1 > /logs/verifier/reward.txt\n"
        "else\n"
        "  echo 0 > /logs/verifier/reward.txt\n"
        "fi\n",
        encoding="utf-8",
    )
    descriptor = inspect_local_harbor_task(task_dir)
    private_dir = tmp_path / "run" / ".coral" / "private"
    private_dir.mkdir(parents=True)
    stage_local_harbor_task(
        task_dir,
        base_dir=tmp_path,
        private_dir=private_dir,
        expected_digest=descriptor.digest,
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "hello.txt").write_text("Hello, world!\n", encoding="utf-8")

    grader = HarborTaskGrader(
        GraderConfig(
            timeout=600,
            args={
                "harbor_adapter": HARBOR_ADAPTER_MARKER,
                "harbor_runtime_version": HARBOR_RUNTIME_VERSION,
                "harbor_schema_version": descriptor.schema_version,
                "harbor_task_name": descriptor.name,
                "harbor_task_subdir": "harbor_task",
                "harbor_task_digest": descriptor.digest,
                "primary_reward": "reward",
            },
        )
    )
    grader.private_dir = str(private_dir)
    grader.codebase_path = str(candidate)

    bundle = grader.evaluate()

    assert bundle.aggregated == 1.0
    assert bundle.scores["reward"].value == 1.0
    summary = private_dir.parent / "public" / "eval_logs" / "candidate" / "harbor-summary.json"
    assert json.loads(summary.read_text(encoding="utf-8"))["rewards"] == {"reward": 1.0}
    attempt_runs = list((private_dir / "harbor_runs" / "candidate").iterdir())
    assert all(not (run / "candidate").exists() for run in attempt_runs)
