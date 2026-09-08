"""Artifact isolation and result handling for the RSI-Exam example (no Docker)."""

import asyncio
import hashlib
import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path

import pytest
import yaml

from coral.cli.validation import validate_task
from coral.config import CoralConfig, GraderConfig
from coral.types import Task

SUITE = Path(__file__).resolve().parents[1] / "examples/rsi_exam"
EXAMPLE = SUITE / "tidal_friction_inverse"
sys.path.insert(0, str(SUITE / "_grader/src"))
from rsi_exam_grader.contract import artifact_sources, submission_path  # noqa: E402
from rsi_exam_grader.grader import Grader, read_reward  # noqa: E402


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_result(job, reward=0.25, **overrides):
    result = {
        "finished_at": "2026-09-08T00:00:00Z",
        "exception_info": None,
        "verifier_environment_mode": "separate",
        "verifier_result": {"rewards": {"reward": reward}},
        **overrides,
    }
    path = job / "trial/result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))


@pytest.mark.parametrize("reward", [0.0, 0.6, 17.5])
def test_reward_preserves_upstream_scale(tmp_path, reward):
    write_result(tmp_path, reward)
    # The top-level result is a job aggregate, not the trial reward.
    (tmp_path / "result.json").write_text('{"reward": 999}')
    assert read_reward(tmp_path) == reward


@pytest.mark.parametrize("reward", [None, True, "0.5", float("nan"), float("inf")])
def test_invalid_reward_is_not_a_score(tmp_path, reward):
    write_result(tmp_path, reward)
    with pytest.raises(ValueError, match="reward"):
        read_reward(tmp_path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"exception_info": {"exception_message": "secret heldout case"}},
        {"finished_at": None},
        {"verifier_environment_mode": "shared"},
        {"verifier_result": None},
    ],
)
def test_failed_or_unsealed_trial_is_rejected(tmp_path, overrides):
    write_result(tmp_path, **overrides)
    with pytest.raises(ValueError) as exc:
        read_reward(tmp_path)
    assert "secret heldout case" not in str(exc.value)


def test_missing_or_multiple_trials_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="found 0"):
        read_reward(tmp_path)
    write_result(tmp_path)
    shutil.copytree(tmp_path / "trial", tmp_path / "trial2")
    with pytest.raises(ValueError, match="found 2"):
        read_reward(tmp_path)


def test_both_upstream_artifact_formats():
    assert artifact_sources(
        {"artifacts": ["/app/methods", {"source": "/app/submission", "destination": "submission"}]}
    ) == ["/app/methods", "/app/submission"]


@pytest.mark.parametrize(
    "sources",
    [
        [],
        ["/app"],
        ["/tests"],
        ["methods"],
        ["/app/../tests"],
        ["/app/methods", "/app/methods/main"],
        [{"source": "/app/methods", "service": "database"}],
    ],
)
def test_invalid_artifact_sources(sources):
    with pytest.raises(ValueError):
        artifact_sources({"artifacts": sources})


@pytest.mark.parametrize("nested", [False, True])
def test_submission_symlinks_cannot_read_outside_checkout(tmp_path, nested):
    root = tmp_path / "checkout"
    root.mkdir()
    secret = tmp_path / "secret"
    secret.write_text("hidden")
    if nested:
        (root / "methods").mkdir()
        (root / "methods/key").symlink_to(secret)
    else:
        (root / "methods").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        submission_path(root, "/app/methods")


def test_default_task_and_pinned_upstream_integrity():
    assert validate_task(EXAMPLE) == []
    cfg = CoralConfig.from_yaml(EXAMPLE / "task.yaml")
    assert cfg.grader.direction == "maximize"
    assert cfg.grader.private == ["rsi_task"]
    assert cfg.run.stop.max_real_attempts == 1
    manifest = json.loads((EXAMPLE / "UPSTREAM.json").read_text())
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((EXAMPLE / "rsi_task" / name).read_bytes()).hexdigest() == digest
    assert not (EXAMPLE / "seed/harbor/tests").exists()
    assert not (EXAMPLE / "grader/src/rsi_exam_grader/tests").exists()
    assert (EXAMPLE / "seed/methods/main/solver.py").exists()


def test_every_public_task_has_its_own_coral_config():
    names = json.loads((SUITE / "tasks.json").read_text())
    configs = sorted(p.parent.name for p in SUITE.glob("*/task.yaml"))
    assert configs == sorted(names)
    for name in names:
        task = SUITE / name
        assert validate_task(task) == [], name
        cfg = CoralConfig.from_yaml(task / "task.yaml")
        assert Path(cfg.workspace.repo_path).resolve() == task / "seed"
        assert cfg.task.name == f"rsi-exam-{name}"
        assert (task / "seed/instruction.md").is_file()
        assert not (task / "seed/harbor/tests").exists()
        for filename in ["grader.py", "contract.py", "replay.py", "qlib.py", "qlib_docker.py"]:
            canonical = SUITE / "_grader/src/rsi_exam_grader" / filename
            assert (
                task / "grader/src/rsi_exam_grader" / filename
            ).read_bytes() == canonical.read_bytes()
        assert (task / "seed/rsi_runtime.py").read_bytes() == (
            SUITE / "_grader/src/rsi_exam_grader/replay.py"
        ).read_bytes()
        manifest = json.loads((task / "UPSTREAM.json").read_text())
        for path, digest in manifest["sha256"].items():
            assert hashlib.sha256((task / "rsi_task" / path).read_bytes()).hexdigest() == digest


def test_qlib_sandbox_configuration_and_workspace_helpers():
    task = SUITE / "qlib_alpha_factor_icir"
    config = CoralConfig.from_yaml(task / "task.yaml")
    assert config.agents.runtime == "codex"
    assert config.agents.model == "gpt-6-astra"
    assert config.agents.count == 2
    assert config.agents.sandbox.enabled
    assert config.run.stop.max_real_attempts == 10
    assert config.grader.parallel.max_workers == 1
    assert config.grader.entrypoint == "rsi_exam_grader.qlib:Grader"
    assert "python setup_visible.py" in config.workspace.setup
    assert (task / "seed/qlib_docker.py").read_bytes() == (
        SUITE / "_grader/src/rsi_exam_grader/qlib_docker.py"
    ).read_bytes()
    assert (task / "seed/selfcheck.py").read_bytes() == (
        task / "rsi_task/environment/selfcheck.py"
    ).read_bytes()


def test_importer_is_standalone_and_keeps_tests_private(tmp_path):
    prepare = load_file("rsi_prepare_test", SUITE / "prepare.py")
    assert len(prepare.public_tasks()) == 35
    output = tmp_path / "new-task"
    prepare.build_task(EXAMPLE / "rsi_task", output, revision="test-source")
    assert validate_task(output) == []
    cfg = yaml.safe_load((output / "task.yaml").read_text())
    assert Path(cfg["workspace"]["repo_path"]) == output / "seed"
    assert cfg["grader"]["timeout"] >= 1200 + 900 + 900
    assert (output / "rsi_task/tests/anchors.json").is_file()
    assert not list((output / "seed").rglob("anchors.json"))
    assert not list((output / "grader").rglob("anchors.json"))
    assert (output / "seed/rsi_runtime.py").is_file()
    with pytest.raises(FileExistsError):
        prepare.build_task(EXAMPLE / "rsi_task", output)


def make_grader(tmp_path):
    private = tmp_path / ".coral/private"
    private.mkdir(parents=True)
    shutil.copytree(EXAMPLE / "rsi_task", private / "rsi_task")
    worktree = tmp_path / "worktree"
    shutil.copytree(EXAMPLE / "seed", worktree)
    grader = Grader(GraderConfig(timeout=90))
    grader.private_dir = str(private)
    grader.codebase_path = str(worktree)
    return grader, private, worktree


def test_grader_uses_private_runtime_and_logs(tmp_path, monkeypatch):
    grader, private, worktree = make_grader(tmp_path)
    before = {str(p): p.read_bytes() for p in worktree.rglob("*") if p.is_file()}
    monkeypatch.setenv("PYTHONPATH", str(worktree))

    class Process:
        def __init__(self, command, **kwargs):
            assert Path(kwargs["cwd"]).is_relative_to(private)
            assert "PYTHONPATH" not in kwargs["env"]
            assert kwargs["start_new_session"]
            assert command[command.index("--path") + 1] == str(private / "rsi_task")
            assert "--disable-verification" not in command
            jobs = Path(command[command.index("--jobs-dir") + 1])
            job = command[command.index("--job-name") + 1]
            assert jobs.is_relative_to(private)
            kwargs["stdout"].write("secret trial diagnostics")
            write_result(jobs / job, reward=2.5)

        def wait(self, timeout):
            assert timeout < 90
            return 0

    monkeypatch.setattr("rsi_exam_grader.grader.subprocess.Popen", Process)
    result = grader.evaluate()
    assert result.aggregated == 2.5
    assert "secret" not in str(result)
    assert not (private.parent / "public").exists()
    assert before == {str(p): p.read_bytes() for p in worktree.rglob("*") if p.is_file()}


def test_tune_never_launches_hidden_grading(tmp_path, monkeypatch):
    grader, private, _ = make_grader(tmp_path)
    grader.tasks = [Task(id="tune", name="tune", description="", metadata={"budget_class": "tune"})]
    monkeypatch.setattr("rsi_exam_grader.grader.subprocess.Popen", lambda *a, **k: pytest.fail())
    assert grader.evaluate().aggregated is None
    assert not (private / "rsi_jobs").exists()


def test_missing_data_is_reported_before_harbor_launch(tmp_path, monkeypatch):
    grader, private, _ = make_grader(tmp_path)
    (private / "rsi_task/assets.json").write_text('[{"path":"tests/heldout/missing.npz"}]')
    monkeypatch.setattr("rsi_exam_grader.grader.subprocess.Popen", lambda *a, **k: pytest.fail())
    with pytest.raises(RuntimeError, match="prepare.py"):
        grader.evaluate()


def test_harbor_launch_failure_is_grader_error(tmp_path, monkeypatch):
    grader, _, _ = make_grader(tmp_path)

    class FailedProcess:
        def __init__(self, *args, **kwargs):
            pass

        def wait(self, timeout):
            return 1

    monkeypatch.setattr("rsi_exam_grader.grader.subprocess.Popen", FailedProcess)
    with pytest.raises(RuntimeError, match="Harbor exited 1"):
        grader.evaluate()


@pytest.mark.parametrize(
    "failure", [None, "create", "start_timeout", "missing_reward", "nonfinite_reward"]
)
def test_qlib_offline_verifier_isolation_and_cleanup(tmp_path, monkeypatch, failure):
    import subprocess

    from rsi_exam_grader.qlib import Grader as QlibGrader

    private = tmp_path / ".coral/private"
    (private / "rsi_task").mkdir(parents=True)
    shutil.copy2(
        SUITE / "qlib_alpha_factor_icir/rsi_task/task.toml", private / "rsi_task/task.toml"
    )
    worktree = tmp_path / "candidate"
    (worktree / "methods/main").mkdir(parents=True)
    (worktree / "methods/main/solver.py").write_text("candidate")
    grader = QlibGrader(GraderConfig())
    grader.private_dir = str(private)
    grader.codebase_path = str(worktree)
    monkeypatch.setattr("rsi_exam_grader.qlib.ensure_image", lambda *a: "trusted-image")
    operations = []

    def run(command, **kwargs):
        operations.append(command)
        assert "-v" not in command and "--mount" not in command
        if command[1] == "create":
            assert command[command.index("--network") + 1] == "none"
            assert "trusted-image" in command
            if failure == "create":
                raise subprocess.CalledProcessError(1, command, stderr="hidden case")
        if command[1] == "start" and failure == "start_timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if command[1] == "cp":
            if command[2].endswith(":/logs/verifier/."):
                output = Path(command[-1])
                assert output.is_relative_to(private)
                if failure != "missing_reward":
                    reward = float("nan") if failure == "nonfinite_reward" else 0.37
                    (output / "reward.json").write_text(json.dumps({"reward": reward}))
            else:
                assert Path(command[2]) == worktree / "methods"
                assert command[-1].endswith(":/submission")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr("rsi_exam_grader.qlib.subprocess.run", run)
    if failure in ("create", "missing_reward", "nonfinite_reward"):
        with pytest.raises(RuntimeError, match="infrastructure failed") as exc:
            grader.evaluate()
        assert "hidden case" not in str(exc.value)
        public = private.parent / "public"
        assert "hidden case" not in "".join(p.read_text() for p in public.rglob("*.txt"))
    else:
        result = grader.evaluate()
        assert result.aggregated == (0.0 if failure else 0.37)
    assert operations[-1][1:3] == ["rm", "-f"]
    assert (worktree / "methods/main/solver.py").read_text() == "candidate"


def test_asset_preparation_checks_hashes_and_keeps_edits(tmp_path):
    prepare = load_file("rsi_prepare_assets", SUITE / "prepare.py")
    task = tmp_path / "task"
    task.mkdir()
    source = tmp_path / "source"
    data_path = source / "test_task/environment/data/visible.json"
    data_path.parent.mkdir(parents=True)
    data_path.write_text("original data")
    data = data_path.read_bytes()
    record = {
        "path": "environment/data/visible.json",
        "size": len(data),
        "git_blob_sha1": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
    }
    (task / "UPSTREAM.json").write_text(json.dumps({"task": "test_task", "downloads": [record]}))
    edited = task / "seed/harbor/environment/data/visible.json"
    edited.parent.mkdir(parents=True)
    edited.write_text("user edit")
    prepare.hydrate_task(task, source=source)
    assert (task / "rsi_task/environment/data/visible.json").read_bytes() == data
    assert edited.read_text() == "user edit"
    # Existing files make repeated preparation a no-op, even if the source is gone.
    prepare.hydrate_task(task, source=tmp_path / "missing")
    (task / "rsi_task/environment/data/visible.json").unlink()
    data_path.write_text("corrupt data")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare.hydrate_task(task, source=source)


def test_replay_transfers_only_artifacts_and_removes_deleted_baseline(tmp_path, monkeypatch):
    # The shared test environment intentionally does not install Harbor/LiteLLM.
    # This stub supplies only the BaseAgent constructor; all transfer code is real.
    base = types.ModuleType("harbor.agents.base")
    base.BaseAgent = type("BaseAgent", (), {"__init__": lambda self, *a, **k: None})
    monkeypatch.setitem(sys.modules, "harbor", types.ModuleType("harbor"))
    monkeypatch.setitem(sys.modules, "harbor.agents", types.ModuleType("harbor.agents"))
    monkeypatch.setitem(sys.modules, "harbor.agents.base", base)
    replay = load_file(
        "rsi_exam_grader.replay_test", SUITE / "_grader/src/rsi_exam_grader/replay.py"
    )
    (tmp_path / "methods").mkdir()
    (tmp_path / "methods/solver.py").write_text("candidate")
    (tmp_path / "task.toml").write_text("untrusted config")
    operations = []

    class Environment:
        async def exec(self, **kwargs):
            operations.append(kwargs["command"])
            return types.SimpleNamespace(return_code=0)

        async def upload_dir(self, source_dir, target_dir):
            operations.append((source_dir, target_dir))

    agent = replay.ReplayAgent(
        source_dir=str(tmp_path), sources=["/app/methods", "/app/submission"]
    )
    asyncio.run(agent.run("", Environment(), None))
    assert len(operations) == 3
    assert operations[1] == (tmp_path / "methods", "/app/methods")
    assert "rm -rf -- /app/submission" in operations[2]
    assert all("task.toml" not in str(op) for op in operations)


def test_bootstrap_merges_image_files_and_preserves_local_work(tmp_path, monkeypatch):
    base = types.ModuleType("harbor.agents.base")
    base.BaseAgent = type("BaseAgent", (), {"__init__": lambda self, *a, **k: None})
    monkeypatch.setitem(sys.modules, "harbor", types.ModuleType("harbor"))
    monkeypatch.setitem(sys.modules, "harbor.agents", types.ModuleType("harbor.agents"))
    monkeypatch.setitem(sys.modules, "harbor.agents.base", base)
    replay = load_file(
        "rsi_exam_grader.bootstrap_test", SUITE / "_grader/src/rsi_exam_grader/replay.py"
    )
    (tmp_path / "methods").mkdir()
    (tmp_path / "methods/solver.py").write_text("my changes")
    (tmp_path / "experiment_log.md").write_text("local notes")

    class Environment:
        async def exec(self, command):
            return types.SimpleNamespace(return_code=int("experiment_log" in command))

        async def download_dir(self, source_dir, target_dir):
            target_dir.mkdir()
            (target_dir / "solver.py").write_text("baseline")
            (target_dir / "generated.json").write_text("built in Dockerfile")

    agent = replay.VisibleAgent(
        source_dir=str(tmp_path), sources=["/app/methods", "/app/experiment_log.md"], bootstrap=True
    )
    asyncio.run(agent.run("", Environment(), None))
    assert (tmp_path / "methods/solver.py").read_text() == "my changes"
    assert (tmp_path / "methods/generated.json").read_text() == "built in Dockerfile"
    assert (tmp_path / "experiment_log.md").read_text() == "local notes"
