"""Unit tests for the DeepSeek Harness runtime."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from coral.agent.builtin.deepseek_harness import DeepSeekHarnessRuntime
from coral.agent.registry import (
    default_command_for_runtime,
    default_model_for_runtime,
    get_runtime,
)


class _FakePopen:
    captured: list[dict[str, Any]] = []

    def __init__(self, cmd, **kwargs) -> None:  # type: ignore[no-untyped-def]
        type(self).captured.append({"cmd": list(cmd), "kwargs": dict(kwargs)})
        self.pid = 4242
        # Report an already-exited process. If the fake claims to be alive,
        # AgentHandle.__del__ SIGKILLs the process group of whatever real
        # process happens to own pid 4242 on the host, and the AttributeError
        # from the missing Popen surface aborts cleanup before the runtime's
        # log handle is closed (ResourceWarning at teardown).
        self.returncode: int | None = 0
        self.stdout = None
        self.stderr = None

    def poll(self) -> int | None:
        return self.returncode


@pytest.fixture(autouse=True)
def _reset_fake_popen() -> None:
    _FakePopen.captured = []


def _make_worktree(tmp_path: Path) -> Path:
    worktree = tmp_path / "agent-1"
    worktree.mkdir()
    (worktree / ".coral_agent_id").write_text("agent-1")
    return worktree


@pytest.mark.parametrize("name", ["dsh", "deepseek", "deepseek-harness", "deepseek_harness"])
def test_registry_resolves_runtime_and_aliases(name: str) -> None:
    assert isinstance(get_runtime(name), DeepSeekHarnessRuntime)


def test_registry_defaults() -> None:
    assert default_model_for_runtime("dsh") == "deepseek-v4-flash"
    assert default_command_for_runtime("deepseek") == "dsh"


def test_runtime_conventions() -> None:
    runtime = DeepSeekHarnessRuntime()
    assert runtime.instruction_filename == "AGENTS.md"
    assert runtime.shared_dir_name == ".dsh"
    assert runtime.extract_session_id(Path("unused")) is None


def test_start_builds_headless_command_and_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    worktree = _make_worktree(tmp_path)

    handle = DeepSeekHarnessRuntime().start(
        worktree_path=worktree,
        coral_md_path=worktree / "AGENTS.md",
        log_dir=tmp_path / "logs",
        prompt="solve this",
        gateway_url="https://gateway.example/v1",
        gateway_api_key="secret",
    )

    assert handle.agent_id == "agent-1"
    call = _FakePopen.captured[0]
    model_patch = worktree / ".dsh" / "coral-model.patch.yml"
    assert call["cmd"] == [
        "dsh",
        "--profile",
        "headless",
        "--patch",
        str(model_patch),
        "solve this",
    ]
    assert 'provider: "deepseek-official"' in model_patch.read_text()
    assert 'model: "deepseek-v4-flash"' in model_patch.read_text()
    assert call["kwargs"]["cwd"] == str(worktree)
    env = call["kwargs"]["env"]
    assert env["DSH_HOME"] == str(worktree / ".dsh")
    assert env["DEEPSEEK_BASE_URL"] == "https://gateway.example/v1"
    assert env["DEEPSEEK_API_KEY"] == "secret"


def test_start_honors_runtime_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    worktree = _make_worktree(tmp_path)

    DeepSeekHarnessRuntime().start(
        worktree_path=worktree,
        coral_md_path=worktree / "AGENTS.md",
        log_dir=tmp_path / "logs",
        prompt="task",
        runtime_options={
            "command": "/opt/dsh",
            "profile": "coral",
            "provider": "private-deepseek",
            "patch": ["one.yml", "two.yml"],
            "permission_mode": "workspace-write",
            "tools_mode": "native",
        },
    )

    call = _FakePopen.captured[0]
    assert call["cmd"] == [
        "/opt/dsh",
        "--profile",
        "coral",
        "--patch",
        "one.yml",
        "--patch",
        "two.yml",
        "--patch",
        str(worktree / ".dsh" / "coral-model.patch.yml"),
        "task",
    ]
    assert (
        'provider: "private-deepseek"' in (worktree / ".dsh" / "coral-model.patch.yml").read_text()
    )
    assert call["kwargs"]["env"]["DSH_PERMISSION_MODE"] == "workspace-write"
    assert call["kwargs"]["env"]["DSH_TOOLS_MODE"] == "native"


def test_restart_does_not_invent_unsupported_resume_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    worktree = _make_worktree(tmp_path)

    DeepSeekHarnessRuntime().start(
        worktree_path=worktree,
        coral_md_path=worktree / "AGENTS.md",
        log_dir=tmp_path / "logs",
        resume_session_id="not-exposed-by-headless",
        prompt="continue",
    )

    cmd = _FakePopen.captured[0]["cmd"]
    assert cmd[:3] == ["dsh", "--profile", "headless"]
    assert cmd[-1] == "continue"
    assert "--resume" not in cmd


def test_start_prepends_platform_aware_venv_bin_to_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PATH must use the platform's venv executable dir (bin/ vs Scripts/).

    Guards the cross-platform venv path contract (#231): a hardcoded
    ``.venv/bin`` would point at a nonexistent directory on Windows.
    """
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    worktree = _make_worktree(tmp_path)

    monkeypatch.setattr(sys, "platform", "win32")
    DeepSeekHarnessRuntime().start(
        worktree_path=worktree,
        coral_md_path=worktree / "AGENTS.md",
        log_dir=tmp_path / "logs",
        prompt="task",
    )

    path_value = _FakePopen.captured[0]["kwargs"]["env"]["PATH"]
    first_entry = path_value.split(os.pathsep)[0]
    assert first_entry == str(worktree / ".venv" / "Scripts")


def test_start_opens_log_file_utf8_with_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The agent log must be UTF-8 with errors='replace' like all runtimes.

    Without an explicit encoding, the log file inherits the locale encoding,
    and the verbose tee (which writes UTF-8-decoded agent output through this
    handle) raises UnicodeEncodeError under a non-UTF-8 locale.
    """
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    worktree = _make_worktree(tmp_path)

    handle = DeepSeekHarnessRuntime().start(
        worktree_path=worktree,
        coral_md_path=worktree / "AGENTS.md",
        log_dir=tmp_path / "logs",
        prompt="task",
    )

    log_file = handle._log_file
    assert log_file is not None
    assert log_file.encoding.lower().replace("-", "") == "utf8"
    assert log_file.errors == "replace"
