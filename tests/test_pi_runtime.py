# ABOUTME: Verifies CORAL can launch Pi coding agent subprocesses as agent runtimes.
# ABOUTME: Covers runtime registration, command construction, and session extraction.
from __future__ import annotations

from pathlib import Path

from coral.agent.builtin.pi_agent import PiAgentRuntime, _extract_pi_session_id
from coral.agent.registry import default_model_for_runtime, get_runtime


class FakeProcess:
    def __init__(self) -> None:
        self.pid = 4321
        # Report an already-exited process. If the fake claims to be alive,
        # AgentHandle.__del__ SIGKILLs the process group of whatever real
        # process happens to own pid 4321 on the host, and the AttributeError
        # from the missing Popen surface aborts cleanup before the runtime's
        # log handle is closed (ResourceWarning at teardown).
        self.returncode: int | None = 0
        self.stdout = None
        self.stderr = None

    def poll(self) -> int | None:
        return self.returncode


def test_pi_runtime_is_registered() -> None:
    assert isinstance(get_runtime("pi"), PiAgentRuntime)
    assert isinstance(get_runtime("pi-agent"), PiAgentRuntime)
    assert default_model_for_runtime("pi") == "zai/glm-5.1"


def test_extract_pi_session_id_from_json_log(tmp_path: Path) -> None:
    log = tmp_path / "agent.log"
    log.write_text(
        '{"type":"session","version":3,"id":"019e4307-ac12-738d-83cf-8d0b8a05bd8d"}\n'
        '{"type":"agent_start"}\n'
    )

    assert _extract_pi_session_id(log) == "019e4307-ac12-738d-83cf-8d0b8a05bd8d"


def test_pi_runtime_start_builds_noninteractive_command(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return FakeProcess()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / ".coral_agent_id").write_text("agent-1")

    runtime = PiAgentRuntime()
    handle = runtime.start(
        worktree_path=worktree,
        coral_md_path=worktree / "AGENTS.md",
        model="zai/glm-5.1",
        runtime_options={"model_reasoning_effort": "medium"},
        prompt="Begin.",
    )

    cmd, kwargs = calls[0]
    assert handle.agent_id == "agent-1"
    assert cmd == [
        "pi",
        "--print",
        "--mode",
        "json",
        "--model",
        "zai/glm-5.1",
        "--thinking",
        "medium",
        "--session-dir",
        str(worktree / ".pi" / "sessions"),
        "--tools",
        "read,bash,edit,write,grep,find,ls",
        "Begin.",
    ]
    assert kwargs["cwd"] == str(worktree)
    assert kwargs["start_new_session"] is True


def test_pi_runtime_start_resumes_specific_session(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return FakeProcess()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / ".coral_agent_id").write_text("agent-2")

    runtime = PiAgentRuntime()
    runtime.start(
        worktree_path=worktree,
        coral_md_path=worktree / "AGENTS.md",
        model="zai/glm-5.1",
        resume_session_id="019e4307-ac12-738d-83cf-8d0b8a05bd8d",
        prompt="Continue.",
    )

    cmd = calls[0][0]
    assert "--continue" in cmd
    assert cmd[cmd.index("--session") + 1] == "019e4307-ac12-738d-83cf-8d0b8a05bd8d"
