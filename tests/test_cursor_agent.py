"""Unit tests for the cursor_agent runtime."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from coral.agent.builtin.cursor_agent import (
    CursorAgentRuntime,
    _extract_cursor_session_id,
)
from coral.agent.registry import default_model_for_runtime, get_runtime
from coral.workspace.worktree import setup_cursor_settings

# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_registry_resolves_canonical_name() -> None:
    assert isinstance(get_runtime("cursor_agent"), CursorAgentRuntime)


@pytest.mark.parametrize("alias", ["cursor", "cursor-agent"])
def test_registry_resolves_aliases(alias: str) -> None:
    assert isinstance(get_runtime(alias), CursorAgentRuntime)


def test_registry_default_model() -> None:
    assert default_model_for_runtime("cursor_agent") == "auto"
    assert default_model_for_runtime("cursor") == "auto"


# ---------------------------------------------------------------------------
# Static properties
# ---------------------------------------------------------------------------


def test_runtime_filenames_match_cursor_conventions() -> None:
    runtime = CursorAgentRuntime()
    assert runtime.instruction_filename == "AGENTS.md"
    assert runtime.shared_dir_name == ".cursor"


# ---------------------------------------------------------------------------
# classify_exit — uptime-based fallback (same shape as codex/kiro/opencode)
# ---------------------------------------------------------------------------


def test_classify_exit_clean_after_long_uptime(tmp_path: Path) -> None:
    log = tmp_path / "agent.log"
    log.write_text('{"type":"result"}\n')
    runtime = CursorAgentRuntime()
    assert runtime.classify_exit(log, exit_code=0, uptime_seconds=120.0) == "clean"


def test_classify_exit_no_result_when_short_uptime(tmp_path: Path) -> None:
    log = tmp_path / "agent.log"
    log.write_text('{"type":"system","subtype":"init","session_id":"x"}\n')
    runtime = CursorAgentRuntime()
    assert runtime.classify_exit(log, exit_code=0, uptime_seconds=5.0) == "no_result"


def test_classify_exit_no_result_when_nonzero_exit(tmp_path: Path) -> None:
    log = tmp_path / "agent.log"
    log.write_text("\n")
    runtime = CursorAgentRuntime()
    assert runtime.classify_exit(log, exit_code=1, uptime_seconds=300.0) == "no_result"


# ---------------------------------------------------------------------------
# Session id extraction — `system/init` event is authoritative
# ---------------------------------------------------------------------------


def test_extract_session_id_from_system_init(tmp_path: Path) -> None:
    log = tmp_path / "agent.log"
    log.write_text(
        '{"type":"system","subtype":"init","session_id":"abc-123","permissionMode":"force"}\n'
        '{"type":"assistant","message":{"content":[]}}\n'
    )
    assert _extract_cursor_session_id(log) == "abc-123"


def test_extract_session_id_falls_back_to_any_session_field(tmp_path: Path) -> None:
    log = tmp_path / "agent.log"
    log.write_text('{"type":"assistant","session_id":"fallback-1"}\n{"type":"result"}\n')
    assert _extract_cursor_session_id(log) == "fallback-1"


def test_extract_session_id_skips_malformed_lines(tmp_path: Path) -> None:
    log = tmp_path / "agent.log"
    log.write_text(
        'not-json\n{"type":"system","subtype":"init","session_id":"sid-7"}\nanother bad line\n'
    )
    assert _extract_cursor_session_id(log) == "sid-7"


def test_extract_session_id_returns_none_when_absent(tmp_path: Path) -> None:
    log = tmp_path / "agent.log"
    log.write_text('{"type":"assistant"}\n{"type":"result"}\n')
    assert _extract_cursor_session_id(log) is None


def test_extract_session_id_returns_none_when_log_missing(tmp_path: Path) -> None:
    assert _extract_cursor_session_id(tmp_path / "missing.log") is None


# ---------------------------------------------------------------------------
# Spawn argv — the whole point of the runtime; lock the flag set down so
# upstream cursor-agent flag drift surfaces here instead of in production
# ---------------------------------------------------------------------------


class _FakePopen:
    """Minimal subprocess.Popen stand-in — captures argv and env."""

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

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.returncode = -9

    def terminate(self) -> None:
        self.returncode = -15

    def send_signal(self, sig: int) -> None:
        pass


@pytest.fixture(autouse=True)
def _reset_fake_popen() -> None:
    _FakePopen.captured = []


def _make_worktree(tmp_path: Path, agent_id: str = "agent-1") -> Path:
    wt = tmp_path / agent_id
    wt.mkdir()
    (wt / ".coral_agent_id").write_text(agent_id)
    return wt


def test_start_builds_baseline_argv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    wt = _make_worktree(tmp_path)
    runtime = CursorAgentRuntime()
    log_dir = tmp_path / "logs"
    handle = runtime.start(
        worktree_path=wt,
        coral_md_path=wt / "AGENTS.md",
        model="auto",
        log_dir=log_dir,
        prompt="hello world",
    )

    assert handle.agent_id == "agent-1"
    assert len(_FakePopen.captured) == 1
    cmd = _FakePopen.captured[0]["cmd"]

    # Required flags, in the order the cursor-acp reference impl uses.
    assert cmd[0] == "cursor-agent"
    assert cmd[1:5] == ["--print", "--output-format", "stream-json", "--force"]
    assert "--workspace" in cmd
    assert cmd[cmd.index("--workspace") + 1] == str(wt)

    # model=="auto" is the registry default → omit --model so the CLI picks
    # its own default model.
    assert "--model" not in cmd

    # Prompt is the final positional arg, never a flag.
    assert cmd[-1] == "hello world"


def test_start_includes_resume_and_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    wt = _make_worktree(tmp_path)
    runtime = CursorAgentRuntime()
    runtime.start(
        worktree_path=wt,
        coral_md_path=wt / "AGENTS.md",
        model="claude-4.6-sonnet",
        log_dir=tmp_path / "logs",
        resume_session_id="sess-xyz",
        prompt="continue",
    )

    cmd = _FakePopen.captured[0]["cmd"]
    assert cmd[cmd.index("--model") + 1] == "claude-4.6-sonnet"
    assert cmd[cmd.index("--resume") + 1] == "sess-xyz"
    assert cmd[-1] == "continue"


def test_start_honors_runtime_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    wt = _make_worktree(tmp_path)
    runtime = CursorAgentRuntime()
    runtime.start(
        worktree_path=wt,
        coral_md_path=wt / "AGENTS.md",
        log_dir=tmp_path / "logs",
        runtime_options={
            "command": "/usr/local/bin/agent",
            "mode": "plan",
            "stream_partial_output": True,
        },
        prompt="x",
    )

    cmd = _FakePopen.captured[0]["cmd"]
    assert cmd[0] == "/usr/local/bin/agent"
    assert cmd[cmd.index("--mode") + 1] == "plan"
    assert "--stream-partial-output" in cmd


def test_start_uses_default_prompt_for_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    wt = _make_worktree(tmp_path)
    runtime = CursorAgentRuntime()
    runtime.start(
        worktree_path=wt,
        coral_md_path=wt / "AGENTS.md",
        log_dir=tmp_path / "logs",
        resume_session_id="sess-1",
    )

    cmd = _FakePopen.captured[0]["cmd"]
    # Resume default prompt — agent should not get a bare empty string.
    assert cmd[-1].startswith("Session resumed")


def test_start_ignores_unknown_runtime_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)

    wt = _make_worktree(tmp_path)
    runtime = CursorAgentRuntime()
    with caplog.at_level("WARNING"):
        runtime.start(
            worktree_path=wt,
            coral_md_path=wt / "AGENTS.md",
            log_dir=tmp_path / "logs",
            runtime_options={"bogus_flag": True},
            prompt="x",
        )

    cmd = _FakePopen.captured[0]["cmd"]
    assert "--bogus_flag" not in cmd
    assert "--bogus-flag" not in cmd
    assert any("bogus_flag" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# setup_cursor_settings — `.cursor/rules/coral.mdc` always-apply guardrails
# ---------------------------------------------------------------------------


def test_setup_cursor_settings_writes_alwaysapply_rule(tmp_path: Path) -> None:
    wt = tmp_path / "wt"
    wt.mkdir()
    coral_dir = tmp_path / ".coral"
    (coral_dir / "private").mkdir(parents=True)

    setup_cursor_settings(wt, coral_dir=coral_dir, research=True)

    rule = wt / ".cursor" / "rules" / "coral.mdc"
    assert rule.exists()
    text = rule.read_text()

    # Frontmatter must mark the rule alwaysApply so it survives context pressure
    assert text.startswith("---\n")
    assert "alwaysApply: true" in text.split("---\n", 2)[1]

    # Body must reference coral eval, AGENTS.md pointer, and private-dir guard
    assert "coral eval" in text
    assert "AGENTS.md" in text
    assert str((coral_dir / "private").resolve()) in text


def test_setup_cursor_settings_omits_research_line_when_disabled(tmp_path: Path) -> None:
    wt = tmp_path / "wt"
    wt.mkdir()
    coral_dir = tmp_path / ".coral"
    (coral_dir / "private").mkdir(parents=True)

    setup_cursor_settings(wt, coral_dir=coral_dir, research=False)

    text = (wt / ".cursor" / "rules" / "coral.mdc").read_text()
    assert "Web search and web fetch are disabled" in text


def test_setup_cursor_settings_includes_research_only_when_disabled(tmp_path: Path) -> None:
    wt = tmp_path / "wt"
    wt.mkdir()
    coral_dir = tmp_path / ".coral"
    (coral_dir / "private").mkdir(parents=True)

    setup_cursor_settings(wt, coral_dir=coral_dir, research=True)
    text = (wt / ".cursor" / "rules" / "coral.mdc").read_text()
    assert "Web search and web fetch are disabled" not in text


def test_setup_cursor_settings_overwrites_existing_rule(tmp_path: Path) -> None:
    wt = tmp_path / "wt"
    rules_dir = wt / ".cursor" / "rules"
    rules_dir.mkdir(parents=True)
    (rules_dir / "coral.mdc").write_text("stale content from a previous run\n")

    coral_dir = tmp_path / ".coral"
    (coral_dir / "private").mkdir(parents=True)

    setup_cursor_settings(wt, coral_dir=coral_dir, research=True)

    text = (rules_dir / "coral.mdc").read_text()
    assert "stale content" not in text
    assert "alwaysApply: true" in text


def test_setup_cursor_settings_accepts_gateway_kwargs(tmp_path: Path) -> None:
    """Cursor doesn't route through the gateway, but the kwargs must be accepted."""
    wt = tmp_path / "wt"
    wt.mkdir()
    coral_dir = tmp_path / ".coral"
    (coral_dir / "private").mkdir(parents=True)

    setup_cursor_settings(
        wt,
        coral_dir=coral_dir,
        research=True,
        gateway_url="http://localhost:4000",
        gateway_api_key="sk-test",
    )

    text = (wt / ".cursor" / "rules" / "coral.mdc").read_text()
    # Gateway args are intentionally ignored — must not leak into the rule
    assert "localhost:4000" not in text
    assert "sk-test" not in text
