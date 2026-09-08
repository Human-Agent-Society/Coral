"""`coral start` must persist the session mode the user actually asked for.

The tmux/docker wrappers relaunch an inner process with ``run.session=local``
(to avoid recursion) and pass ``--wrapped-session`` so the inner process can
restore the real mode into the saved config. That restore must key on the
explicit marker — NOT ``in_tmux()``/``in_docker()`` — otherwise a user who runs
``run.session=local`` from their own tmux session or container silently gets
``tmux``/``docker`` persisted, and ``coral resume`` re-launches in the wrong
wrapper.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import coral.cli.start as start_mod
from coral.config import CoralConfig


def _drive_cmd_start(
    monkeypatch,
    tmp_path,
    *,
    session,
    wrapped_session,
    in_tmux,
    in_docker,
    tmux_session_name=None,
):
    """Run cmd_start through the real restore path, returning the config the
    manager was constructed with."""
    base = CoralConfig()
    base.task.name = "t"
    base.task.description = "t"
    base.run.session = session

    monkeypatch.setattr(start_mod.CoralConfig, "from_yaml", classmethod(lambda cls, path: base))
    monkeypatch.setattr(start_mod, "in_tmux", lambda: in_tmux)
    monkeypatch.setattr(start_mod, "in_docker", lambda: in_docker)
    monkeypatch.setattr(start_mod, "has_tmux", lambda: True)
    monkeypatch.setattr(start_mod, "_current_tmux_session_name", lambda: tmux_session_name)
    monkeypatch.setattr("coral.cli.validation.validate_task", lambda task_dir: [])

    captured = {}

    class FakeManager:
        def __init__(self, config, verbose=False, config_dir=None):
            captured["config"] = config
            self.specs = []
            self.paths = SimpleNamespace(run_dir=tmp_path, coral_dir=tmp_path / ".coral")
            (self.paths.coral_dir / "public").mkdir(parents=True, exist_ok=True)

        def start_all(self):
            return []

        def monitor_loop(self):
            pass

        def wait_for_completion(self):
            pass

    monkeypatch.setattr("coral.agent.manager.AgentManager", FakeManager)

    args = SimpleNamespace(
        config=str(tmp_path / "task.yaml"),
        overrides=[],
        wrapped_session=wrapped_session,
    )
    start_mod.cmd_start(args)
    return captured["config"]


def test_local_from_users_own_tmux_stays_local(monkeypatch, tmp_path):
    """The reported bug: run.session=local launched from inside the user's own
    tmux (no wrapper) must NOT be rewritten to tmux."""
    config = _drive_cmd_start(
        monkeypatch,
        tmp_path,
        session="local",
        wrapped_session=None,
        in_tmux=True,
        in_docker=False,
    )
    assert config.run.session == "local"


def test_tmux_wrapper_restores_tmux(monkeypatch, tmp_path):
    """The inner process launched by the tmux wrapper (session=local +
    --wrapped-session tmux) restores tmux so resume re-launches in tmux."""
    config = _drive_cmd_start(
        monkeypatch,
        tmp_path,
        session="local",
        wrapped_session="tmux",
        in_tmux=True,
        in_docker=False,
    )
    assert config.run.session == "tmux"


def test_docker_wrapper_restores_docker(monkeypatch, tmp_path):
    config = _drive_cmd_start(
        monkeypatch,
        tmp_path,
        session="local",
        wrapped_session="docker",
        in_tmux=False,
        in_docker=True,
    )
    assert config.run.session == "docker"


def test_plain_local_stays_local(monkeypatch, tmp_path):
    config = _drive_cmd_start(
        monkeypatch,
        tmp_path,
        session="local",
        wrapped_session=None,
        in_tmux=False,
        in_docker=False,
    )
    assert config.run.session == "local"


@pytest.mark.parametrize("mode", ["tmux", "docker"])
def test_wrappers_pass_wrapped_session_flag(mode):
    """The wrappers must actually emit --wrapped-session <mode> so the inner
    process can restore it. Guards against a wrapper dropping the marker."""
    import inspect

    if mode == "tmux":
        src = inspect.getsource(start_mod._build_coral_command)
    else:
        src = inspect.getsource(start_mod._start_in_docker)
    assert "--wrapped-session" in src
    assert f'"{mode}"' in src


@pytest.mark.parametrize(
    ("session_name", "expect_owned"),
    [("coral-mytask-20260906", True), ("users-own-session", False)],
)
def test_start_in_tmux_saves_session_marker(monkeypatch, tmp_path, session_name, expect_owned):
    """cmd_start inside tmux persists the session marker (and the owned flag
    exactly when coral created the session) without touching a real tmux."""
    _drive_cmd_start(
        monkeypatch,
        tmp_path,
        session="local",
        wrapped_session="tmux",
        in_tmux=True,
        in_docker=False,
        tmux_session_name=session_name,
    )
    public = tmp_path / ".coral" / "public"
    assert (public / ".coral_tmux_session").read_text(encoding="utf-8") == session_name
    assert (public / ".coral_tmux_owned").exists() is expect_owned


def test_start_in_tmux_skips_marker_when_probe_fails(monkeypatch, tmp_path):
    """A failing tmux probe (no server, tmux missing) must skip the marker,
    not crash cmd_start."""
    _drive_cmd_start(
        monkeypatch,
        tmp_path,
        session="local",
        wrapped_session=None,
        in_tmux=True,
        in_docker=False,
        tmux_session_name=None,
    )
    assert not (tmp_path / ".coral" / "public" / ".coral_tmux_session").exists()
