"""Tests for CLI island scoping when invoked from an agent worktree.

The agent's CORAL.md (rendered at `coral/template/coral_md.py:108-115`) tells
the agent "you cannot see [other islands'] state directly. Each island
evolves independently." This file pins that contract for every read- and
write-side CLI command: when cwd has a `.coral_island` breadcrumb, the CLI
must scope to that island; otherwise it must aggregate across islands (the
"global" view for users outside any worktree).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import textwrap
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest

from coral.cli import heartbeat as heartbeat_module
from coral.cli import query as query_module
from coral.cli import start as start_module
from coral.cli._helpers import find_coral_dir_and_island
from coral.cli.eval import cmd_checkout, cmd_eval, cmd_wait
from coral.cli.heartbeat import (
    _cmd_heartbeat_remove,
    _cmd_heartbeat_reset,
    _cmd_heartbeat_set,
    _cmd_heartbeat_show,
)
from coral.cli.query import (
    cmd_log,
    cmd_notes,
    cmd_show,
    cmd_skills,
)
from coral.cli.start import cmd_status
from coral.hub.attempts import write_attempt
from coral.hub.auto_stop import write_auto_stop
from coral.hub.heartbeat import (
    write_global_heartbeat,
)
from coral.types import Attempt


def _write_note(coral_dir: Path, island: str, name: str, body: str) -> None:
    """Write a minimal frontmatter note under islands/<island>/notes/<name>.md."""
    (coral_dir / "islands" / island / "notes" / f"{name}.md").write_text(
        f"---\ncreator: test\ncreated: 2026-06-01T00:00:00Z\n---\n{body}\n"
    )


def _write_skill(coral_dir: Path, island: str, name: str, description: str) -> None:
    """Write a minimal SKILL.md under islands/<island>/skills/<name>/."""
    skill_dir = coral_dir / "islands" / island / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nbody\n"
    )


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _make_attempt(
    commit: str,
    island: str,
    *,
    agent: str | None = None,
    score: float = 0.5,
    title: str | None = None,
) -> Attempt:
    # Default: an attempt owned by an agent whose birth island matches.
    if agent is None:
        agent = f"{island}-agent-1"
    return Attempt(
        commit_hash=commit,
        agent_id=agent,
        title=title or f"attempt-island-{island}",
        score=score,
        status="improved",
        parent_hash=None,
        timestamp=datetime.now(UTC).isoformat(),
        metadata={"budget_class": "real", "island_id": island},
    )


@pytest.fixture
def multi_island_layout(tmp_path: Path) -> Path:
    """Create a multi-island run layout with attempts, notes, and skills on each island.

    Returns the ``coral_dir``. Each island has at least one attempt and one
    note so that any of the CLI commands we exercise has something to scope
    on. Public/ is intentionally empty — that's the state in a real run.
    """
    coral_dir = tmp_path / ".coral"
    for i in (0, 1, 2):
        island_dir = coral_dir / "islands" / str(i)
        (island_dir / "attempts").mkdir(parents=True)
        (island_dir / "notes").mkdir(parents=True)
        (island_dir / "skills").mkdir(parents=True)
        # Distinct, easy-to-grep markers so test assertions are precise.
        # The commit-hash first-8 is what surfaces in `coral log`'s table;
        # we use 40-char hex-looking hashes (so cmd_show / cmd_wait prefix
        # resolution exercises glob) but the assertions target the agent
        # id (which is the most robust scoping signal in the table).
        write_attempt(
            coral_dir,
            _make_attempt(f"{i:02d}0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a", str(i)),
            island_id=str(i),
        )
        _write_note(coral_dir, str(i), f"note-island-{i}", f"body for island {i}")
        _write_skill(coral_dir, str(i), f"skill-island-{i}", f"description for island {i}")
    # Stale top-level notes/attempts/skills dirs (should not surface when
    # scoped to an island).
    (coral_dir / "public" / "attempts").mkdir(parents=True)
    (coral_dir / "public" / "notes").mkdir(parents=True)
    (coral_dir / "public" / "skills").mkdir(parents=True)
    return coral_dir


@pytest.fixture
def worktree(tmp_path: Path, multi_island_layout: Path) -> Path:
    """A scratch worktree at ``tmp_path/wt`` pointing at the multi-island run.

    Writes ``.coral_dir`` and ``.coral_island=0`` breadcrumbs so the helper
    sees a worktree context pinned to island 0. The test parameterises which
    island the worktree claims by overwriting ``.coral_island`` after the
    fixture is in place.
    """
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".coral_dir").write_text(str(multi_island_layout))
    (wt / ".coral_island").write_text("0")
    (wt / ".coral_agent_id").write_text("0-agent-1")
    return wt


@contextmanager
def _chdir(path: Path):
    """Temporarily cd into ``path`` for the duration of a test."""
    orig = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig)


# --------------------------------------------------------------------------- #
# Step 1 helper: find_coral_dir_and_island                                   #
# --------------------------------------------------------------------------- #


def test_helper_in_worktree_returns_island(monkeypatch, tmp_path, multi_island_layout):
    """Both breadcrumbs present -> returns the island id from the breadcrumb."""
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".coral_dir").write_text(str(multi_island_layout))
    (wt / ".coral_island").write_text("1")
    with _chdir(wt):
        coral_dir, island_id = find_coral_dir_and_island()
    assert coral_dir.resolve() == multi_island_layout.resolve()
    assert island_id == "1"


def test_helper_in_worktree_without_island_returns_none(monkeypatch, tmp_path, multi_island_layout):
    """Worktree with only ``.coral_dir`` (e.g. single-island run) returns ``None``."""
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".coral_dir").write_text(str(multi_island_layout))
    with _chdir(wt):
        coral_dir, island_id = find_coral_dir_and_island()
    assert coral_dir.resolve() == multi_island_layout.resolve()
    assert island_id is None


def test_helper_walks_up_to_find_breadcrumbs(monkeypatch, tmp_path, multi_island_layout):
    """``coral log`` from a subdir of the worktree still finds the breadcrumb."""
    wt = tmp_path / "wt"
    sub = wt / "deep" / "nested"
    sub.mkdir(parents=True)
    (wt / ".coral_dir").write_text(str(multi_island_layout))
    (wt / ".coral_island").write_text("2")
    with _chdir(sub):
        coral_dir, island_id = find_coral_dir_and_island()
    assert coral_dir.resolve() == multi_island_layout.resolve()
    assert island_id == "2"


def test_helper_stale_breadcrumb_returns_none(monkeypatch, tmp_path, multi_island_layout):
    """If ``.coral_island`` points at a non-existent island, treat as unscoped."""
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".coral_dir").write_text(str(multi_island_layout))
    (wt / ".coral_island").write_text("9")  # no islands/9 on disk
    with _chdir(wt):
        coral_dir, island_id = find_coral_dir_and_island()
    assert coral_dir.resolve() == multi_island_layout.resolve()
    assert island_id is None


def test_helper_explicit_task_arg_defers(monkeypatch, tmp_path, multi_island_layout):
    """``--task other`` must override the worktree breadcrumb (no silent scope)."""
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".coral_dir").write_text(str(multi_island_layout))
    (wt / ".coral_island").write_text("0")
    with _chdir(wt):
        # Monkeypatch the find_coral_dir fallback so the test is hermetic
        # — without this we'd need a results/ tree on disk.
        monkeypatch.setattr(
            "coral.cli._helpers.find_coral_dir",
            lambda task=None, run=None: multi_island_layout,
        )
        coral_dir, island_id = find_coral_dir_and_island(task="other")
    assert coral_dir.resolve() == multi_island_layout.resolve()
    assert island_id is None


# --------------------------------------------------------------------------- #
# Step 2: cmd_log                                                             #
# --------------------------------------------------------------------------- #


def _run_cmd_log(capsys, **flags) -> str:
    args = argparse.Namespace(
        count=20,
        recent=False,
        agent=None,
        search=None,
        all=False,
        budget_class=None,
        task=None,
        run=None,
    )
    for k, v in flags.items():
        setattr(args, k, v)
    cmd_log(args)
    return capsys.readouterr().out


def test_log_in_worktree_scopes_to_agent_island(monkeypatch, multi_island_layout, worktree, capsys):
    """`coral log` from an island-0 worktree shows only island 0's attempts."""
    monkeypatch.setattr(
        query_module,
        "find_coral_dir_and_island",
        lambda task=None, run=None: (multi_island_layout, "0"),
    )
    monkeypatch.setattr(query_module, "read_direction", lambda _coral_dir: "maximize")
    with _chdir(worktree):
        out = _run_cmd_log(capsys)
    # Island 0's agent appears in the table.
    assert "0-agent-1" in out
    # Island 1 + 2 agents are NOT in the output.
    assert "1-agent-1" not in out
    assert "2-agent-1" not in out


def test_log_without_breadcrumb_aggregates_all_islands(monkeypatch, multi_island_layout, capsys):
    """Regression: ``coral log`` from a run dir (no breadcrumb) shows every island."""
    monkeypatch.setattr(
        query_module,
        "find_coral_dir_and_island",
        lambda task=None, run=None: (multi_island_layout, None),
    )
    monkeypatch.setattr(query_module, "read_direction", lambda _coral_dir: "maximize")
    out = _run_cmd_log(capsys)
    for i in (0, 1, 2):
        assert f"{i}-agent-1" in out


# --------------------------------------------------------------------------- #
# Step 3: cmd_wait                                                            #
# --------------------------------------------------------------------------- #


def test_eval_reads_agent_id_from_workdir_breadcrumb(monkeypatch, tmp_path, capsys):
    from coral.hooks import post_commit

    worktree = tmp_path / "wt"
    subdir = worktree / "deep"
    subdir.mkdir(parents=True)
    (worktree / ".coral_agent_id").write_text("0-agent-1")
    captured: dict[str, str] = {}

    def fake_submit_eval(**kwargs):
        captured["agent_id"] = kwargs["agent_id"]
        return Attempt(
            commit_hash="abc123",
            agent_id=kwargs["agent_id"],
            title="t",
            score=None,
            status="pending",
            parent_hash=None,
            timestamp=datetime.now(UTC).isoformat(),
        )

    monkeypatch.setattr(post_commit, "submit_eval", fake_submit_eval)

    cmd_eval(
        argparse.Namespace(
            agent=None,
            message="m",
            workdir=str(subdir),
            wait=False,
            timeout=None,
            tune=False,
        )
    )

    assert captured["agent_id"] == "0-agent-1"


def test_wait_resolves_hash_on_other_island_when_unscoped(monkeypatch, multi_island_layout, capsys):
    """Pre-fix bug: ``coral wait`` looked in public/attempts (empty in multi-island)
    and returned "No attempt matches" for hashes living on any island."""
    monkeypatch.setattr(
        "coral.cli.eval.find_coral_dir_and_island",
        lambda task=None, run=None: (multi_island_layout, None),
    )
    args = argparse.Namespace(
        hash="010a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a",
        workdir=str(multi_island_layout.parent),
        timeout=0.1,
        task=None,
        run=None,
    )
    # The attempt is already finalized (status="improved") so cmd_wait returns
    # immediately instead of polling until timeout.
    cmd_wait(args)
    out = capsys.readouterr().out
    assert "CORAL Wait" in out
    assert "0.5" in out  # score


def test_wait_in_island0_worktree_cannot_see_island1_attempt(
    monkeypatch, multi_island_layout, worktree, capsys
):
    """``coral wait`` from an island-0 worktree must not resolve a hash
    that lives only on island 1."""
    monkeypatch.setattr(
        "coral.cli.eval.find_coral_dir_and_island",
        lambda task=None, run=None: (multi_island_layout, "0"),
    )
    args = argparse.Namespace(
        hash="010a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a",
        workdir=str(worktree),
        timeout=0.1,
        task=None,
        run=None,
    )
    with pytest.raises(SystemExit):
        cmd_wait(args)
    err = capsys.readouterr()
    assert "No attempt matches" in (err.out + err.err)


# --------------------------------------------------------------------------- #
# Step 4: cmd_checkout                                                        #
# --------------------------------------------------------------------------- #


def _make_git_repo(path: Path) -> None:
    """Init a git repo with one commit at ``path``.

    The ``git config`` calls must run with ``cwd=str(path)`` (or use
    ``--local``) so the user.email/user.name land in the new repo's
    ``.git/config``. Without it, config writes to the test process's cwd
    or ``$HOME``, the commit inherits an empty identity from CI, and
    fails with "Author identity unknown".
    """
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "config", "--local", "user.email", "test@coral"], cwd=str(path), check=True
    )
    subprocess.run(["git", "config", "--local", "user.name", "test"], cwd=str(path), check=True)
    (path / "README.md").write_text("init")
    subprocess.run(["git", "add", "-A"], cwd=str(path), check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init", "--allow-empty"], cwd=str(path), check=True
    )


def test_checkout_resolves_hash_on_other_island_when_unscoped(
    monkeypatch, multi_island_layout, tmp_path, capsys
):
    """Pre-fix bug: ``coral checkout`` looked in public/attempts and bailed."""
    repo = tmp_path / "checkout_repo"
    _make_git_repo(repo)
    monkeypatch.setattr(
        "coral.cli.eval.find_coral_dir_and_island",
        lambda task=None, run=None: (multi_island_layout, None),
    )
    args = argparse.Namespace(
        hash="020a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a",
        workdir=str(repo),
        task=None,
        run=None,
    )
    # We only need to confirm the hash resolution works — git cat-file will
    # fail because that commit doesn't exist in the test repo, but the
    # important thing is we no longer bail with "No attempt matches".
    with pytest.raises(SystemExit):
        cmd_checkout(args)
    err = capsys.readouterr()
    assert "No attempt matches" not in (err.out + err.err)


def test_checkout_in_island0_worktree_cannot_see_island2_attempt(
    monkeypatch, multi_island_layout, worktree, capsys
):
    """``coral checkout`` from an island-0 worktree must not resolve an
    attempt that lives only on island 2."""
    monkeypatch.setattr(
        "coral.cli.eval.find_coral_dir_and_island",
        lambda task=None, run=None: (multi_island_layout, "0"),
    )
    args = argparse.Namespace(
        hash="020a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a",
        workdir=str(worktree),
        task=None,
        run=None,
    )
    cmd_checkout(args)
    err = capsys.readouterr()
    assert "No attempt matches" in (err.out + err.err)


# --------------------------------------------------------------------------- #
# Step 5: cmd_show                                                            #
# --------------------------------------------------------------------------- #


def test_show_in_worktree_scoped_to_island(monkeypatch, multi_island_layout, worktree, capsys):
    """``coral show <hash>`` from a worktree on island 0 must not resolve an
    attempt that lives only on island 1."""
    monkeypatch.setattr(
        query_module,
        "find_coral_dir_and_island",
        lambda task=None, run=None: (multi_island_layout, "0"),
    )
    args = argparse.Namespace(
        hash="10a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a",
        diff=False,
        task=None,
        run=None,
    )
    cmd_show(args)
    out = capsys.readouterr().out
    assert "not found" in out


def test_show_without_breadcrumb_resolves_anywhere(monkeypatch, multi_island_layout, capsys):
    """Regression: unscoped (no breadcrumb) ``coral show`` still finds any island's attempt."""
    monkeypatch.setattr(
        query_module,
        "find_coral_dir_and_island",
        lambda task=None, run=None: (multi_island_layout, None),
    )
    args = argparse.Namespace(
        hash="010a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a",
        diff=False,
        task=None,
        run=None,
    )
    # The attempt is finalized, so cmd_show prints Commit/Agent/etc.
    cmd_show(args)
    out = capsys.readouterr().out
    assert "Commit:" in out
    assert "1-agent-1" in out


def test_show_renders_meta_evolve_attribution(monkeypatch, multi_island_layout, capsys):
    commit = "010a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a"
    attempt = _make_attempt(commit, "1")
    attempt.metadata["meta_evolve"] = {
        "operator": "prompt",
        "mutation": "rewrite",
    }
    write_attempt(multi_island_layout, attempt, island_id="1")
    monkeypatch.setattr(
        query_module,
        "find_coral_dir_and_island",
        lambda task=None, run=None: (multi_island_layout, None),
    )

    cmd_show(
        argparse.Namespace(
            hash=commit,
            diff=False,
            task=None,
            run=None,
        )
    )

    assert "Meta:    prompt / rewrite" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Step 6: cmd_status                                                          #
# --------------------------------------------------------------------------- #


def test_status_in_worktree_only_lists_island_attempts(
    monkeypatch, multi_island_layout, worktree, capsys
):
    """``coral status`` from an island-0 worktree should only show
    island 0's attempts in the leaderboard."""
    # Stub the picker so the test doesn't try to interactively pick a run.
    monkeypatch.setattr(start_module, "pick_run", lambda: multi_island_layout)
    monkeypatch.setattr(
        start_module,
        "find_coral_dir_and_island",
        lambda task=None, run=None: (multi_island_layout, "0"),
    )
    # No manager pid file -> "not running" branch
    (multi_island_layout / "public" / "manager.pid").parent.mkdir(parents=True, exist_ok=True)
    island0_logs = multi_island_layout / "islands" / "0" / "logs"
    island0_logs.mkdir(parents=True)
    (island0_logs / "0-agent-1.1.log").write_text("current island 0 agent\n")
    (island0_logs / "0-agent-2.1.log").write_text("stale pre-migration log\n")
    agents_dir = multi_island_layout.parent / "agents"
    for agent_id, current_island in {
        "0-agent-1": "0",
        "0-agent-2": "1",
        "1-agent-1": "1",
        "2-agent-1": "2",
    }.items():
        agent_dir = agents_dir / agent_id
        agent_dir.mkdir(parents=True)
        (agent_dir / ".coral_island").write_text(current_island)
    # cd into a subdir to ensure command-level discovery walks up to the
    # worktree breadcrumbs before falling back to the run picker.
    subdir = worktree / "deep" / "nested"
    subdir.mkdir(parents=True)
    with _chdir(subdir):
        args = argparse.Namespace(task=None, run=None)
        cmd_status(args)
    out = capsys.readouterr().out
    # The leaderboard should mention island 0's agent, not islands 1 or 2.
    assert "0-agent-1" in out
    assert "0-agent-2" not in out
    assert "1-agent-1" not in out
    assert "2-agent-1" not in out
    assert "Agents: 1" in out


def test_stop_from_worktree_subdir_uses_breadcrumb_run(monkeypatch, multi_island_layout, worktree):
    """``coral stop`` from a worktree subdir should not fall through to the picker."""
    stopped: list[Path] = []
    monkeypatch.setattr(start_module, "pick_run", lambda *args, **kwargs: pytest.fail("picker"))
    monkeypatch.setattr(start_module, "_stop_one", lambda coral_dir: stopped.append(coral_dir))

    subdir = worktree / "deep" / "nested"
    subdir.mkdir(parents=True)
    with _chdir(subdir):
        start_module.cmd_stop(argparse.Namespace(task=None, run=None, all=False))

    assert stopped == [multi_island_layout.resolve()]


def test_resume_from_worktree_subdir_uses_breadcrumb_run(monkeypatch, tmp_path):
    """``coral resume`` from a worktree subdir should resume that run directly."""
    from coral.agent import manager as manager_module

    run_dir = tmp_path / "results" / "task" / "run"
    coral_dir = run_dir / ".coral"
    (coral_dir / "public").mkdir(parents=True)
    (coral_dir / "config.yaml").write_text(
        textwrap.dedent(
            """
            task:
              name: task
              description: resume test
            agents:
              count: 2
            islands:
              count: 2
            run:
              session: local
              ui: false
            workspace:
              results_dir: results
            """
        ).strip()
    )
    worktree = tmp_path / "wt"
    subdir = worktree / "deep" / "nested"
    subdir.mkdir(parents=True)
    (worktree / ".coral_dir").write_text(str(coral_dir))
    (worktree / ".coral_island").write_text("0")

    resumed: list[tuple[Path, str | None, str | None]] = []

    class DummyManager:
        def __init__(self, *_args, **_kwargs):
            pass

        def resume_all(self, paths, instruction=None, resume_from=None):
            resumed.append((paths.coral_dir, instruction, resume_from))
            return [argparse.Namespace(agent_id="0-agent-1", process=None, session_id=None)]

        def monitor_loop(self):
            raise StopIteration

    monkeypatch.setattr(start_module, "pick_run", lambda *args, **kwargs: pytest.fail("picker"))
    monkeypatch.setattr(manager_module, "AgentManager", DummyManager)

    with _chdir(subdir), pytest.raises(StopIteration):
        start_module.cmd_resume(
            argparse.Namespace(
                task=None,
                run=None,
                overrides=[],
                instruction="try SIMD",
                resume_from="abc123",
            )
        )

    assert resumed == [(coral_dir.resolve(), "try SIMD", "abc123")]


def test_status_default_hides_tune_attempts(monkeypatch, tmp_path, capsys):
    """When all attempts are tune-mode, default ``coral status`` should
    not surface them in the body summary or leaderboard — only the
    per-agent counter line shows the tune count (the per-agent counter
    is intentionally not gated by include_tune so operators can see the
    team is alive even when no real attempt has landed yet)."""
    coral_dir = tmp_path / ".coral"
    for i in (0, 1):
        island_dir = coral_dir / "islands" / str(i)
        (island_dir / "attempts").mkdir(parents=True)
        # A per-agent log file is what triggers the per-agent counter
        # section in cmd_status; without one, the whole agents block is
        # suppressed and the tune=1 line never gets a chance to render.
        (island_dir / "logs").mkdir(parents=True)
        (island_dir / "logs" / f"{i}-agent-1.1.log").write_text("dummy\n")
        write_attempt(
            coral_dir,
            Attempt(
                commit_hash=f"{i:02d}tune0000000000000000000000000000000000",
                agent_id=f"{i}-agent-1",
                title=f"tune-attempt-island-{i}",
                score=0.0,
                status="improved",
                parent_hash=None,
                timestamp=datetime.now(UTC).isoformat(),
                metadata={"budget_class": "tune", "island_id": str(i)},
            ),
            island_id=str(i),
        )
    monkeypatch.setattr(start_module, "pick_run", lambda: coral_dir)
    monkeypatch.setattr(
        start_module,
        "find_coral_dir_and_island",
        lambda task=None, run=None: (coral_dir, None),
    )
    cmd_status(argparse.Namespace(task=None, run=None, all=False))
    out = capsys.readouterr().out
    # Body summary hides tune — operator sees "No attempts yet." for the
    # headline even though agents are clearly busy (the per-agent counter
    # line below still surfaces the activity).
    assert "No attempts yet." in out
    # But the per-agent counter row IS rendered and shows the tune count.
    assert "tune=1" in out
    # And the actual tune attempt title is hidden from both the body
    # summary and the leaderboard.
    assert "tune-attempt-island-0" not in out
    assert "tune-attempt-island-1" not in out
    assert "## Leaderboard" not in out


def test_status_shows_auto_stop_reason(monkeypatch, tmp_path, capsys):
    coral_dir = tmp_path / ".coral"
    (coral_dir / "public" / "attempts").mkdir(parents=True)
    write_auto_stop(
        coral_dir,
        {
            "reason": "max_real_attempts",
            "timestamp": "2026-06-24T00:00:00+00:00",
            "attempt_id": "abc",
            "agent_id": "agent-1",
            "score": 0.4,
            "score_threshold": None,
            "direction": "maximize",
            "real_attempt_count": 30,
            "max_real_attempts": 30,
        },
    )
    monkeypatch.setattr(start_module, "pick_run", lambda: coral_dir)
    monkeypatch.setattr(
        start_module,
        "find_coral_dir_and_island",
        lambda task=None, run=None: (coral_dir, None),
    )

    cmd_status(argparse.Namespace(task=None, run=None, all=False))

    out = capsys.readouterr().out
    assert "Auto-stop: max real attempts reached" in out
    assert "real_attempts=30" in out
    assert "max=30" in out


def test_status_all_shows_tune_attempts(monkeypatch, tmp_path, capsys):
    """``coral status --all`` should include tune attempts in the body
    summary and the leaderboard — the operator's escape hatch for the
    'team is alive but only on tune mode' case."""
    coral_dir = tmp_path / ".coral"
    (coral_dir / "islands" / "0" / "attempts").mkdir(parents=True)
    write_attempt(
        coral_dir,
        Attempt(
            commit_hash="00tune0000000000000000000000000000000a0a",
            agent_id="0-agent-1",
            title="tune-baseline-seed",
            score=1829.16,
            status="improved",
            parent_hash=None,
            timestamp=datetime.now(UTC).isoformat(),
            metadata={"budget_class": "tune", "island_id": "0"},
        ),
        island_id="0",
    )
    monkeypatch.setattr(start_module, "pick_run", lambda: coral_dir)
    monkeypatch.setattr(
        start_module,
        "find_coral_dir_and_island",
        lambda task=None, run=None: (coral_dir, None),
    )
    cmd_status(argparse.Namespace(task=None, run=None, all=True))
    out = capsys.readouterr().out
    # Body summary now sees the tune attempt and surfaces the score.
    assert "tune-baseline-seed" in out
    assert "1829" in out
    # Leaderboard table is also rendered (was suppressed when the
    # filtered attempt set was empty in the default path).
    assert "## Leaderboard" in out


# --------------------------------------------------------------------------- #
# Step 7: cmd_notes                                                           #
# --------------------------------------------------------------------------- #


def test_notes_in_worktree_only_lists_island_notes(
    monkeypatch, multi_island_layout, worktree, capsys
):
    """``coral notes`` from an island-0 worktree shows only island 0's notes."""
    monkeypatch.setattr(
        query_module,
        "find_coral_dir_and_island",
        lambda task=None, run=None: (multi_island_layout, "0"),
    )
    args = argparse.Namespace(
        history=False,
        diff=None,
        read=None,
        search=None,
        recent=None,
        task=None,
        run=None,
    )
    cmd_notes(args)
    out = capsys.readouterr().out
    # The note's display title is derived from the file stem
    # (`note-island-0` -> "Note Island 0") — that's the robust signal here.
    assert "Note Island 0" in out
    assert "Note Island 1" not in out
    assert "Note Island 2" not in out


# --------------------------------------------------------------------------- #
# Step 8: cmd_skills                                                          #
# --------------------------------------------------------------------------- #


def test_skills_in_worktree_only_lists_island_skills(
    monkeypatch, multi_island_layout, worktree, capsys
):
    """``coral skills`` from an island-0 worktree shows only island 0's skills."""
    monkeypatch.setattr(
        query_module,
        "find_coral_dir_and_island",
        lambda task=None, run=None: (multi_island_layout, "0"),
    )
    args = argparse.Namespace(read=None, task=None, run=None)
    cmd_skills(args)
    out = capsys.readouterr().out
    assert "skill-island-0" in out
    assert "skill-island-1" not in out
    assert "skill-island-2" not in out


# --------------------------------------------------------------------------- #
# Step 9: cmd_heartbeat (write side)                                          #
# --------------------------------------------------------------------------- #


def test_heartbeat_mutations_require_island_scope(monkeypatch, multi_island_layout, capsys):
    """Unscoped multi-island heartbeat writes fail clearly instead of crashing."""
    monkeypatch.setattr(
        heartbeat_module,
        "find_coral_dir_and_island",
        lambda task=None, run=None: (multi_island_layout, None),
    )
    cases = [
        (
            _cmd_heartbeat_set,
            argparse.Namespace(
                name="custom-rotate",
                every=3,
                prompt="rotate strategy",
                is_global=True,
                trigger=None,
                epsilon=None,
                task=None,
                run=None,
            ),
        ),
        (_cmd_heartbeat_remove, argparse.Namespace(name="custom-rotate", task=None, run=None)),
        (_cmd_heartbeat_reset, argparse.Namespace(task=None, run=None)),
    ]

    for command, args in cases:
        with pytest.raises(SystemExit):
            command(args)

    err = capsys.readouterr().err
    assert "must be made from an agent worktree" in err


def test_heartbeat_set_global_writes_to_worktree_island(monkeypatch, multi_island_layout, worktree):
    """``coral heartbeat set --global <name> --every N`` from an island-1
    worktree must write to ``islands/1/heartbeat/_global.json`` — not
    ``islands/0/`` (the first sorted island) and not aggregate across."""
    (worktree / ".coral_island").write_text("1")
    monkeypatch.setattr(
        heartbeat_module,
        "find_coral_dir_and_island",
        lambda task=None, run=None: (multi_island_layout, "1"),
    )
    args = argparse.Namespace(
        name="custom-rotate",
        every=3,
        prompt="rotate strategy",
        is_global=True,
        trigger=None,
        epsilon=None,
        task=None,
        run=None,
    )
    _cmd_heartbeat_set(args)

    # The file landed under the worktree's island (1), not 0 or 2.
    island1 = multi_island_layout / "islands" / "1" / "heartbeat" / "_global.json"
    assert island1.exists(), f"Expected global heartbeat at {island1}"
    data = json.loads(island1.read_text())["actions"]
    assert any(a["name"] == "custom-rotate" for a in data)
    # No cross-island bleed: island 0 and 2 must NOT have the new action.
    for other in (0, 2):
        other_file = multi_island_layout / "islands" / str(other) / "heartbeat" / "_global.json"
        if other_file.exists():
            other_data = json.loads(other_file.read_text())["actions"]
            assert not any(a["name"] == "custom-rotate" for a in other_data)


def test_heartbeat_set_local_writes_to_worktree_island(monkeypatch, multi_island_layout, worktree):
    """``coral heartbeat set <name> --every N`` (local scope) from an
    island-2 worktree writes the per-agent heartbeat under ``islands/2/``."""
    (worktree / ".coral_island").write_text("2")
    monkeypatch.setattr(
        heartbeat_module,
        "find_coral_dir_and_island",
        lambda task=None, run=None: (multi_island_layout, "2"),
    )
    args = argparse.Namespace(
        name="custom-local",
        every=2,
        prompt="local action",
        is_global=False,
        trigger=None,
        epsilon=None,
        task=None,
        run=None,
    )
    # chdir into the worktree so the per-agent file is named after its
    # .coral_agent_id breadcrumb.
    with _chdir(worktree):
        _cmd_heartbeat_set(args)

    local_file = multi_island_layout / "islands" / "2" / "heartbeat" / "0-agent-1.json"
    assert local_file.exists()
    data = json.loads(local_file.read_text())["actions"]
    assert any(a["name"] == "custom-local" for a in data)


def test_heartbeat_show_only_prints_worktree_island_actions(
    monkeypatch, multi_island_layout, worktree, capsys
):
    """``coral heartbeat`` from an island-1 worktree must not print
    actions that exist only on island 0."""
    write_global_heartbeat(
        multi_island_layout,
        [{"name": "island0-only", "every": 5, "trigger": "interval", "prompt": "x"}],
        island_id="0",
    )
    write_global_heartbeat(
        multi_island_layout,
        [{"name": "island1-only", "every": 7, "trigger": "interval", "prompt": "y"}],
        island_id="1",
    )
    (worktree / ".coral_island").write_text("1")
    monkeypatch.setattr(
        heartbeat_module,
        "find_coral_dir_and_island",
        lambda task=None, run=None: (multi_island_layout, "1"),
    )
    args = argparse.Namespace(task=None, run=None)
    _cmd_heartbeat_show(args)
    out = capsys.readouterr().out
    assert "island1-only" in out
    assert "island0-only" not in out


def test_heartbeat_remove_does_not_clobber_other_island(
    monkeypatch, multi_island_layout, worktree, capsys
):
    """``coral heartbeat remove foo`` from an island-1 worktree must
    only mutate ``islands/1/heartbeat/_global.json`` — island 0's copy
    stays intact."""
    write_global_heartbeat(
        multi_island_layout,
        [{"name": "shared", "every": 5, "trigger": "interval", "prompt": "x"}],
        island_id="0",
    )
    write_global_heartbeat(
        multi_island_layout,
        [{"name": "shared", "every": 7, "trigger": "interval", "prompt": "y"}],
        island_id="1",
    )
    (worktree / ".coral_island").write_text("1")
    monkeypatch.setattr(
        heartbeat_module,
        "find_coral_dir_and_island",
        lambda task=None, run=None: (multi_island_layout, "1"),
    )
    args = argparse.Namespace(name="shared", task=None, run=None)
    _cmd_heartbeat_remove(args)

    # Island 0's copy is untouched.
    island0 = json.loads(
        (multi_island_layout / "islands" / "0" / "heartbeat" / "_global.json").read_text()
    )["actions"]
    assert any(a["name"] == "shared" and a["every"] == 5 for a in island0)
    # Island 1's copy no longer has the action.
    island1_file = multi_island_layout / "islands" / "1" / "heartbeat" / "_global.json"
    if island1_file.exists():
        island1 = json.loads(island1_file.read_text())["actions"]
        assert not any(a["name"] == "shared" for a in island1)


def test_heartbeat_reset_only_resets_worktree_island(
    monkeypatch, multi_island_layout, worktree, capsys
):
    """``coral heartbeat reset`` from an island-2 worktree must only
    overwrite ``islands/2/heartbeat/_global.json``."""
    # Pre-populate island 0 with a custom action; it must survive a reset
    # issued from a different island's worktree.
    write_global_heartbeat(
        multi_island_layout,
        [{"name": "keep-me", "every": 9, "trigger": "interval", "prompt": "p"}],
        island_id="0",
    )
    (worktree / ".coral_island").write_text("2")
    # Seed a config.yaml so reset can read default_global_actions.
    (multi_island_layout / "config.yaml").write_text(
        textwrap.dedent(
            """
            task:
              name: t
              description: reset test
            agents:
              count: 1
            grader:
              direction: maximize
            islands:
              count: 3
            """
        ).strip()
    )
    monkeypatch.setattr(
        heartbeat_module,
        "find_coral_dir_and_island",
        lambda task=None, run=None: (multi_island_layout, "2"),
    )
    args = argparse.Namespace(task=None, run=None)
    _cmd_heartbeat_reset(args)

    island0 = json.loads(
        (multi_island_layout / "islands" / "0" / "heartbeat" / "_global.json").read_text()
    )["actions"]
    assert any(a["name"] == "keep-me" for a in island0), "island 0's custom action was clobbered"
