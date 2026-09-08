"""Tests for eval implementation and Claude Code settings."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

from coral.grader.daemon import process_pending_once
from coral.hooks.post_commit import (
    _increment_eval_count,
    submit_eval,
)
from coral.workspace import setup_claude_settings


def _submit_and_grade(message: str, agent_id: str, workdir: str):
    """Submit a pending attempt, then synchronously drain the grader queue.

    Mirrors the production flow (submit + async grader + wait) without
    needing to spawn a separate grader daemon process in tests.
    """
    from coral.hooks.post_commit import _find_coral_dir
    from coral.hub.attempts import read_attempt, read_eval_count

    # Stage+commit+write pending; no wait because there's no daemon running.
    pending = submit_eval(message=message, agent_id=agent_id, workdir=workdir, wait=False)

    coral_dir = _find_coral_dir(Path(workdir).resolve())
    assert coral_dir is not None
    process_pending_once(coral_dir)

    final = read_attempt(coral_dir, pending.commit_hash)
    assert final is not None
    try:
        final._eval_count = read_eval_count(coral_dir)  # type: ignore[attr-defined]
    except Exception:
        pass
    return final


def _setup_repo_with_config(base_dir: Path) -> Path:
    """Create a git repo with .coral/config.yaml wired to an entrypoint grader.

    No real grader venv: `grader_venv/bin/python` is a shell wrapper around
    the test interpreter with PYTHONPATH pointing at `private/grader_pkg/`,
    where the testgrader module lives.
    """
    repo = base_dir / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@test.com"], capture_output=True
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], capture_output=True)

    # Create a file and .gitignore, then make an initial commit
    (repo / "hello.py").write_text("print('hello')\n")
    (repo / ".gitignore").write_text(".coral/\n.coral_dir\n.claude/\n.coral_agent_id\nCLAUDE.md\n")
    subprocess.run(["git", "-C", str(repo), "add", "hello.py", ".gitignore"], capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "Initial"], capture_output=True, check=True
    )

    # Set up .coral directory with config + entrypoint grader package
    coral_dir = repo / ".coral"
    coral_dir.mkdir()
    (coral_dir / "public" / "attempts").mkdir(parents=True)

    # Write .coral_dir breadcrumb (as write_coral_dir does)
    (repo / ".coral_dir").write_text(str(coral_dir.resolve()))

    pkg_dir = coral_dir / "private" / "grader_pkg"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "testgrader.py").write_text(
        "from coral.grader.task_grader import TaskGrader\n"
        "class Grader(TaskGrader):\n"
        "    def evaluate(self):\n"
        "        return 0.75\n"
    )
    bin_dir = coral_dir / "private" / "grader_venv" / "bin"
    bin_dir.mkdir(parents=True)
    wrapper = bin_dir / "python"
    wrapper.write_text(
        "#!/bin/sh\n"
        f'export PYTHONPATH="{pkg_dir}${{PYTHONPATH:+:$PYTHONPATH}}"\n'
        f'exec "{sys.executable}" "$@"\n'
    )
    wrapper.chmod(0o755)

    config = {
        "task": {"name": "test_task", "description": "A test"},
        "grader": {"entrypoint": "testgrader:Grader"},
        "agents": {"count": 1},
        "sharing": {"attempts": True, "notes": True, "skills": True},
        "workspace": {"base_dir": str(repo), "repo_path": str(repo)},
    }
    with open(coral_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    return repo


def test_submit_eval_pending_then_graded():
    """submit_eval writes a pending record; daemon finalizes it with a score."""
    import sys

    with tempfile.TemporaryDirectory() as d:
        repo = _setup_repo_with_config(Path(d))

        (repo / "hello.py").write_text("print('hello world')\n")

        sys.path.insert(0, str(repo))
        try:
            # Stage without running grader yet.
            pending = submit_eval(
                message="Update hello message",
                agent_id="agent-test",
                workdir=str(repo),
                wait=False,
            )
            assert pending.status == "pending"
            assert pending.score is None

            attempt_file = repo / ".coral" / "public" / "attempts" / f"{pending.commit_hash}.json"
            assert attempt_file.exists()
            pending_data = json.loads(attempt_file.read_text())
            assert pending_data["status"] == "pending"
            assert pending_data["score"] is None

            # Drain the grader queue synchronously.
            process_pending_once(repo / ".coral")

            final_data = json.loads(attempt_file.read_text())
            assert final_data["score"] == 0.75
            assert final_data["status"] == "improved"
            assert final_data["commit_hash"] == pending.commit_hash
        finally:
            sys.path.pop(0)


def test_submit_eval_no_changes():
    """submit_eval should fail if there are no changes to commit."""
    import sys

    with tempfile.TemporaryDirectory() as d:
        repo = _setup_repo_with_config(Path(d))

        sys.path.insert(0, str(repo))
        try:
            submit_eval(
                message="No changes",
                agent_id="agent-test",
                workdir=str(repo),
                wait=False,
            )
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "Nothing to commit" in str(e)
        finally:
            sys.path.pop(0)


def test_eval_count_and_reflection():
    """Test that eval count increments."""
    with tempfile.TemporaryDirectory() as d:
        coral_dir = Path(d)
        (coral_dir / "public").mkdir()

        # Counter starts at 0, increments to 1
        assert _increment_eval_count(coral_dir) == 1
        assert _increment_eval_count(coral_dir) == 2
        assert _increment_eval_count(coral_dir) == 3

        # Check file contents
        assert (coral_dir / "public" / "eval_count").read_text() == "3"


def test_submit_eval_tracks_eval_count():
    """Integration: daemon bumps the eval counter when finalizing attempts."""
    import sys

    with tempfile.TemporaryDirectory() as d:
        repo = _setup_repo_with_config(Path(d))

        sys.path.insert(0, str(repo))
        try:
            (repo / "hello.py").write_text("print('v1')\n")
            a1 = _submit_and_grade("v1", "agent-test", str(repo))
            assert getattr(a1, "_eval_count", None) == 1

            (repo / "hello.py").write_text("print('v2')\n")
            a2 = _submit_and_grade("v2", "agent-test", str(repo))
            assert getattr(a2, "_eval_count", None) == 2
        finally:
            sys.path.pop(0)


def _set_grader_config(repo: Path, **fields) -> None:
    """Rewrite .coral/config.yaml's grader section with the given overrides."""
    cfg_path = repo / ".coral" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("grader", {}).update(fields)
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)


def _enable_meta_evolve(repo: Path) -> None:
    cfg_path = repo / ".coral" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("agents", {})["meta_evolve"] = {
        "enabled": True,
        "arms": [
            {"operator": "prompt", "mutation": "rewrite"},
            {"operator": "implementation", "mutation": "replace"},
        ],
    }
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)


def _head_hash(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_submit_eval_rejects_unknown_meta_arm_before_commit():
    with tempfile.TemporaryDirectory() as d:
        repo = _setup_repo_with_config(Path(d))
        _enable_meta_evolve(repo)
        (repo / "hello.py").write_text("print('changed')\n")
        head_before = _head_hash(repo)

        with pytest.raises(RuntimeError, match="configured arm"):
            submit_eval(
                message="unknown arm",
                agent_id="agent-test",
                workdir=str(repo),
                wait=False,
                operator="prompt",
                mutation="unknown",
            )

        assert _head_hash(repo) == head_before
        assert (repo / "hello.py").read_text() == "print('changed')\n"


def test_eval_cli_writes_meta_evolve_attribution():
    with tempfile.TemporaryDirectory() as d:
        repo = _setup_repo_with_config(Path(d))
        _enable_meta_evolve(repo)
        (repo / "hello.py").write_text("print('changed')\n")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "coral.cli",
                "eval",
                "--no-wait",
                "--workdir",
                str(repo),
                "--agent",
                "agent-test",
                "--operator",
                "prompt",
                "--mutation",
                "rewrite",
                "-m",
                "attribute attempt",
            ],
            capture_output=True,
            check=False,
            text=True,
        )

        assert result.returncode == 0, result.stderr
        assert "Meta:    prompt / rewrite" in result.stdout
        attempt_files = list((repo / ".coral" / "public" / "attempts").glob("*.json"))
        assert len(attempt_files) == 1
        data = json.loads(attempt_files[0].read_text())
        assert data["metadata"]["meta_evolve"] == {
            "operator": "prompt",
            "mutation": "rewrite",
        }


def test_submit_eval_rejects_when_agent_at_pending_limit():
    """Default cap is 1: a second submit while the first is pending must raise
    and must not create a new commit."""
    import sys

    with tempfile.TemporaryDirectory() as d:
        repo = _setup_repo_with_config(Path(d))
        sys.path.insert(0, str(repo))
        try:
            (repo / "hello.py").write_text("print('v1')\n")
            first = submit_eval(
                message="v1",
                agent_id="agent-test",
                workdir=str(repo),
                wait=False,
            )
            assert first.status == "pending"
            head_after_first = _head_hash(repo)

            # Second submit while first is still pending — must reject.
            (repo / "hello.py").write_text("print('v2')\n")
            with pytest.raises(RuntimeError, match=r"pending attempt"):
                submit_eval(
                    message="v2",
                    agent_id="agent-test",
                    workdir=str(repo),
                    wait=False,
                )

            # No orphan commit was created by the rejected submit.
            assert _head_hash(repo) == head_after_first
        finally:
            sys.path.pop(0)


def test_submit_eval_allows_after_drain():
    """After the daemon grades the pending attempt, a new submit succeeds."""
    import sys

    with tempfile.TemporaryDirectory() as d:
        repo = _setup_repo_with_config(Path(d))
        sys.path.insert(0, str(repo))
        try:
            (repo / "hello.py").write_text("print('v1')\n")
            _submit_and_grade("v1", "agent-test", str(repo))

            (repo / "hello.py").write_text("print('v2')\n")
            second = submit_eval(
                message="v2",
                agent_id="agent-test",
                workdir=str(repo),
                wait=False,
            )
            assert second.status == "pending"
        finally:
            sys.path.pop(0)


def test_submit_eval_respects_higher_limit():
    """grader.max_pending_per_agent: 3 lets three pending stack, rejects the fourth."""
    import sys

    with tempfile.TemporaryDirectory() as d:
        repo = _setup_repo_with_config(Path(d))
        _set_grader_config(repo, max_pending_per_agent=3)

        sys.path.insert(0, str(repo))
        try:
            for i in range(3):
                (repo / "hello.py").write_text(f"print('v{i}')\n")
                submit_eval(
                    message=f"v{i}",
                    agent_id="agent-test",
                    workdir=str(repo),
                    wait=False,
                )

            (repo / "hello.py").write_text("print('overflow')\n")
            with pytest.raises(RuntimeError, match=r"pending attempt"):
                submit_eval(
                    message="overflow",
                    agent_id="agent-test",
                    workdir=str(repo),
                    wait=False,
                )
        finally:
            sys.path.pop(0)


def test_submit_eval_unlimited_when_zero():
    """grader.max_pending_per_agent: 0 disables the cap entirely."""
    import sys

    with tempfile.TemporaryDirectory() as d:
        repo = _setup_repo_with_config(Path(d))
        _set_grader_config(repo, max_pending_per_agent=0)

        sys.path.insert(0, str(repo))
        try:
            for i in range(5):
                (repo / "hello.py").write_text(f"print('v{i}')\n")
                submit_eval(
                    message=f"v{i}",
                    agent_id="agent-test",
                    workdir=str(repo),
                    wait=False,
                )
            # All five sit in the queue as pending; nothing was rejected.
            attempts_dir = repo / ".coral" / "public" / "attempts"
            assert len(list(attempts_dir.glob("*.json"))) == 5
        finally:
            sys.path.pop(0)


def test_submit_eval_per_agent_isolation():
    """A pending submission from agent-A must not block agent-B from submitting."""
    import sys

    with tempfile.TemporaryDirectory() as d:
        repo = _setup_repo_with_config(Path(d))
        sys.path.insert(0, str(repo))
        try:
            (repo / "hello.py").write_text("print('a')\n")
            submit_eval(message="a", agent_id="agent-A", workdir=str(repo), wait=False)

            # agent-A is at its limit, but agent-B has no pending attempts.
            (repo / "hello.py").write_text("print('b')\n")
            second = submit_eval(
                message="b",
                agent_id="agent-B",
                workdir=str(repo),
                wait=False,
            )
            assert second.status == "pending"
            assert second.agent_id == "agent-B"
        finally:
            sys.path.pop(0)


def test_submit_eval_sets_shared_state_hash():
    """submit_eval should checkpoint shared state and store hash in the attempt.

    The checkpoint runs before write_attempt, so the first eval has no prior
    shared state changes (hash is None). The second eval sees the first eval's
    attempt JSON and eval_count, producing a non-None hash.
    """
    import sys

    with tempfile.TemporaryDirectory() as d:
        repo = _setup_repo_with_config(Path(d))

        sys.path.insert(0, str(repo))
        try:
            # First eval — no prior shared state changes, hash should be None
            (repo / "hello.py").write_text("print('v1')\n")
            a1 = _submit_and_grade("first", "agent-test", str(repo))
            assert a1.shared_state_hash is None

            # Second eval — first eval wrote attempt JSON + eval_count, so checkpoint finds changes
            (repo / "hello.py").write_text("print('v2')\n")
            a2 = _submit_and_grade("second", "agent-test", str(repo))
            assert a2.shared_state_hash is not None
            assert len(a2.shared_state_hash) == 40
            # Parent shared state hash comes from the first attempt
            assert a2.parent_shared_state_hash == a1.shared_state_hash

            # Verify hashes were persisted in the attempt JSON
            attempt_file = repo / ".coral" / "public" / "attempts" / f"{a2.commit_hash}.json"
            data = json.loads(attempt_file.read_text())
            assert data["shared_state_hash"] == a2.shared_state_hash
        finally:
            sys.path.pop(0)


# --- setup_claude_settings tests ---


def test_setup_claude_settings_permissions():
    """Settings should grant tool permissions."""
    with tempfile.TemporaryDirectory() as d:
        worktree = Path(d) / "worktree"
        worktree.mkdir()
        coral_dir = Path(d) / ".coral"
        (coral_dir / "private").mkdir(parents=True)
        (coral_dir / "public").mkdir(parents=True)

        setup_claude_settings(worktree, coral_dir)

        settings = json.loads((worktree / ".claude" / "settings.local.json").read_text())
        private_dir = str(coral_dir.resolve() / "private")

        worktree_str = str(worktree.resolve())
        # Read access is scoped to the agent's island root (public/ in
        # single-island mode), which transitively covers the bundled
        # subagent definitions under public/agents/.
        state_root_str = str(coral_dir.resolve() / "public")
        agents_dir = str(coral_dir.resolve() / "public" / "agents")

        # No sandbox
        assert "sandbox" not in settings

        # Permission allow rules grant agent autonomy
        allow = settings["permissions"]["allow"]
        # Bash is unscoped; Read/Edit/Write scoped to own worktree
        assert "Bash" in allow
        assert any("Read" in r and worktree_str in r for r in allow)
        assert any("Read" in r and state_root_str in r for r in allow)
        assert any("Read" in r and agents_dir in r for r in allow)
        assert any("Edit" in r and worktree_str in r for r in allow)
        assert any("Write" in r and worktree_str in r for r in allow)
        assert "WebSearch" in allow  # research=True by default
        assert "WebFetch" in allow

        # Permission deny rules block git and private dir
        deny = settings["permissions"]["deny"]
        assert "Bash(git *)" in deny
        assert any(private_dir in r for r in deny)
        assert not any("WebSearch" in r for r in deny)

        assert "hooks" not in settings

        # No defaultMode written: a project-level "auto" is silently downgraded
        # to "default" in headless -p mode, so the runtime sets the mode via
        # the --permission-mode CLI flag instead (see claude_code.py).
        assert "defaultMode" not in settings["permissions"]


def test_setup_claude_settings_no_research():
    """Settings should deny WebSearch/WebFetch when research=False."""
    with tempfile.TemporaryDirectory() as d:
        worktree = Path(d) / "worktree"
        worktree.mkdir()
        coral_dir = Path(d) / ".coral"
        (coral_dir / "private").mkdir(parents=True)
        (coral_dir / "public").mkdir(parents=True)

        setup_claude_settings(worktree, coral_dir, research=False)

        settings = json.loads((worktree / ".claude" / "settings.local.json").read_text())
        allow = settings["permissions"]["allow"]
        deny = settings["permissions"]["deny"]

        assert "WebSearch" not in allow
        assert "WebFetch" not in allow
        assert "WebSearch" in deny
        assert "WebFetch" in deny


def test_submit_eval_multi_island_writes_to_island_attempts(tmp_path, monkeypatch):
    """When .coral_island is set, submit_eval writes to islands/<id>/attempts/."""
    import subprocess

    from coral.config import CoralConfig
    from coral.hooks.post_commit import submit_eval

    # Build a minimal multi-island layout: coral_dir + a worktree
    coral_dir = tmp_path / ".coral"
    (coral_dir / "islands" / "1" / "attempts").mkdir(parents=True)
    cfg = CoralConfig.from_dict(
        {
            "task": {"name": "t", "description": "d"},
            "islands": {"count": 2},
            "workspace": {
                "results_dir": str(tmp_path / "results"),
                "repo_path": str(tmp_path / "src"),
            },
        }
    )
    cfg.to_yaml(coral_dir / "config.yaml")

    worktree = tmp_path / "worktree"
    worktree.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"], cwd=str(worktree), check=True, capture_output=True
    )
    # Set repo-local identity so the commit performed inside submit_eval also
    # has a user/email (CI runners don't carry a global identity).
    subprocess.run(
        ["git", "config", "user.email", "t@t"],
        cwd=str(worktree),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "t"],
        cwd=str(worktree),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(worktree),
        check=True,
        capture_output=True,
    )
    (worktree / "file.txt").write_text("change")
    (worktree / ".coral_dir").write_text(str(coral_dir.resolve()))
    (worktree / ".coral_agent_id").write_text("1-agent-1")
    (worktree / ".coral_island").write_text("1")

    attempt = submit_eval(
        message="island-1 eval",
        agent_id="1-agent-1",
        workdir=str(worktree),
        wait=False,
    )

    # Attempt JSON landed in islands/1/attempts/
    expected = coral_dir / "islands" / "1" / "attempts" / f"{attempt.commit_hash}.json"
    assert expected.exists(), f"attempt was not written to {expected}"
    # Did NOT land in public/
    assert not (coral_dir / "public" / "attempts" / f"{attempt.commit_hash}.json").exists()
    # metadata.island_id stamped
    assert (attempt.metadata or {}).get("island_id") == "1"


def test_submit_eval_walks_up_for_multi_island_breadcrumbs(tmp_path):
    """submit_eval from a worktree subdir still writes to that worktree's island."""
    import subprocess

    from coral.config import CoralConfig
    from coral.hooks.post_commit import submit_eval

    coral_dir = tmp_path / ".coral"
    (coral_dir / "islands" / "1" / "attempts").mkdir(parents=True)
    cfg = CoralConfig.from_dict(
        {
            "task": {"name": "t", "description": "d"},
            "islands": {"count": 2},
            "workspace": {
                "results_dir": str(tmp_path / "results"),
                "repo_path": str(tmp_path / "src"),
            },
        }
    )
    cfg.to_yaml(coral_dir / "config.yaml")

    worktree = tmp_path / "worktree"
    subdir = worktree / "pkg"
    subdir.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "-b", "main"], cwd=str(worktree), check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "t@t"],
        cwd=str(worktree),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "t"],
        cwd=str(worktree),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(worktree),
        check=True,
        capture_output=True,
    )
    (subdir / "file.txt").write_text("change")
    (worktree / ".coral_dir").write_text(str(coral_dir.resolve()))
    (worktree / ".coral_agent_id").write_text("1-agent-1")
    (worktree / ".coral_island").write_text("1")

    attempt = submit_eval(
        message="island-1 eval from subdir",
        agent_id="1-agent-1",
        workdir=str(subdir),
        wait=False,
    )

    expected = coral_dir / "islands" / "1" / "attempts" / f"{attempt.commit_hash}.json"
    assert expected.exists()
    assert (attempt.metadata or {}).get("island_id") == "1"


def test_submit_eval_single_island_unchanged(tmp_path):
    """No .coral_island -> today's behavior: write to public/attempts/."""
    import subprocess

    from coral.config import CoralConfig
    from coral.hooks.post_commit import submit_eval

    coral_dir = tmp_path / ".coral"
    (coral_dir / "public" / "attempts").mkdir(parents=True)
    cfg = CoralConfig.from_dict(
        {
            "task": {"name": "t", "description": "d"},
            "workspace": {
                "results_dir": str(tmp_path / "results"),
                "repo_path": str(tmp_path / "src"),
            },
        }
    )
    cfg.to_yaml(coral_dir / "config.yaml")

    worktree = tmp_path / "worktree"
    worktree.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"], cwd=str(worktree), check=True, capture_output=True
    )
    # Set repo-local identity so the commit performed inside submit_eval also
    # has a user/email (CI runners don't carry a global identity).
    subprocess.run(
        ["git", "config", "user.email", "t@t"],
        cwd=str(worktree),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "t"],
        cwd=str(worktree),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=str(worktree),
        check=True,
        capture_output=True,
    )
    (worktree / "file.txt").write_text("change")
    (worktree / ".coral_dir").write_text(str(coral_dir.resolve()))
    (worktree / ".coral_agent_id").write_text("agent-1")

    attempt = submit_eval(
        message="single-island eval",
        agent_id="agent-1",
        workdir=str(worktree),
        wait=False,
    )

    expected = coral_dir / "public" / "attempts" / f"{attempt.commit_hash}.json"
    assert expected.exists()
    # No island_id stamped
    assert "island_id" not in (attempt.metadata or {})
