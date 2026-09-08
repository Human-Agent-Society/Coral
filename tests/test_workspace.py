"""Tests for workspace setup."""

import os
import subprocess
import tempfile
import tomllib
from pathlib import Path

import pytest

from coral.config import AgentConfig, CoralConfig, GraderConfig, TaskConfig, WorkspaceConfig
from coral.workspace import (
    apply_runtime_mounts,
    create_agent_worktree,
    create_project,
    seed_agent_role,
    setup_codex_settings,
    setup_git_exclude,
    setup_shared_state,
    setup_worktree_env,
    write_agent_id,
)


def _make_config(repo_path: str, results_dir: str | None = None) -> CoralConfig:
    return CoralConfig(
        task=TaskConfig(name="Test Task", description="Test task"),
        grader=GraderConfig(),
        agents=AgentConfig(count=2),
        workspace=WorkspaceConfig(
            results_dir=results_dir or os.path.join(repo_path, "results"),
            repo_path=repo_path,
        ),
    )


def _git_init(d: str) -> None:
    """Initialise a git repo with a dummy commit (works without global config)."""
    subprocess.run(["git", "init", d], capture_output=True, check=True)
    subprocess.run(
        [
            "git",
            "-C",
            d,
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@test.com",
            "commit",
            "--allow-empty",
            "-m",
            "init",
        ],
        capture_output=True,
        check=True,
    )


def test_create_project_structure():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        # Init a git repo so workspace can create worktrees
        _git_init(d)

        config = _make_config(d)
        paths = create_project(config)

        assert paths.run_dir.exists()
        assert paths.task_dir.exists()
        assert paths.coral_dir.exists()
        assert (paths.coral_dir / "public").is_dir()
        assert (paths.coral_dir / "public" / "attempts").is_dir()
        assert (paths.coral_dir / "public" / "logs").is_dir()
        assert (paths.coral_dir / "public" / "skills").is_dir()
        assert (paths.coral_dir / "public" / "notes").is_dir()
        assert (paths.coral_dir / "private").is_dir()
        assert (paths.coral_dir / "config.yaml").is_file()
        assert paths.agents_dir.exists()
        # Structure: results/<task-slug>/<timestamp>/
        assert "test-task" in str(paths.task_dir)
        # latest symlink
        latest = paths.task_dir / "latest"
        assert latest.is_symlink()


def test_create_project_unique_runs():
    """Each create_project call gets a unique run directory."""
    # ignore_cleanup_errors: git's .git/objects can have background writes
    # racing with rmtree at exit. The test logic is complete by then.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        _git_init(d)

        config = _make_config(d)
        paths1 = create_project(config)

        import time

        time.sleep(1.1)  # ensure different timestamp

        paths2 = create_project(config)

        assert paths1.run_dir != paths2.run_dir
        assert paths1.coral_dir != paths2.coral_dir
        # latest should point to the second run directory
        latest = paths1.task_dir / "latest"
        assert latest.resolve() == paths2.run_dir.resolve()


def test_write_agent_id():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        worktree = Path(d)
        write_agent_id(worktree, "agent-42")
        content = (worktree / ".coral_agent_id").read_text()
        assert content == "agent-42"


def test_setup_git_exclude():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        _git_init(d)
        worktree = Path(d)
        setup_git_exclude(worktree)

        exclude = worktree / ".git" / "info" / "exclude"
        assert exclude.exists()
        content = exclude.read_text()
        assert ".coral_agent_id" in content
        assert "CLAUDE.md" in content
        assert ".claude/" in content
        assert ".coral_island" in content
        # The tracked .gitignore is never touched
        assert not (worktree / ".gitignore").exists()


def test_setup_git_exclude_preserves_existing():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        _git_init(d)
        worktree = Path(d)
        exclude = worktree / ".git" / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        exclude.write_text("*.pyc\n__pycache__/\n")

        setup_git_exclude(worktree)

        content = exclude.read_text()
        assert "*.pyc" in content
        assert ".coral_agent_id" in content
        assert ".claude/" in content


def test_setup_git_exclude_idempotent():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        _git_init(d)
        worktree = Path(d)
        setup_git_exclude(worktree)
        setup_git_exclude(worktree)

        content = (worktree / ".git" / "info" / "exclude").read_text()
        assert content.count(".claude/") == 1


def test_setup_git_exclude_requires_git_repo():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        with pytest.raises(RuntimeError, match="git-common-dir"):
            setup_git_exclude(Path(d))


def test_setup_git_exclude_survives_reset_hard():
    """Regression test for #171: breadcrumbs stay ignored after any hard reset.

    The old implementation appended entries to the working-tree .gitignore;
    `coral revert` / `coral checkout` (or an agent running `git reset --hard`
    directly) restored .gitignore to its committed state, and the next
    `git add -A` staged the breadcrumb files into the attempt's commit.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        repo = Path(d)
        _git_init(d)
        setup_git_exclude(repo)

        # CORAL breadcrumbs (untracked, must stay that way)
        (repo / ".coral_agent_id").write_text("agent-1")
        (repo / ".coral_dir").write_text("/tmp/coral")

        # Agent's first eval: change + `git add -A` + commit
        (repo / "main.py").write_text("print('attempt 1')\n")

        def _run(*args: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                ["git", "-C", d, "-c", "user.name=test", "-c", "user.email=test@test.com", *args],
                capture_output=True,
                text=True,
                check=True,
            )

        _run("add", "-A")
        _run("commit", "-m", "attempt 1")

        # Revert past the first attempt (what `coral revert` does)
        _run("reset", "--hard", "HEAD~1")

        # The next eval's `git add -A` must not stage breadcrumbs
        _run("add", "-A")
        status = _run("status", "--porcelain").stdout
        assert ".coral_agent_id" not in status
        assert ".coral_dir" not in status


def test_setup_git_exclude_shared_across_worktrees():
    """Entries written from one worktree apply to all worktrees of the repo."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        repo = Path(d) / "repo"
        repo.mkdir()
        _git_init(str(repo))

        agents_dir = Path(d) / "agents"
        agents_dir.mkdir()
        worktree = create_agent_worktree(repo, "agent-1", agents_dir)

        # Configure from the linked worktree; must land in the shared exclude
        setup_git_exclude(worktree)
        assert ".coral_agent_id" in (repo / ".git" / "info" / "exclude").read_text()

        (worktree / ".coral_agent_id").write_text("agent-1")
        result = subprocess.run(
            ["git", "-C", str(worktree), "check-ignore", ".coral_agent_id"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # ...and in the main checkout too
        (repo / ".coral_agent_id").write_text("main")
        result = subprocess.run(
            ["git", "-C", str(repo), "check-ignore", ".coral_agent_id"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


@pytest.mark.parametrize(
    ("research", "expected"),
    [
        (True, "live"),
        (False, "disabled"),
    ],
)
def test_setup_codex_settings_writes_top_level_web_search(
    research: bool,
    expected: str,
):
    """Codex expects web_search as a top-level mode, not under [tools]."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        worktree = Path(d) / "worktree"
        coral_dir = Path(d) / ".coral"
        worktree.mkdir()
        coral_dir.mkdir()

        setup_codex_settings(worktree, coral_dir, research=research)

        config_toml = (worktree / ".codex" / "config.toml").read_text()
        config = tomllib.loads(config_toml)
        assert config["web_search"] == expected
        assert "tools" not in config


def test_create_project_runs_setup_commands():
    """Setup commands execute in the worktree directory."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        worktree = Path(d) / "worktree"
        worktree.mkdir()

        setup_worktree_env(worktree, ["echo hello > setup_marker.txt"])

        marker = worktree / "setup_marker.txt"
        assert marker.exists()
        assert marker.read_text().strip() == "hello"


def test_create_project_setup_command_failure():
    """A failing setup command raises RuntimeError."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        worktree = Path(d) / "worktree"
        worktree.mkdir()

        with pytest.raises(RuntimeError, match="Setup command failed"):
            setup_worktree_env(worktree, ["exit 1"])


def test_create_project_setup_runs_sequentially():
    """Setup commands run in order so later commands can depend on earlier ones."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        worktree = Path(d) / "worktree"
        worktree.mkdir()

        setup_worktree_env(
            worktree,
            [
                "mkdir -p mydir",
                "echo done > mydir/result.txt",
            ],
        )

        result_file = worktree / "mydir" / "result.txt"
        assert result_file.exists()
        assert result_file.read_text().strip() == "done"


def test_setup_worktree_env_runs_even_when_venv_exists(monkeypatch):
    """Verify setup commands run even if .venv/bin/python already exists."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        worktree = Path(d) / "worktree"
        worktree.mkdir()
        # Pre-create a fake populated venv
        venv_bin = worktree / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").write_text("#!/bin/sh\nexit 0\n")
        (venv_bin / "python").chmod(0o755)

        monkeypatch.setattr("shutil.which", lambda cmd: None)

        marker = worktree / "setup_ran.marker"
        setup_worktree_env(worktree, [f"touch {marker}"])

        assert marker.exists(), "Setup commands should run even when .venv exists"


def test_setup_worktree_env_runs_when_venv_missing():
    """When .venv doesn't exist yet, setup runs as normal (first launch path)."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        worktree = Path(d) / "worktree"
        worktree.mkdir()

        marker = worktree / "setup_ran.marker"
        setup_worktree_env(worktree, [f"touch {marker}"])

        assert marker.exists(), "Setup should have run on first launch"


# --- apply_runtime_mounts tests ---


def _mount_workspace(d: Path) -> tuple[Path, Path]:
    """Create a worktree dir and a base_dir under d; return both."""
    worktree = d / "worktree"
    worktree.mkdir()
    base = d / "base"
    base.mkdir()
    return worktree, base


def test_apply_runtime_mounts_no_mounts_is_noop():
    """Empty/missing mounts must not error or touch the worktree."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        worktree, base = _mount_workspace(Path(d))
        before = sorted(worktree.iterdir())
        apply_runtime_mounts(worktree, {}, base)
        apply_runtime_mounts(worktree, None, base)  # type: ignore[arg-type]
        assert sorted(worktree.iterdir()) == before


def test_apply_runtime_mounts_copies_file_with_relative_source():
    """Relative source resolves against base_dir; dest is worktree-relative."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        worktree, base = _mount_workspace(Path(d))
        (base / "settings.json").write_text('{"foo": 1}')

        apply_runtime_mounts(
            worktree,
            {"settings.json": ".claude/settings.json"},
            base,
        )

        dest = worktree / ".claude" / "settings.json"
        assert dest.exists()
        assert dest.read_text() == '{"foo": 1}'


def test_apply_runtime_mounts_absolute_source():
    """Absolute source is used as-is (base_dir ignored)."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        worktree, base = _mount_workspace(Path(d))
        elsewhere = Path(d) / "elsewhere"
        elsewhere.mkdir()
        src = elsewhere / "src.json"
        src.write_text("absolute")

        apply_runtime_mounts(worktree, {str(src): ".claude/settings.json"}, base)

        assert (worktree / ".claude" / "settings.json").read_text() == "absolute"


def test_apply_runtime_mounts_expands_tilde(monkeypatch):
    """``~`` in source expands to $HOME."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        worktree, base = _mount_workspace(Path(d))
        fake_home = Path(d) / "fake_home"
        fake_home.mkdir()
        (fake_home / "settings.json").write_text("from-home")
        monkeypatch.setenv("HOME", str(fake_home))

        apply_runtime_mounts(
            worktree,
            {"~/settings.json": ".claude/settings.json"},
            base,
        )

        assert (worktree / ".claude" / "settings.json").read_text() == "from-home"


def test_apply_runtime_mounts_copies_directory():
    """Directory sources copy recursively."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        worktree, base = _mount_workspace(Path(d))
        srcdir = base / "mcp"
        srcdir.mkdir()
        (srcdir / "db.json").write_text("db config")
        (srcdir / "fs.json").write_text("fs config")

        apply_runtime_mounts(worktree, {"mcp": ".claude/mcp"}, base)

        dest = worktree / ".claude" / "mcp"
        assert (dest / "db.json").read_text() == "db config"
        assert (dest / "fs.json").read_text() == "fs config"


def test_apply_runtime_mounts_overwrites_existing_file():
    """Existing dest is overwritten — second invocation refreshes."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        worktree, base = _mount_workspace(Path(d))
        src = base / "settings.json"
        src.write_text("v1")

        apply_runtime_mounts(worktree, {"settings.json": ".claude/settings.json"}, base)
        src.write_text("v2")
        apply_runtime_mounts(worktree, {"settings.json": ".claude/settings.json"}, base)

        assert (worktree / ".claude" / "settings.json").read_text() == "v2"


def test_apply_runtime_mounts_overwrites_corals_settings_local_json():
    """User can replace CORAL's settings.local.json (mounts run last, user wins)."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        worktree, base = _mount_workspace(Path(d))
        # Simulate CORAL having already written settings.local.json
        (worktree / ".claude").mkdir()
        (worktree / ".claude" / "settings.local.json").write_text('{"coral": true}')

        (base / "user-settings.json").write_text('{"user": true}')

        apply_runtime_mounts(
            worktree,
            {"user-settings.json": ".claude/settings.local.json"},
            base,
        )

        assert (worktree / ".claude" / "settings.local.json").read_text() == '{"user": true}'


def test_apply_runtime_mounts_missing_source_raises():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        worktree, base = _mount_workspace(Path(d))
        with pytest.raises(FileNotFoundError, match="mount source"):
            apply_runtime_mounts(worktree, {"nope.json": ".claude/x.json"}, base)


def test_apply_runtime_mounts_absolute_dest_rejected():
    """Dest must be worktree-relative — absolute paths are rejected."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        worktree, base = _mount_workspace(Path(d))
        (base / "src").write_text("x")
        with pytest.raises(ValueError, match="must be worktree-relative"):
            apply_runtime_mounts(worktree, {"src": "/etc/passwd"}, base)


def test_apply_runtime_mounts_dest_escape_rejected():
    """Dest cannot escape the worktree via ``..``."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        worktree, base = _mount_workspace(Path(d))
        (base / "src").write_text("x")
        with pytest.raises(ValueError, match="escapes worktree"):
            apply_runtime_mounts(worktree, {"src": "../escape.txt"}, base)


def test_apply_runtime_mounts_creates_parent_dirs():
    """Nested dest paths get their parent directories created."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        worktree, base = _mount_workspace(Path(d))
        (base / "src.json").write_text("nested")

        apply_runtime_mounts(
            worktree,
            {"src.json": "deeply/nested/dir/file.json"},
            base,
        )

        assert (worktree / "deeply" / "nested" / "dir" / "file.json").read_text() == "nested"


def test_apply_runtime_mounts_multiple_files():
    """All entries in the mounts dict get copied."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        worktree, base = _mount_workspace(Path(d))
        (base / "a.json").write_text("A")
        (base / "b.json").write_text("B")

        apply_runtime_mounts(
            worktree,
            {
                "a.json": ".claude/a.json",
                "b.json": ".claude/b.json",
            },
            base,
        )

        assert (worktree / ".claude" / "a.json").read_text() == "A"
        assert (worktree / ".claude" / "b.json").read_text() == "B"


# --- seed_agent_role tests ---


def _role_workspace(d: Path) -> tuple[Path, Path]:
    """Create a coral_dir + base_dir under d; return both."""
    coral_dir = d / ".coral"
    coral_dir.mkdir()
    base = d / "base"
    base.mkdir()
    return coral_dir, base


def test_seed_agent_role_copies_file():
    """A user-provided .md is copied to public/roles/<agent_id>.md."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        coral_dir, base = _role_workspace(Path(d))
        (base / "integrator.md").write_text("# Integrator role\n")

        dst = seed_agent_role(coral_dir, "agent-1", "integrator.md", base)

        assert dst == coral_dir / "public" / "roles" / "agent-1.md"
        assert dst.read_text() == "# Integrator role\n"


def test_seed_agent_role_idempotent_preserves_existing():
    """Existing role file is never overwritten — agent evolution is preserved."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        coral_dir, base = _role_workspace(Path(d))
        (base / "seed.md").write_text("# seed")

        # First seed
        seed_agent_role(coral_dir, "agent-1", "seed.md", base)
        # Simulate the agent evolving its own role
        evolved = coral_dir / "public" / "roles" / "agent-1.md"
        evolved.write_text("# evolved gen 3")

        # Re-seed (e.g. on resume) must not clobber
        seed_agent_role(coral_dir, "agent-1", "seed.md", base)

        assert evolved.read_text() == "# evolved gen 3"


def test_seed_agent_role_absolute_source():
    """Absolute source paths are used as-is."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        coral_dir, base = _role_workspace(Path(d))
        elsewhere = Path(d) / "elsewhere"
        elsewhere.mkdir()
        src = elsewhere / "skeptic.md"
        src.write_text("# skeptic")

        seed_agent_role(coral_dir, "agent-2", str(src), base)

        assert (coral_dir / "public" / "roles" / "agent-2.md").read_text() == "# skeptic"


def test_seed_agent_role_expands_tilde(monkeypatch):
    """``~`` in source expands to $HOME, matching apply_runtime_mounts."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        coral_dir, base = _role_workspace(Path(d))
        fake_home = Path(d) / "fake_home"
        fake_home.mkdir()
        (fake_home / "id.md").write_text("from-home")
        monkeypatch.setenv("HOME", str(fake_home))

        seed_agent_role(coral_dir, "agent-1", "~/id.md", base)

        assert (coral_dir / "public" / "roles" / "agent-1.md").read_text() == "from-home"


def test_seed_agent_role_missing_source_raises():
    """A missing source surfaces as FileNotFoundError so misconfig fails loudly."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        coral_dir, base = _role_workspace(Path(d))
        with pytest.raises(FileNotFoundError, match="role_file"):
            seed_agent_role(coral_dir, "agent-1", "nope.md", base)


def test_seed_agent_role_per_agent_distinct_content():
    """Multiple agents can each be seeded from a different file."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        coral_dir, base = _role_workspace(Path(d))
        (base / "a.md").write_text("A's role")
        (base / "b.md").write_text("B's role")

        seed_agent_role(coral_dir, "agent-1", "a.md", base)
        seed_agent_role(coral_dir, "agent-2", "b.md", base)

        ids = coral_dir / "public" / "roles"
        assert (ids / "agent-1.md").read_text() == "A's role"
        assert (ids / "agent-2.md").read_text() == "B's role"


def test_seed_agent_role_default_template_when_no_source():
    """source=None falls back to the bundled gen-0 role template."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        coral_dir, _ = _role_workspace(Path(d))

        dst = seed_agent_role(coral_dir, "agent-1")

        assert dst.exists()
        body = dst.read_text()
        # Bundled template renders the agent_id and frontmatter
        assert "agent_id: agent-1" in body
        assert "generation: 0" in body


def test_seed_agent_role_relative_source_without_base_dir_raises():
    """A relative source with no base_dir surfaces as ValueError, not silent cwd resolution."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        coral_dir, _ = _role_workspace(Path(d))
        with pytest.raises(ValueError, match="base_dir is required"):
            seed_agent_role(coral_dir, "agent-1", "relative.md")


# --- setup_shared_state role-symlink tests ---


def test_setup_shared_state_symlinks_roles():
    """roles/ is symlinked into the worktree's shared dir."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        coral_dir = Path(d) / ".coral"
        (coral_dir / "public" / "roles").mkdir(parents=True)
        worktree = Path(d) / "worktree"
        worktree.mkdir()

        setup_shared_state(worktree, coral_dir, ".claude")

        link = worktree / ".claude" / "roles"
        assert link.is_symlink()
        assert link.resolve() == (coral_dir / "public" / "roles").resolve()


def test_setup_shared_state_migrates_real_roles_dir():
    """A previous run's real roles/ dir is migrated into shared, then symlinked."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        coral_dir = Path(d) / ".coral"
        (coral_dir / "public" / "roles").mkdir(parents=True)
        worktree = Path(d) / "worktree"
        (worktree / ".claude" / "roles").mkdir(parents=True)
        # An agent wrote its role into a real local dir before the symlink
        # behavior shipped — make sure we don't lose that file.
        (worktree / ".claude" / "roles" / "agent-1.md").write_text("local content")

        setup_shared_state(worktree, coral_dir, ".claude")

        link = worktree / ".claude" / "roles"
        assert link.is_symlink()
        assert (coral_dir / "public" / "roles" / "agent-1.md").read_text() == "local content"


# --- grader-source surfacing tests ---


def _setup_coral_with_grader(d: str) -> tuple[Path, Path]:
    """Build a minimal .coral with a config_dir breadcrumb + real grader dir.

    Returns (coral_dir, grader_source).
    """
    coral_dir = Path(d) / ".coral"
    (coral_dir / "public").mkdir(parents=True)
    task_dir = Path(d) / "task"
    grader_source = task_dir / "grader"
    grader_source.mkdir(parents=True)
    (grader_source / "grade.py").write_text("# grading logic")
    (coral_dir / "config_dir").write_text(str(task_dir))
    return coral_dir, grader_source


def test_setup_shared_state_symlinks_grader_source():
    """The real grader package is surfaced in the worktree as a symlink (not a
    copy) so the agent can read the exact code that scores it."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        coral_dir, grader_source = _setup_coral_with_grader(d)
        worktree = Path(d) / "worktree"
        worktree.mkdir()

        setup_shared_state(worktree, coral_dir, ".claude")

        link = worktree / ".claude" / "grader"
        assert link.is_symlink()
        assert link.resolve() == grader_source.resolve()
        # Reads resolve through the symlink to the real source.
        assert (link / "grade.py").read_text() == "# grading logic"


def test_setup_shared_state_no_grader_symlink_when_source_missing():
    """A task with no config_dir breadcrumb or no grader/ dir gets no dangling
    grader symlink."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        coral_dir = Path(d) / ".coral"
        (coral_dir / "public").mkdir(parents=True)
        worktree = Path(d) / "worktree"
        worktree.mkdir()

        # No breadcrumb at all.
        setup_shared_state(worktree, coral_dir, ".claude")
        assert not (worktree / ".claude" / "grader").is_symlink()

        # Breadcrumb points at a task dir with no grader/ subdir.
        (coral_dir / "config_dir").write_text(str(Path(d) / "task"))
        (Path(d) / "task").mkdir()
        setup_shared_state(worktree, coral_dir, ".claude")
        assert not (worktree / ".claude" / "grader").exists()
        assert not (worktree / ".claude" / "grader").is_symlink()


def test_claude_settings_grant_read_on_grader_source():
    """The grader symlink target is outside the worktree/state root, so the
    Claude allow-list must explicitly grant Read on it (else the agent sees the
    link but can't read through it). It must NOT be granted Edit/Write."""
    import json

    from coral.workspace.worktree import setup_claude_settings

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        coral_dir, grader_source = _setup_coral_with_grader(d)
        worktree = Path(d) / "worktree"
        worktree.mkdir()

        setup_claude_settings(worktree, coral_dir=coral_dir)

        settings = json.loads((worktree / ".claude" / "settings.local.json").read_text())
        allow = settings["permissions"]["allow"]
        grader = str(grader_source.resolve())
        assert f"Read(/{grader}/**)" in allow
        # Read-only: no Edit/Write grant on the grader source.
        assert not any(r.startswith("Edit(") and grader in r for r in allow)
        assert not any(r.startswith("Write(") and grader in r for r in allow)


def test_opencode_settings_grant_external_dir_on_grader_source():
    """OpenCode gates out-of-project reads via external_directory; the grader
    source must be listed there so the <shared_dir>/grader symlink resolves."""
    import json

    from coral.workspace.worktree import setup_opencode_settings

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        coral_dir, grader_source = _setup_coral_with_grader(d)
        worktree = Path(d) / "worktree"
        worktree.mkdir()

        setup_opencode_settings(worktree, coral_dir=coral_dir)

        settings = json.loads((worktree / ".opencode" / "opencode.json").read_text())
        ext = settings["permission"]["external_directory"]
        assert ext.get(str(grader_source.resolve()) + "/**") == "allow"


def test_repoint_shared_state_swaps_island_targets():
    """repoint_shared_state moves an agent's symlinks from src island to dst."""
    from coral.workspace.worktree import repoint_shared_state

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        coral_dir = Path(d) / ".coral"
        # Multi-island layout
        (coral_dir / "islands" / "0" / "notes").mkdir(parents=True)
        (coral_dir / "islands" / "1" / "notes").mkdir(parents=True)
        worktree = Path(d) / "worktree"
        worktree.mkdir()

        # First wire to island 0, then repoint to island 1.
        setup_shared_state(worktree, coral_dir, ".claude", island_id="0")
        notes_link = worktree / ".claude" / "notes"
        assert notes_link.resolve() == (coral_dir / "islands" / "0" / "notes").resolve()

        repoint_shared_state(worktree, coral_dir, ".claude", new_island_id="1")
        # Symlink now points at island 1
        assert notes_link.is_symlink()
        assert notes_link.resolve() == (coral_dir / "islands" / "1" / "notes").resolve()
        # Breadcrumb updated
        assert (worktree / ".coral_island").read_text() == "1"


def test_repoint_shared_state_creates_missing_dirs_on_destination():
    """An empty destination island still gets a working set of symlinks."""
    from coral.workspace.worktree import repoint_shared_state

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        coral_dir = Path(d) / ".coral"
        (coral_dir / "islands" / "0" / "notes").mkdir(parents=True)
        # Island 1 exists but is empty (no notes/, attempts/, etc. yet).
        (coral_dir / "islands" / "1").mkdir()
        worktree = Path(d) / "worktree"
        worktree.mkdir()

        setup_shared_state(worktree, coral_dir, ".claude", island_id="0")
        repoint_shared_state(worktree, coral_dir, ".claude", new_island_id="1")

        # The helper backfilled the missing subdirs on island 1.
        assert (coral_dir / "islands" / "1" / "notes").is_dir()
        assert (coral_dir / "islands" / "1" / "attempts").is_dir()
        # And the symlinks resolve cleanly (do not dangle).
        for item in ("notes", "attempts", "skills", "heartbeat"):
            link = worktree / ".claude" / item
            assert link.is_symlink()
            assert link.resolve().is_dir()


def test_create_project_seeds_user_skills():
    """agents.skills directories are copied into .coral/public/skills/."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        _git_init(d)
        root = Path(d)

        skill_name = "test-skill"
        skill_dir = root / "skills" / skill_name
        skill_dir.mkdir(parents=True)
        (skill_dir / "run.sh").write_text("#!/bin/bash\necho hello")

        config = CoralConfig(
            task=TaskConfig(name="Test Task", description="Test task"),
            grader=GraderConfig(),
            agents=AgentConfig(count=1, skills=[f"./skills/{skill_name}"]),
            workspace=WorkspaceConfig(results_dir=str(root / "results"), repo_path=d),
        )
        paths = create_project(config, config_dir=root)

        seeded = paths.coral_dir / "public" / "skills" / skill_name / "run.sh"
        assert seeded.is_file()
        assert "echo hello" in seeded.read_text()


def test_create_project_user_skills_override_builtin():
    """User skills with the same name as a built-in skill take precedence."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        _git_init(d)
        root = Path(d)

        skill_name = "test-skill"
        skill_dir = root / "skills" / skill_name
        skill_dir.mkdir(parents=True)
        (skill_dir / "run.sh").write_text("#!/bin/bash\necho custom")

        config = CoralConfig(
            task=TaskConfig(name="Test Task", description="Test task"),
            grader=GraderConfig(),
            agents=AgentConfig(count=1, skills=[f"./skills/{skill_name}"]),
            workspace=WorkspaceConfig(results_dir=str(root / "results"), repo_path=d),
        )
        paths = create_project(config, config_dir=root)

        seeded = paths.coral_dir / "public" / "skills" / skill_name / "run.sh"
        assert seeded.is_file()
        assert "echo custom" in seeded.read_text()


def test_setup_worktree_env_uv_pip_install_failure(monkeypatch):
    """Verify that a non-zero exit code from uv pip install raises a RuntimeError."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        worktree = Path(d) / "worktree"
        worktree.mkdir()

        # Mock resolve_venv_python and shutil.which to pass presence checks
        fake_python = worktree / ".venv" / "bin" / "python"
        fake_python.parent.mkdir(parents=True)
        fake_python.touch()

        monkeypatch.setattr(
            "coral.workspace.worktree.resolve_venv_python", lambda venv: fake_python
        )
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/uv" if cmd == "uv" else None)

        # Mock subprocess.run to simulate a failed 'uv pip install' call
        real_run = subprocess.run

        def mock_subprocess_run(cmd, *args, **kwargs):
            if isinstance(cmd, list) and len(cmd) >= 3 and cmd[:3] == ["uv", "pip", "install"]:
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=1,
                    stdout="",
                    stderr="forced install failure",
                )
            return real_run(cmd, *args, **kwargs)

        monkeypatch.setattr("subprocess.run", mock_subprocess_run)

        with pytest.raises(RuntimeError, match="Failed to install coral into worktree venv"):
            setup_worktree_env(worktree, ["echo setup"])


def test_setup_grader_env_missing_uv_raises_runtime_error(monkeypatch):
    """A missing `uv` binary must surface as the documented RuntimeError.

    Regression test: setup_grader_env promised RuntimeError on any failure
    but let subprocess raise a raw FileNotFoundError when `uv` was not on
    PATH at venv-creation time.
    """
    from coral.workspace.grader_env import setup_grader_env

    monkeypatch.setattr("coral.workspace.grader_env.shutil.which", lambda cmd: None)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        coral_dir = Path(d) / ".coral"

        with pytest.raises(RuntimeError, match="`uv` was not found on PATH"):
            setup_grader_env(coral_dir, GraderConfig(), Path(d))
