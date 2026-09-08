"""Integration tests for the full CORAL flow."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from coral.config import AgentConfig, CoralConfig, GraderConfig, TaskConfig, WorkspaceConfig
from coral.hub.attempts import (
    format_leaderboard,
    get_leaderboard,
    read_attempts,
    search_attempts,
    write_attempt,
)
from coral.template.coral_md import generate_coral_md
from coral.types import Attempt
from coral.workspace import create_project


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv binary required")
def test_full_workspace_creation():
    """Test creating a complete project structure with worktrees."""
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)

        # Create a source repo
        repo = base / "source"
        repo.mkdir()
        subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], capture_output=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "T"], capture_output=True)
        (repo / "main.py").write_text("x = 1\n")
        subprocess.run(["git", "-C", str(repo), "add", "."], capture_output=True)
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-m", "init"], capture_output=True, check=True
        )

        # Create project
        config = CoralConfig(
            task=TaskConfig(name="optimize", description="Make it fast", tips="Profile first"),
            grader=GraderConfig(),
            agents=AgentConfig(count=2),
            workspace=WorkspaceConfig(results_dir=str(base / "results"), repo_path=str(repo)),
        )

        paths = create_project(config)

        # Verify structure
        assert paths.run_dir.exists()
        assert (paths.coral_dir / "config.yaml").exists()
        assert (paths.coral_dir / "public" / "attempts").is_dir()
        assert (paths.coral_dir / "public" / "logs").is_dir()
        assert (paths.coral_dir / "public" / "skills").is_dir()
        assert (paths.coral_dir / "public" / "notes").is_dir()
        assert (paths.coral_dir / "private").is_dir()

        # Reload config from disk
        restored = CoralConfig.from_yaml(paths.coral_dir / "config.yaml")
        assert restored.task.name == "optimize"
        assert restored.agents.count == 2


def test_multi_agent_shared_state():
    """Test that multiple agents can write to and read from shared .coral/."""
    with tempfile.TemporaryDirectory() as d:
        coral_dir = Path(d)
        (coral_dir / "public" / "attempts").mkdir(parents=True)

        # Agent 1 writes an attempt
        a1 = Attempt(
            commit_hash="aaa111",
            agent_id="agent-1",
            title="Try ReLU",
            score=0.7,
            status="improved",
            parent_hash=None,
            timestamp="2026-03-11T10:00:00Z",
            feedback="Better activation",
        )
        write_attempt(str(coral_dir), a1)

        # Agent 2 writes an attempt
        a2 = Attempt(
            commit_hash="bbb222",
            agent_id="agent-2",
            title="Try GELU",
            score=0.65,
            status="improved",
            parent_hash=None,
            timestamp="2026-03-11T10:01:00Z",
            feedback="Slightly worse",
        )
        write_attempt(str(coral_dir), a2)

        # Agent 3 writes an attempt building on agent-1
        a3 = Attempt(
            commit_hash="ccc333",
            agent_id="agent-3",
            title="ReLU squared",
            score=0.85,
            status="improved",
            parent_hash="aaa111",
            timestamp="2026-03-11T10:02:00Z",
            feedback="Big improvement",
        )
        write_attempt(str(coral_dir), a3)

        # All agents can see the full leaderboard
        leaderboard = get_leaderboard(str(coral_dir))
        assert len(leaderboard) == 3
        assert leaderboard[0].score == 0.85
        assert leaderboard[0].agent_id == "agent-3"

        # Search works across agents
        relu_results = search_attempts(str(coral_dir), "ReLU")
        assert len(relu_results) == 2  # "Try ReLU" and "ReLU squared"

        gelu_results = search_attempts(str(coral_dir), "GELU")
        assert len(gelu_results) == 1

        # Format leaderboard as markdown
        md = format_leaderboard(leaderboard)
        assert "agent-3" in md
        assert "0.8500" in md


def test_coral_md_generation():
    """Test that generated CORAL.md contains all necessary info."""
    config = CoralConfig(
        task=TaskConfig(
            name="Kernel Optimization",
            description="Optimize the VLIW kernel for minimum cycle count.",
            tips="- Use SIMD vectorization\n- Minimize memory stalls",
        ),
        grader=GraderConfig(direction="minimize"),
        agents=AgentConfig(count=3),
    )

    md = generate_coral_md(config, "agent-2")

    # Must contain task info
    assert "Kernel Optimization" in md
    assert "VLIW kernel" in md

    # Must include tips
    assert "SIMD vectorization" in md

    # Must reference the agent ID
    assert "agent-2" in md

    # Must include core sections
    assert "Orientation" in md
    assert "## 1. Plan" in md
    assert "## 2. Edit" in md
    assert "## 5. Share Knowledge" in md
    assert "fully autonomous" in md

    # Must reference shared workspace commands
    assert "coral log --search" in md
    assert "coral log" in md
    assert ".claude/notes/" in md
    assert ".claude/skills/" in md


def test_end_to_end_attempt_lifecycle():
    """Test the full lifecycle: write → search → leaderboard → format."""
    with tempfile.TemporaryDirectory() as d:
        coral_dir = Path(d)
        (coral_dir / "public" / "attempts").mkdir(parents=True)

        # Simulate 5 agents making attempts
        attempts_data = [
            ("a1", "agent-1", "Baseline approach", 0.3),
            ("a2", "agent-2", "Learning rate 0.01", 0.45),
            ("a3", "agent-1", "Learning rate 0.001 with warmup", 0.55),
            ("a4", "agent-3", "Attention heads 8 instead of 4", 0.62),
            ("a5", "agent-2", "Learning rate cosine decay", 0.71),
        ]

        for commit, agent, title, score in attempts_data:
            attempt = Attempt(
                commit_hash=commit,
                agent_id=agent,
                title=title,
                score=score,
                status="improved",
                parent_hash=None,
                timestamp=f"2026-03-11T10:{attempts_data.index((commit, agent, title, score)):02d}:00Z",
            )
            write_attempt(str(coral_dir), attempt)

        # Verify all written
        all_attempts = read_attempts(str(coral_dir))
        assert len(all_attempts) == 5

        # Search for learning rate approaches
        lr_results = search_attempts(str(coral_dir), "learning rate")
        assert len(lr_results) == 3

        # Top 3 leaderboard
        top3 = get_leaderboard(str(coral_dir), top_n=3)
        assert len(top3) == 3
        assert top3[0].score == 0.71
        assert top3[1].score == 0.62
        assert top3[2].score == 0.55

        # Format
        md = format_leaderboard(top3)
        assert "0.7100" in md
        assert "agent-2" in md
