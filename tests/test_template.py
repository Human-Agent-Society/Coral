"""Tests for CORAL.md template generation."""

from pathlib import Path

from coral.config import AgentConfig, CoralConfig, GraderConfig, TaskConfig
from coral.template.coral_md import generate_coral_md


def test_generate_coral_md_has_required_sections():
    config = CoralConfig(
        task=TaskConfig(
            name="Kernel Optimization",
            description="Optimize the kernel for speed.",
            tips="Profile first!",
        ),
        grader=GraderConfig(direction="minimize"),
        agents=AgentConfig(count=2),
    )

    md = generate_coral_md(config, "agent-1")

    # Task info
    assert "Kernel Optimization" in md
    assert "Optimize the kernel for speed" in md

    # Tips
    assert "Profile first!" in md

    # Agent ID
    assert "agent-1" in md
    assert "creator: agent-1" in md

    # Score direction comes from grader.direction now (no type-based table)
    assert "lower is better" in md

    # Core structure
    assert "Orientation" in md
    assert "## 1. Plan" in md
    assert "## 2. Edit" in md
    assert "## 3. Evaluate" in md
    assert "## 5. Share Knowledge" in md
    assert "Ground Rules" in md

    # Key behavioral instructions
    assert "fully autonomous" in md
    assert "Do not duplicate effort" in md
    assert "Keep iterating" in md

    # Multi-agent awareness
    assert "several agents" in md
    assert "other agents" in md

    # Shared state
    assert "coral log --search" in md
    assert ".claude/notes" in md
    assert ".claude/skills/" in md
    assert "collectively maintain and co-evolve the notes schema" in md


def test_generate_coral_md_without_optional_sections():
    config = CoralConfig(
        task=TaskConfig(name="Simple Task", description="Do the thing."),
        grader=GraderConfig(),
    )

    md = generate_coral_md(config, "agent-5")

    assert "Simple Task" in md
    assert "Do the thing." in md
    assert "agent-5" in md
    assert "## Key Files" not in md
    assert "## Tips" not in md
    assert "higher is better" in md


def test_generate_coral_md_disabled_meta_evolve_adds_no_guidance():
    config = CoralConfig(
        task=TaskConfig(name="Simple Task", description="Do the thing."),
        grader=GraderConfig(),
    )

    for single_agent in (False, True):
        md = generate_coral_md(config, "agent-1", single_agent=single_agent)
        assert "## Meta-evolve attribution" not in md
        assert "--operator" not in md
        assert "--mutation" not in md


def test_generate_coral_md_enabled_meta_evolve_lists_arms_and_eval_contract():
    config = CoralConfig(
        task=TaskConfig(name="Adaptive Task", description="Improve the score."),
        grader=GraderConfig(),
        agents=AgentConfig(
            meta_evolve={
                "enabled": True,
                "arms": [
                    {"operator": "prompt", "mutation": "rewrite"},
                    {"operator": "implementation", "mutation": "replace"},
                ],
            }
        ),
    )

    for single_agent in (False, True):
        md = generate_coral_md(config, "agent-1", single_agent=single_agent)
        assert "## Meta-evolve attribution" in md
        assert "`prompt` / `rewrite`" in md
        assert "`implementation` / `replace`" in md
        assert "--operator prompt --mutation rewrite" in md
        assert "Every real eval" in md
        assert "first unobserved arm" in md


def test_generate_coral_md_single_agent():
    """Single-agent template omits multi-agent sharing references."""
    config = CoralConfig(
        task=TaskConfig(
            name="Solo Task",
            description="Optimize alone.",
            tips="Be thorough.",
        ),
        grader=GraderConfig(),
        agents=AgentConfig(count=1),
    )

    md = generate_coral_md(config, "agent-1", single_agent=True)

    # Core content present
    assert "Solo Task" in md
    assert "Optimize alone." in md
    assert "Be thorough." in md
    assert "agent-1" in md
    assert "fully autonomous" in md
    assert "Keep iterating" in md

    # Multi-agent references absent
    assert "several agents" not in md
    assert "other agents" not in md
    assert "Share Knowledge" not in md
    assert "Do not duplicate effort" not in md

    # Single-agent still has notes/skills (for self-use)
    assert "notes" in md.lower()
    assert "skills" in md.lower()
    assert "Record Knowledge" in md
    assert "maintain and co-evolve this living notes schema" in md


def test_create_notes_skill_assigns_collective_schema_stewardship():
    skill = Path("coral/template/skills/create-notes/SKILL.md").read_text(encoding="utf-8")

    assert "collectively maintain and co-evolve the notes schema" in skill
    assert "frontmatter-spec.md" in skill
    assert "stamp.py" in skill
    assert "lint.py" in skill


def test_generate_coral_md_tune_guardrails_present():
    """The when-to / when-not-to guardrails are explicit in both templates."""
    config = CoralConfig(
        task=TaskConfig(name="t", description="d"),
        grader=GraderConfig(),
        agents=AgentConfig(count=2),
    )
    md_multi = generate_coral_md(config, "agent-1")
    md_single = generate_coral_md(config, "agent-1", single_agent=True)
    for md in (md_multi, md_single):
        assert "Use `--tune` for" in md or "Use `--tune` for:" in md
        assert "Do NOT use `--tune` for" in md
        assert "final" in md.lower()
        # Plateau-dodge guardrail.
        assert "plateau" in md.lower()
        # Per-grader description now ships in feedback, not in CORAL.md.
        # The template should advertise that contract so the agent knows
        # to look for the [--tune mode] line in their next eval result.
        assert "[--tune mode]" in md


def test_generate_coral_md_does_not_describe_tune_per_grader():
    """Per-grader tune description is now delivered via feedback, not CORAL.md."""
    config = CoralConfig(
        task=TaskConfig(name="t", description="d"),
        grader=GraderConfig(),
    )
    md = generate_coral_md(config, "agent-1")
    # Old placeholder must be gone.
    assert "{tune_description}" not in md
    assert "What this grader does in tune mode" not in md


def test_generate_coral_md_score_direction_from_config():
    """Score direction now comes solely from grader.direction (no type table)."""
    for direction, expected in [
        ("maximize", "higher is better"),
        ("minimize", "lower is better"),
    ]:
        config = CoralConfig(
            task=TaskConfig(name="t", description="d"),
            grader=GraderConfig(direction=direction),
        )
        md = generate_coral_md(config, "agent-1")
        assert expected in md, f"Missing '{expected}' for direction '{direction}'"


def test_generate_coral_md_single_island_has_no_island_mention():
    """Without island_id, no provenance hint about islands is in the output."""
    from coral.config import CoralConfig
    from coral.template.coral_md import generate_coral_md

    cfg = CoralConfig.from_dict({"task": {"name": "t", "description": "d"}})
    md = generate_coral_md(cfg, agent_id="agent-1")
    assert "island" not in md.lower()


def test_generate_coral_md_multi_island_mentions_island(tmp_path):
    """With island_id and islands.count > 1, the prompt mentions the agent's island."""
    from coral.config import CoralConfig
    from coral.template.coral_md import generate_coral_md

    cfg = CoralConfig.from_dict(
        {
            "task": {"name": "t", "description": "d"},
            "islands": {"count": 4},
        }
    )
    md = generate_coral_md(cfg, agent_id="2-agent-1", island_id="2")
    md_lower = md.lower()
    assert "island" in md_lower
    assert "island `2`" in md or "island 2" in md, "expected mention of island 2"
