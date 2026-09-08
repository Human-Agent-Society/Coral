"""Generate CORAL.md agent instructions from template."""

from __future__ import annotations

from pathlib import Path

from coral.config import CoralConfig

_TEMPLATE_PATH = Path(__file__).parent / "coral.md.template"
_SINGLE_TEMPLATE_PATH = Path(__file__).parent / "coral_single.md.template"


def generate_coral_md(
    config: CoralConfig,
    agent_id: str,
    single_agent: bool = False,
    shared_dir: str = ".claude",
    island_id: str | int | None = None,
) -> str:
    """Produce the CORAL.md file that agents read at startup.

    Args:
        config: The coral config
        agent_id: This agent's ID
        single_agent: If True, use simplified single-agent template (no sharing references)
        shared_dir: Name of the shared state directory (e.g. ".claude", ".codex", ".opencode")
        island_id: Agent's island in multi-island mode (formats a provenance hint)
    """
    template_path = _SINGLE_TEMPLATE_PATH if single_agent else _TEMPLATE_PATH
    template = template_path.read_text(encoding="utf-8")

    # Build optional sections
    tips_section = ""
    if config.task.tips:
        tips_section = f"\n## Tips\n{config.task.tips}\n"

    # Determine score direction from config or grader type
    score_direction = _get_score_direction(config)
    # Local import avoids coral.agent's eager AgentManager export looping back
    # through this template module during package initialization.
    from coral.agent.meta_evolve import render_bootstrap_guidance

    meta_evolve_section = render_bootstrap_guidance(config.agents.meta_evolve)

    # Research step is conditional
    research_enabled = config.agents.research
    if research_enabled:
        workflow_summary = "research → plan → edit → eval → repeat"
        research_section = (
            "\n## 1. Research\n\n"
            "**On your first iteration and whenever you're changing direction**, "
            "invest time in deep research before planning. "
            f"Read the `deep-research` skill (`{shared_dir}/skills/deep-research/SKILL.md`) "
            "for a structured research workflow.\n\n"
            "**Research steps:**\n"
            "- **Understand the problem deeply** — read the grader source at "
            f"`{shared_dir}/grader/` (the exact code that scores you — read it to "
            "understand the objective, but never modify it), identify constraints "
            "and evaluation criteria.\n"
            "- **Survey the literature** — use web search to find state-of-the-art approaches, "
            "academic papers, benchmark comparisons, and existing implementations. "
            'Search broadly first (`"[problem] state of the art"`), then drill into '
            "specific techniques.\n"
            "- **Review domain knowledge** — if the task involves specialized domains "
            "(biology, chemistry, physics, math), research the underlying science. "
            "Understanding the domain often reveals approaches that pure ML/CS thinking misses.\n"
            "- **Analyze existing solutions** — check shared notes, past attempts, and "
            "what has been tried before. Build on what's known.\n"
            "- **Compare 2-4 candidate approaches** — document trade-offs, evidence, "
            "and implementation complexity for each.\n"
            f"- **Track coverage** — maintain the research coverage ledger at "
            f"`{shared_dir}/notes/research/_coverage.md` (task dimensions × "
            "covered/partial/missing); target the gaps, don't re-cover what's done.\n"
            f"- **Write a research summary** — save findings to `{shared_dir}/notes/research/[topic].md` "
            f"so all agents benefit (see `{shared_dir}/skills/deep-research/references/` for "
            f"templates), then run `{shared_dir}/skills/deep-research/scripts/check_grounding.py "
            f"{shared_dir}/notes` to verify every finding traces to a saved source.\n\n"
            "**When to research:**\n"
            "- First iteration: always. Understand the landscape before writing code.\n"
            "- After getting stuck (3+ evals without improvement): step back and "
            "look for new angles.\n"
            "- When pivoting to a fundamentally different approach.\n"
            "- When the task involves unfamiliar domain knowledge.\n\n"
            "**When to skip:** If you have a clear plan from your last eval's feedback "
            "and just need to iterate on an existing approach, go straight to Step 2.\n"
        )
        step_offset = 2  # Plan starts at step 2
        research_back_reference = " (or **Step 1: Research** if you need a new direction)"
        repeat_research_hint = (
            "go back to **Step 1: Research** to find new techniques via web search, "
        )
    else:
        workflow_summary = "plan → edit → eval → repeat"
        research_section = ""
        step_offset = 1  # Plan starts at step 1
        research_back_reference = ""
        repeat_research_hint = "research new techniques, "

    rendered = template.format(
        task_name=config.task.name,
        task_description=config.task.description,
        tips_section=tips_section,
        score_direction=score_direction,
        agent_id=agent_id,
        shared_dir=shared_dir,
        workflow_summary=workflow_summary,
        research_section=research_section,
        plan_step_num=step_offset,
        edit_step_num=step_offset + 1,
        eval_step_num=step_offset + 2,
        results_step_num=step_offset + 3,
        knowledge_step_num=step_offset + 4,
        research_back_reference=research_back_reference,
        repeat_research_hint=repeat_research_hint,
        meta_evolve_section=meta_evolve_section,
    )

    # Multi-island provenance hint: append once at the end so the existing
    # template stays untouched. Only when there's actually more than one island.
    if island_id is not None and config.islands.count > 1:
        rendered += (
            "\n\n## Multi-island provenance\n\n"
            f"You are working on island `{island_id}` in a multi-island run "
            f"({config.islands.count} islands total). Other islands exist with "
            "their own attempts, notes, and skills, but you cannot see their "
            "state directly. Each island evolves independently.\n"
        )

    return rendered


def _get_score_direction(config: CoralConfig) -> str:
    """Return a human-readable description of what 'better' means for this grader."""
    if config.grader.direction == "minimize":
        return "lower is better"
    return "higher is better"
