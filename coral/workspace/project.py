"""Project-level directory structure and orchestration."""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from coral.config import CoralConfig
from coral.harbor_task import stage_local_harbor_task
from coral.hub._island import island_root
from coral.hub.checkpoint import init_checkpoint_repo
from coral.workspace.repo import (
    clone_or_init_repo,
    copy_private_data,
    copy_seed_directory,
)

logger = logging.getLogger(__name__)


@dataclass
class ProjectPaths:
    """Paths created by create_project."""

    results_dir: Path  # e.g. results/
    task_dir: Path  # e.g. results/erdos-minimum-overlap-problem/
    run_dir: Path  # e.g. results/erdos-minimum-overlap-problem/2026-03-11_163000/
    coral_dir: Path  # run_dir/.coral/
    agents_dir: Path  # run_dir/agents/
    repo_dir: Path  # run_dir/repo/ (cloned per-run)


def slugify(name: str) -> str:
    """Convert a task name to a filesystem-safe slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug or "task"


_SEED_SKILLS_DIR = Path(__file__).parent.parent / "template" / "skills"
_SEED_AGENTS_DIR = Path(__file__).parent.parent / "template" / "agents"
_ROLE_TEMPLATE_PATH = Path(__file__).parent.parent / "template" / "role_template.md"


_PER_ISLAND_SUBDIRS = (
    "attempts",
    "logs",
    "skills",
    "agents",
    "notes",
    "heartbeat",
    "eval_logs",
    "roles",
)


def _island_id_from_root(coral_dir: Path, island_root: Path) -> str | None:
    """Return the island_id implied by island_root, or None for single-island."""
    try:
        rel = island_root.resolve().relative_to((coral_dir / "islands").resolve())
        # rel is like Path("0"); take first segment
        rel_str = str(rel)
        return rel_str.split("/", 1)[0] if rel_str and rel_str != "." else None
    except ValueError:
        return None


def _build_island_subtree(
    coral_dir: Path,
    island_root: Path,
    effective_config_dir: Path,
    user_skill_paths: list[str],
) -> None:
    """Create the per-island state directory tree and seed bundled assets.

    Used once for ``public/`` in single-island mode, and once per
    ``islands/<id>/`` in multi-island mode. Seeds bundled skills + bundled
    subagent templates + initializes the checkpoint git repo for this island.
    """
    for sub in _PER_ISLAND_SUBDIRS:
        (island_root / sub).mkdir(parents=True, exist_ok=True)

    # Seed bundled skills from coral/template/skills/
    if _SEED_SKILLS_DIR.is_dir():
        for skill_dir in _SEED_SKILLS_DIR.iterdir():
            if skill_dir.is_dir():
                dst = island_root / "skills" / skill_dir.name
                if not dst.exists():
                    shutil.copytree(skill_dir, dst)
                    logger.info(f"Seeded skill: {skill_dir.name}")

    # Seed user-provided skills from agents.skills config
    for skill_path in user_skill_paths:
        src = Path(skill_path)
        if not src.is_absolute():
            src = (effective_config_dir / src).resolve()
        if src.is_dir():
            dst = island_root / "skills" / src.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            logger.info(f"Seeded user skill: {src.name}")
        else:
            logger.warning(f"Skill directory not found: {src}")

    # Seed bundled subagent templates from coral/template/agents/
    if _SEED_AGENTS_DIR.is_dir():
        for agent_file in _SEED_AGENTS_DIR.iterdir():
            if agent_file.is_file():
                dst = island_root / "agents" / agent_file.name
                if not dst.exists():
                    shutil.copy2(agent_file, dst)
                    logger.info(f"Seeded agent template: {agent_file.name}")

    # Per-island checkpoint git repo (one .git per island, scoped locks)
    init_checkpoint_repo(
        str(coral_dir),
        island_id=_island_id_from_root(coral_dir, island_root),
    )


def seed_agent_role(
    coral_dir: Path,
    agent_id: str,
    source: str | None = None,
    base_dir: Path | None = None,
    *,
    island_id: str | int | None = None,
) -> Path:
    """Write the per-agent role description at the per-island roles dir.

    The role describes *what the agent does* on the team — its posture, lane,
    objectives, and accumulated self-knowledge. It is mutable and evolves over
    the run.

    Resolves to ``coral_dir/public/roles/<agent_id>.md`` in single-island
    runs, or ``coral_dir/islands/<island_id>/roles/<agent_id>.md`` in
    multi-island runs. The latter matches the symlink installed by
    ``worktree.setup_shared_state``, so the agent's worktree
    ``.claude/roles/<agent_id>.md`` resolves to the file we write here.

    Idempotent: does nothing if the file already exists, so an agent's evolved
    role description is never clobbered by a re-setup or resume.

    When ``source`` is None (the default), renders the bundled gen-0 role
    template — every agent starts with a blank role they earn into.

    When ``source`` is set, copies that user-provided .md file as-is, giving
    each agent a custom starting posture. ``source`` is a host path with ``~``
    expansion; resolved against ``base_dir`` (typically the task directory)
    when not absolute. Matches the path convention of ``apply_runtime_mounts``.

    Raises:
        FileNotFoundError: if ``source`` is given but does not resolve to a file.
        ValueError: if ``source`` is given but ``base_dir`` is None and ``source``
            is a relative path.
    """
    roles_dir = island_root(coral_dir, island_id) / "roles"
    roles_dir.mkdir(parents=True, exist_ok=True)
    dst = roles_dir / f"{agent_id}.md"
    if dst.exists():
        return dst

    if source is None:
        template = _ROLE_TEMPLATE_PATH.read_text(encoding="utf-8")
        rendered = template.format(
            agent_id=agent_id,
            created_at=datetime.now().isoformat(),
        )
        dst.write_text(rendered, encoding="utf-8")
        return dst

    src = Path(source).expanduser()
    if not src.is_absolute():
        if base_dir is None:
            raise ValueError(
                f"role_file {source!r} is relative; base_dir is required to resolve it"
            )
        src = (base_dir / src).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"role_file {source!r} (resolved to {src}) does not exist")
    shutil.copy2(src, dst)
    return dst


def create_project(config: CoralConfig, config_dir: Path | None = None) -> ProjectPaths:
    """Create the full project directory structure.

    Each run gets its own clone of the source repo so runs are fully independent.

    Layout:
        results/
        └── <task-slug>/
            ├── latest -> 2026-03-11_163000   (symlink)
            └── <timestamp>/
                ├── .coral/
                │   ├── public/          # contents symlinked into .claude/ in worktrees
                │   │   ├── CLAUDE.md
                │   │   ├── notes/
                │   │   ├── change_summary.md
                │   │   ├── skills/
                │   │   ├── agents/
                │   │   ├── attempts/
                │   │   ├── logs/
                │   │   └── settings.local.json
                │   ├── private/
                │   └── config.yaml
                ├── repo/                # cloned from source
                └── agents/              # worktrees off repo/
    """
    results_dir = Path(config.workspace.results_dir).resolve()
    source_repo = None if config.task.source else Path(config.workspace.repo_path).resolve()

    task_slug = slugify(config.task.name)
    task_dir = results_dir / task_slug

    # Use explicit run_dir if provided, otherwise generate timestamped one
    if config.workspace.run_dir:
        run_dir = Path(config.workspace.run_dir).resolve()
        task_dir = run_dir.parent
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = task_dir / timestamp
    coral_dir = run_dir / ".coral"
    agents_dir = run_dir / "agents"
    run_repo = run_dir / "repo"

    logger.debug(f"results_dir={results_dir}, task_dir={task_dir}, run_dir={run_dir}")

    # Resolve task directory for relative path resolution
    effective_config_dir = config.task_dir or config_dir or Path.cwd()

    # Create shared state directories.
    # Single-island (count == 1): keep today's exact layout under public/.
    # Multi-island (count > 1):   build islands/<id>/ subtree per island, leave
    #                              public/ minimal (only global meta).
    (coral_dir / "public").mkdir(parents=True, exist_ok=True)
    (coral_dir / "private").mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)

    if config.task.source:
        expected_digest = str(config.grader.args.get("harbor_task_digest", ""))
        if not expected_digest:
            raise ValueError("Resolved Harbor task configuration is missing its source digest")
        stage_local_harbor_task(
            config.task.source,
            base_dir=effective_config_dir,
            private_dir=coral_dir / "private",
            expected_digest=expected_digest,
        )

    if config.islands.count == 1:
        _build_island_subtree(
            coral_dir,
            coral_dir / "public",
            effective_config_dir,
            list(config.agents.skills),
        )
    else:
        # Lazy import: coral.agent's package __init__ pulls in the heavy
        # manager, which imports coral.workspace — importing at module top
        # would form a cycle. nicknames itself has no such deps.
        from coral.agent.nicknames import island_name_for_index

        (coral_dir / "islands").mkdir(parents=True, exist_ok=True)
        for i in range(config.islands.count):
            island_root = coral_dir / "islands" / island_name_for_index(i)
            island_root.mkdir(parents=True, exist_ok=True)
            _build_island_subtree(
                coral_dir,
                island_root,
                effective_config_dir,
                list(config.agents.skills),
            )

    # Save config
    config.to_yaml(coral_dir / "config.yaml")

    # Save config_dir so resume can restore task_dir for relative path resolution
    (coral_dir / "config_dir").write_text(str(effective_config_dir), encoding="utf-8")

    # Create/update "latest" symlink at task_dir/latest -> this run directory
    latest_link = task_dir / "latest"
    if latest_link.is_symlink():
        latest_link.unlink()
    if not latest_link.exists():
        rel = os.path.relpath(run_dir, task_dir)
        latest_link.symlink_to(rel)
        logger.info(f"Symlinked {latest_link} -> {rel}")

    # Clone source repo into run_dir/repo/
    if config.task.source:
        # Gate B starts from an empty, agent-owned workspace.  The Harbor task
        # definition, tests, and solution stay under .coral/private/ and are
        # uploaded only by the manager-side verifier adapter.
        empty_workspace = coral_dir / "private" / "harbor_workspace"
        empty_workspace.mkdir()
        repo_dir = clone_or_init_repo(empty_workspace, run_repo)
    else:
        if source_repo is None:
            raise RuntimeError("Legacy workspace source unexpectedly missing")
        repo_dir = clone_or_init_repo(source_repo, run_repo)

    # Resolve task_dir (directory containing task.yaml)
    task_source_dir = config.task_dir or config_dir or Path.cwd()

    # Auto-copy seed/ into repo (if present in task directory)
    seed_dir = task_source_dir / "seed"
    if not config.task.source and seed_dir.is_dir():
        copy_seed_directory(seed_dir, repo_dir)

    # Copy private grader data into .coral/ (hidden from agents)
    if config.grader.private:
        copy_private_data(config.grader.private, coral_dir, config_dir or Path.cwd())

    # Bootstrap the grader's isolated venv at .coral/private/grader_venv/ and
    # run any user-supplied install steps.
    from coral.workspace.grader_env import setup_grader_env

    setup_grader_env(coral_dir, config.grader, config_dir or Path.cwd())

    return ProjectPaths(
        results_dir=results_dir,
        task_dir=task_dir,
        run_dir=run_dir,
        coral_dir=coral_dir,
        agents_dir=agents_dir,
        repo_dir=repo_dir,
    )


def reconstruct_paths(coral_dir: Path) -> ProjectPaths:
    """Reconstruct ProjectPaths from an existing .coral directory.

    Used by `coral resume` to rebuild paths without creating a new run.
    """
    coral_dir = coral_dir.resolve()
    run_dir = coral_dir.parent
    task_dir = run_dir.parent
    results_dir = task_dir.parent

    return ProjectPaths(
        results_dir=results_dir,
        task_dir=task_dir,
        run_dir=run_dir,
        coral_dir=coral_dir,
        agents_dir=run_dir / "agents",
        repo_dir=run_dir / "repo",
    )
