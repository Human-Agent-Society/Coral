"""Eval submission: git-add, git-commit, write pending attempt, optionally wait.

The grading itself happens asynchronously in the grader daemon
(coral/grader/daemon.py). `submit_eval` only stages+commits, writes a
pending attempt record, and optionally polls for the final score.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from coral.config import CoralConfig
from coral.hub.attempts import (
    agent_in_grader_queue,
    count_agent_pending,
    increment_eval_count,
    read_attempt,
    read_eval_count,
    write_attempt,
)
from coral.hub.checkpoint import checkpoint
from coral.types import BUDGET_CLASS_TUNE, Attempt
from coral.workspace.breadcrumbs import find_coral_breadcrumb, read_island_breadcrumb

# Legacy alias — external tests/hooks may still import the underscore-prefixed
# name. Prefer `coral.hub.attempts.increment_eval_count` directly.
_increment_eval_count = increment_eval_count

logger = logging.getLogger(__name__)

# How often submit_eval(wait=True) polls the attempt file for score updates.
_POLL_INTERVAL_SEC = 0.2


def _git_add_and_commit(message: str, workdir: str) -> str:
    """Stage all changes and commit. Returns the new commit hash."""
    # Stage all changes
    result = subprocess.run(
        ["git", "add", "-A"],
        capture_output=True,
        text=True,
        cwd=workdir,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git add failed: {result.stderr}")

    # Check if there's anything to commit
    status = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        capture_output=True,
        cwd=workdir,
    )
    if status.returncode == 0:
        raise RuntimeError("Nothing to commit — no changes detected.")

    # Commit
    result = subprocess.run(
        ["git", "commit", "-m", message],
        capture_output=True,
        text=True,
        cwd=workdir,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git commit failed: {result.stderr}")

    # Get the commit hash
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=workdir,
    )
    return result.stdout.strip()


def _get_parent_hash(commit_hash: str, cwd: str) -> str | None:
    """Get the parent commit hash."""
    result = subprocess.run(
        ["git", "log", "--format=%P", "-n", "1", commit_hash],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().split()[0]
    return None


def _find_coral_dir(workdir: Path) -> Path | None:
    """Find the shared .coral directory from the .coral_dir breadcrumb file."""
    found = find_coral_breadcrumb(workdir)
    if found is None:
        return None
    coral_dir, _breadcrumb_dir = found
    return coral_dir


def _poll_until_graded(
    coral_dir: Path,
    commit_hash: str,
    timeout: float,
    island_id: str | None = None,
) -> Attempt:
    """Poll the attempt file until status != 'pending' or timeout elapses.

    Raises TimeoutError if no grader finalizes the attempt within `timeout` seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        attempt = read_attempt(coral_dir, commit_hash, island_id=island_id)
        if attempt is not None and attempt.status != "pending":
            return attempt
        time.sleep(_POLL_INTERVAL_SEC)
    raise TimeoutError(
        f"Grader did not finalize attempt {commit_hash[:12]} within {timeout:.0f}s "
        f"(is the grader daemon running?)"
    )


def submit_eval(
    message: str,
    agent_id: str,
    workdir: str = ".",
    wait: bool = True,
    poll_timeout: float | None = None,
    tune: bool = False,
    operator: str | None = None,
    mutation: str | None = None,
) -> Attempt:
    """Stage changes, commit with message, write a pending attempt record.

    If ``wait`` is True (default), also polls the attempt file until the
    grader daemon finalizes it (score populated, status != "pending") and
    returns the final Attempt. If False, returns immediately with a pending
    Attempt — the caller (or a future `coral wait` invocation) is responsible
    for observing the final result.

    If ``tune`` is True, the attempt is marked as a tune-mode submission
    (``budget_class="tune"`` on its metadata). The grader still runs and the
    score is recorded, but the manager will not count it toward the agent's
    plateau / heartbeat budget — see issue #73.

    This is the core of `coral eval -m "description"` on the agent side.
    The grader itself runs asynchronously in `coral.grader.daemon`.
    """
    workdir_path = Path(workdir).resolve()

    breadcrumb = find_coral_breadcrumb(workdir_path)
    if breadcrumb is None:
        raise FileNotFoundError(f"No .coral directory found from {workdir_path}")
    coral_dir, breadcrumb_dir = breadcrumb

    config_path = coral_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml found at {config_path}")
    config = CoralConfig.from_yaml(config_path)

    from coral.agent.meta_evolve import attribution_metadata, validate_attribution

    try:
        attribution = validate_attribution(
            config.agents.meta_evolve,
            operator=operator,
            mutation=mutation,
            tune=tune,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    # Determine the agent's island (if any) from the .coral_island breadcrumb
    # adjacent to the .coral_dir breadcrumb we discovered above.
    island_id = read_island_breadcrumb(coral_dir, breadcrumb_dir)

    # Producer-side queue cap: refuse to commit when this agent already has
    # `max_pending_per_agent` ungraded submissions in flight. The grader is
    # serial; without this cap, a slow grader plus a fast agent piles up
    # arbitrarily many pending JSONs (issue #80). 0 = unlimited (legacy).
    pending_limit = config.grader.max_pending_per_agent
    if pending_limit > 0:
        pending_count = count_agent_pending(coral_dir, agent_id, island_id=island_id)
        if pending_count >= pending_limit:
            oldest = agent_in_grader_queue(coral_dir, agent_id, island_id=island_id)
            wait_hint = (
                f"Run `coral wait {oldest.commit_hash[:12]}` to block on it "
                f"before submitting again."
                if oldest is not None
                else "Wait for the prior eval to finish before submitting again."
            )
            raise RuntimeError(
                f"You already have {pending_count} pending attempt(s) "
                f"(limit: {pending_limit}). {wait_hint}"
            )

    # Git add + commit
    commit_hash = _git_add_and_commit(message, str(workdir_path))
    parent_hash = _get_parent_hash(commit_hash, str(workdir_path))

    # Checkpoint shared state at submission time (captures agent's current notes/skills).
    shared_state_hash = checkpoint(str(coral_dir), agent_id, message, island_id=island_id)

    # Look up parent attempt's shared state hash for provenance chain.
    parent_shared_state_hash = None
    if parent_hash:
        from coral.hub._island import island_root

        parent_attempt_file = island_root(coral_dir, island_id) / "attempts" / f"{parent_hash}.json"
        if parent_attempt_file.exists():
            try:
                parent_data = json.loads(parent_attempt_file.read_text(encoding="utf-8"))
                parent_shared_state_hash = parent_data.get("shared_state_hash")
            except (json.JSONDecodeError, OSError):
                pass

    # Write pending record. The grader daemon will observe this and fill in
    # score/status/feedback asynchronously.
    metadata: dict = {}
    if attribution is not None:
        metadata.update(attribution_metadata(attribution))
    if tune:
        metadata["budget_class"] = BUDGET_CLASS_TUNE
    if island_id is not None:
        metadata["island_id"] = island_id
    attempt = Attempt(
        commit_hash=commit_hash,
        agent_id=agent_id,
        title=message,
        score=None,
        status="pending",
        parent_hash=parent_hash,
        timestamp=datetime.now(UTC).isoformat(),
        feedback="",
        shared_state_hash=shared_state_hash,
        parent_shared_state_hash=parent_shared_state_hash,
        metadata=metadata,
    )
    write_attempt(str(coral_dir), attempt, island_id=island_id)

    if not wait:
        return attempt

    # Block until grader daemon finalizes. We give it plenty of slack above the
    # grader's own per-eval timeout so the daemon has room to finish + write back.
    if poll_timeout is None:
        grader_timeout = config.grader.timeout if config.grader.timeout > 0 else 0
        # 2x the grader budget + 60s slack, with a floor of 300s for fast graders.
        poll_timeout = max(grader_timeout * 2 + 60, 300) if grader_timeout else 3600

    final = _poll_until_graded(coral_dir, commit_hash, poll_timeout, island_id=island_id)

    # Attach eval_count for display by cmd_eval (best-effort; daemon bumps this).
    try:
        final._eval_count = read_eval_count(coral_dir, island_id=island_id)  # type: ignore[attr-defined]
    except Exception:
        pass

    return final


# Backward-compat alias: older callers / hooks may still import `run_eval`.
# Same semantics as submit_eval(wait=True).
def run_eval(message: str, agent_id: str, workdir: str = ".") -> Attempt:
    """Deprecated. Prefer `submit_eval`. Synchronous (waits for grader)."""
    return submit_eval(message=message, agent_id=agent_id, workdir=workdir, wait=True)
