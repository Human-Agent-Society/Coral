"""Commands: eval, wait, revert, diff, checkout, export."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from coral.cli._helpers import find_coral_dir_and_island, read_agent_id

# Poll cadence used by `coral wait` while blocking on the attempt file.
_WAIT_POLL_INTERVAL_SEC = 0.2


def _resolve_attempt_file(
    coral_dir: Path,
    target: str,
    island_id: str | int | None,
) -> Path | None:
    """Locate the attempt JSON file for ``target`` (full hash or prefix).

    Scoped to a single island when ``island_id`` is set (worktree mode);
    swept across every view root (``islands/<id>`` + ``public``, single-
    island runs have just one root) otherwise. Returns the file path so
    the caller can re-read it as the grader daemon updates it, or None
    if no file matches.

    Ambiguous (multiple matches across view roots) is reported as None so
    the caller can print the standard "No attempt matches" message; the
    global ambiguity case is rare in practice because hash prefixes long
    enough to collide across islands almost never appear in one run.
    """
    from coral.hub._island import all_view_roots, island_root

    if island_id is not None:
        attempts_dirs = [island_root(coral_dir, island_id) / "attempts"]
    else:
        attempts_dirs = [r / "attempts" for r in all_view_roots(coral_dir)]

    if len(target) < 40:
        matches: list[Path] = []
        for d in attempts_dirs:
            if d.is_dir():
                matches.extend(d.glob(f"{target}*.json"))
        if len(matches) == 1:
            return matches[0]
        return None  # 0 or >1 matches: caller prints the standard message

    # Exact 40-char hash: find the file directly.
    for d in attempts_dirs:
        candidate = d / f"{target}.json"
        if candidate.is_file():
            return candidate
    return None


def _read_attempt_file(attempts_file: Path):
    """Read + parse an attempt JSON, returning ``None`` on any I/O error.

    Mirrors the tolerant half of :func:`coral.hub.attempts.read_attempt`
    but takes a file path directly so callers that already resolved the
    file (e.g. after a multi-island sweep) don't have to re-glob.
    """
    import json as _json

    from coral.types import Attempt

    if not attempts_file.exists():
        return None
    try:
        return Attempt.from_dict(_json.loads(attempts_file.read_text(encoding="utf-8")))
    except (_json.JSONDecodeError, KeyError, OSError):
        return None


def cmd_eval(args: argparse.Namespace) -> None:
    """Stage changes, commit, and submit evaluation (blocking by default)."""
    from coral.hooks.post_commit import submit_eval

    workdir = args.workdir or "."
    agent_id = args.agent or read_agent_id(workdir)
    wait = getattr(args, "wait", True)
    timeout = getattr(args, "timeout", None)
    tune = getattr(args, "tune", False)
    operator = getattr(args, "operator", None)
    mutation = getattr(args, "mutation", None)

    try:
        attempt = submit_eval(
            message=args.message,
            agent_id=agent_id,
            workdir=workdir,
            wait=wait,
            poll_timeout=timeout,
            tune=tune,
            operator=operator,
            mutation=mutation,
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except TimeoutError as e:
        print(
            f"\n{'=' * 50}\n"
            f"CORAL Eval: STILL PENDING\n"
            f"{e}\n"
            f"Use `coral wait <hash>` to keep waiting, or check `coral log`.\n"
            f"{'=' * 50}\n",
            file=sys.stderr,
        )
        sys.exit(2)

    _print_attempt_result(attempt, header="CORAL Eval")


def cmd_wait(args: argparse.Namespace) -> None:
    """Block until a previously submitted attempt is finalized by the grader."""

    from coral.config import CoralConfig
    from coral.hub.attempts import read_eval_count

    # Resolve coral_dir + island scope from breadcrumbs/--task. When invoked
    # from a worktree, island_id scopes the hash lookup to that agent's
    # island; outside a worktree (or with explicit --task) we aggregate.
    try:
        coral_dir, island_id = find_coral_dir_and_island(
            getattr(args, "task", None),
            getattr(args, "run", None),
        )
    except Exception as e:
        print(f"Error: Could not locate .coral directory: {e}", file=sys.stderr)
        sys.exit(1)

    # Find the attempt file. Same scoping rules as cmd_checkout: scoped to
    # a single island in a worktree, swept across all view roots otherwise.
    # We resolve to a concrete file path (not just a hash) so the poll
    # loop can re-read the same JSON the grader daemon updates in place.
    target = args.hash
    attempts_file = _resolve_attempt_file(coral_dir, target, island_id)
    if attempts_file is None:
        if len(target) >= 40:
            print(f"Error: No attempt matches '{target}'.", file=sys.stderr)
        else:
            print(f"Error: No attempt matches '{target}'.", file=sys.stderr)
        sys.exit(1)
    target = attempts_file.stem  # normalized 40-char hash

    # Derive timeout.
    timeout = args.timeout
    if timeout is None:
        try:
            config = CoralConfig.from_yaml(coral_dir / "config.yaml")
            grader_timeout = config.grader.timeout if config.grader.timeout > 0 else 0
            timeout = max(grader_timeout * 2 + 60, 300) if grader_timeout else 3600
        except Exception:
            timeout = 3600

    deadline = time.monotonic() + timeout
    attempt = None
    while time.monotonic() < deadline:
        attempt = _read_attempt_file(attempts_file)
        if attempt is not None and attempt.status != "pending":
            try:
                attempt._eval_count = read_eval_count(coral_dir, island_id=island_id)  # type: ignore[attr-defined]
            except Exception:
                pass
            _print_attempt_result(attempt, header="CORAL Wait")
            return
        time.sleep(_WAIT_POLL_INTERVAL_SEC)

    if attempt is None:
        print(f"Error: No attempt found for '{target}'.", file=sys.stderr)
        sys.exit(1)
    print(
        f"\n{'=' * 50}\n"
        f"CORAL Wait: STILL PENDING\n"
        f"Attempt {target[:12]} not graded within {timeout:.0f}s.\n"
        f"Re-run `coral wait` to keep waiting, or check `coral status`.\n"
        f"{'=' * 50}\n",
        file=sys.stderr,
    )
    sys.exit(2)


def _print_attempt_result(attempt, header: str) -> None:
    """Shared formatter for `coral eval` and `coral wait` output."""
    score_str = f"{attempt.score:.10f}" if attempt.score is not None else "FAILED"
    if attempt.status == "pending":
        score_str = "PENDING"
    eval_count = getattr(attempt, "_eval_count", None)
    count_str = f" (#{eval_count})" if eval_count else ""
    print(f"\n{'=' * 50}")
    print(f"{header}{count_str}: {score_str}")
    print(f"Commit:  {attempt.commit_hash[:12]}")
    from coral.types import BUDGET_CLASS_REAL

    status_line = attempt.status
    budget_class = attempt.budget_class
    if budget_class != BUDGET_CLASS_REAL:
        status_line = f"{status_line}  (budget: {budget_class})"
    print(f"Status:  {status_line}")
    from coral.agent.meta_evolve import attempt_attribution

    attribution = attempt_attribution(attempt)
    if attribution is not None:
        print(f"Meta:    {attribution.operator} / {attribution.mutation}")
    if attempt.feedback:
        print(f"Feedback: {attempt.feedback}")
    if attempt.status == "pending":
        print(
            "Tip: grader is still working. "
            f"Run `coral wait {attempt.commit_hash[:12]}` to block on the result."
        )
    print(f"{'=' * 50}\n")


def cmd_revert(args: argparse.Namespace) -> None:
    """Revert to the last commit (undo uncommitted changes and last commit)."""
    workdir = args.workdir or "."

    result = subprocess.run(
        ["git", "log", "--oneline", "-1"],
        capture_output=True,
        text=True,
        cwd=workdir,
    )
    if result.returncode != 0:
        print("Error: No commits to revert.", file=sys.stderr)
        sys.exit(1)

    result = subprocess.run(
        ["git", "reset", "--hard", "HEAD~1"],
        capture_output=True,
        text=True,
        cwd=workdir,
    )
    if result.returncode != 0:
        print(f"Error: git reset failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)


def cmd_checkout(args: argparse.Namespace) -> None:
    """Checkout a previous attempt's code by commit hash."""
    workdir = args.workdir or "."
    target = args.hash

    coral_dir, island_id = find_coral_dir_and_island(
        getattr(args, "task", None),
        getattr(args, "run", None),
    )
    # Resolve to a known attempt file before touching git. When a worktree
    # caller asks for a hash that lives only on another island, we want a
    # clear "No attempt matches" message (and a quiet return — checkout is
    # destructive, so we don't sys.exit on miss; the agent can decide).
    attempts_file = _resolve_attempt_file(coral_dir, target, island_id)
    if attempts_file is None:
        print(f"Error: No attempt matches '{target}'.", file=sys.stderr)
        return
    target = attempts_file.stem  # normalized 40-char hash

    result = subprocess.run(
        ["git", "cat-file", "-t", target],
        capture_output=True,
        text=True,
        cwd=workdir,
    )
    if result.returncode != 0:
        print(f"Error: Commit '{target}' not found.", file=sys.stderr)
        sys.exit(1)

    log_result = subprocess.run(
        ["git", "log", "--oneline", "-1", target],
        capture_output=True,
        text=True,
        cwd=workdir,
    )
    print(f"Checking out: {log_result.stdout.strip()}")

    result = subprocess.run(
        ["git", "reset", "--hard", target],
        capture_output=True,
        text=True,
        cwd=workdir,
    )
    if result.returncode != 0:
        print(f"Error: git reset failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)


def cmd_export(args: argparse.Namespace) -> None:
    """Export an attempt's commit as a normal git branch in the run's repo.

    The attempt commits live in the shared object store of the run's source
    clone (``<run>/repo``), reachable from every agent worktree. Exporting
    creates an ordinary branch there so the user can ``git checkout`` it and
    continue with a normal git workflow.
    """
    target = args.hash
    branch = args.branch

    coral_dir, island_id = find_coral_dir_and_island(
        getattr(args, "task", None),
        getattr(args, "run", None),
    )
    attempts_file = _resolve_attempt_file(coral_dir, target, island_id)
    if attempts_file is None:
        print(f"Error: No attempt matches '{target}'.", file=sys.stderr)
        sys.exit(1)
    target = attempts_file.stem  # normalized 40-char hash

    # coral_dir = <run>/.coral → the source clone is <run>/repo.
    repo_dir = coral_dir.parent / "repo"
    if not repo_dir.is_dir():
        print(f"Error: run repo not found at {repo_dir}.", file=sys.stderr)
        sys.exit(1)

    # The commit object must be reachable from the repo's object store.
    if (
        subprocess.run(
            ["git", "cat-file", "-t", target],
            capture_output=True,
            text=True,
            cwd=repo_dir,
        ).returncode
        != 0
    ):
        print(f"Error: commit '{target}' not found in {repo_dir}.", file=sys.stderr)
        sys.exit(1)

    branch_exists = (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
        ).returncode
        == 0
    )
    if branch_exists and not args.force:
        print(
            f"Error: branch '{branch}' already exists. Use --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = ["git", "branch"]
    if args.force:
        cmd.append("-f")
    cmd += [branch, target]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_dir)
    if result.returncode != 0:
        print(f"Error: git branch failed: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    print(f"Exported {target[:12]} → branch '{branch}' in {repo_dir}")
    print(f"  cd {repo_dir} && git checkout {branch}")


def cmd_diff(args: argparse.Namespace) -> None:
    """Show current uncommitted changes."""
    workdir = args.workdir or "."

    result = subprocess.run(
        ["git", "diff", "HEAD"],
        capture_output=True,
        text=True,
        cwd=workdir,
    )
    if result.returncode != 0:
        result = subprocess.run(
            ["git", "diff"],
            capture_output=True,
            text=True,
            cwd=workdir,
        )

    if result.stdout:
        print(result.stdout)
    else:
        status = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True,
            cwd=workdir,
        )
        if status.stdout:
            print(status.stdout)
        else:
            print("No changes.")
