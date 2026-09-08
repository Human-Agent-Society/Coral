"""Commands: log (attempts), show (attempt), notes, skills, runs, plot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from coral.cli._helpers import (
    find_coral_dir_and_island,
    is_docker_container_running,
    is_process_alive,
    read_direction,
)


def cmd_log(args: argparse.Namespace) -> None:
    """List and search attempts (leaderboard).

    Examples:
      coral log                     Top 20 attempts by score
      coral log -n 5                Top 5
      coral log --recent            Sort by time instead of score
      coral log --agent agent-1     Filter by agent
      coral log --search "kernel"   Full-text search
      coral log --all               Include tune + grader_error attempts
      coral log --class tune        Show only tune-mode attempts
    """
    from coral.hub.attempts import (
        format_leaderboard,
        get_agent_attempts,
        get_leaderboard,
        get_recent,
        search_attempts,
    )
    from coral.types import BUDGET_CLASS_REAL

    coral_dir, island_id = find_coral_dir_and_island(
        getattr(args, "task", None),
        getattr(args, "run", None),
    )
    direction = read_direction(coral_dir)
    count = getattr(args, "count", None) or 20
    show_all = getattr(args, "all", False)
    only_class = getattr(args, "budget_class", None)
    # Over-fetch when filtering, so the trimmed result still has up to `count`
    # rows even when many recent / top attempts happen to be tune or error.
    raw_n = count if (show_all or only_class) else max(count * 4, 40)

    def filter_attempts(attempts):
        if only_class:
            return [a for a in attempts if a.budget_class == only_class]
        if show_all:
            return attempts
        return [a for a in attempts if a.budget_class == BUDGET_CLASS_REAL]

    if args.search:
        attempts = filter_attempts(
            search_attempts(str(coral_dir), args.search, island_id=island_id)
        )[:count]
        if attempts:
            print(f"Search results for '{args.search}':")
            print(format_leaderboard(attempts))
        else:
            print(f"No attempts matching '{args.search}'.")
    elif args.agent:
        attempts = filter_attempts(
            get_agent_attempts(str(coral_dir), args.agent, island_id=island_id)
        )[:count]
        if attempts:
            print(f"Attempts by {args.agent}:")
            print(format_leaderboard(attempts))
        else:
            print(f"No attempts by {args.agent}.")
    elif args.recent:
        attempts = filter_attempts(get_recent(str(coral_dir), n=raw_n, island_id=island_id))[:count]
        if attempts:
            print(f"Recent {len(attempts)} attempt(s):")
            print(format_leaderboard(attempts))
        else:
            print("No attempts yet.")
    else:
        # `coral log` does its own tune/error filtering via --all / --class, so
        # always pull the full set from get_leaderboard and let the filter
        # callback narrow it. get_leaderboard's default hides tune, but log
        # wants everything up front.
        attempts = filter_attempts(
            get_leaderboard(
                str(coral_dir),
                top_n=raw_n,
                direction=direction,
                island_id=island_id,
                include_tune=True,
            )
        )[:count]
        if attempts:
            print(f"Leaderboard (top {len(attempts)}):")
            print(format_leaderboard(attempts))
        else:
            print("No attempts yet.")


def cmd_show(args: argparse.Namespace) -> None:
    """Show details of a specific attempt.

    Examples:
      coral show abc123             Show attempt by hash prefix
      coral show <full-hash>        Show attempt by full hash
    """
    from coral.hub._island import all_view_roots, island_root

    coral_dir, island_id = find_coral_dir_and_island(
        getattr(args, "task", None),
        getattr(args, "run", None),
    )
    # In multi-island mode the attempt lives in islands/<id>/attempts/, not
    # public/attempts/. When a worktree is in scope, look only at that
    # island (an agent must not see another island's attempts); otherwise
    # sweep every view root so a user outside any worktree can resolve a
    # hash from any island.
    if island_id is not None:
        attempts_dirs = [island_root(coral_dir, island_id) / "attempts"]
    else:
        attempts_dirs = [r / "attempts" for r in all_view_roots(coral_dir)]
    candidates: list[Path] = []
    for attempts_dir in attempts_dirs:
        if not attempts_dir.is_dir():
            continue
        direct = attempts_dir / f"{args.hash}.json"
        if direct.exists():
            candidates.append(direct)
        else:
            candidates.extend(attempts_dir.glob(f"{args.hash}*.json"))

    if not candidates:
        print(f"Attempt {args.hash} not found.")
        return
    if len(candidates) > 1:
        print(f"Ambiguous hash prefix '{args.hash}'. Matches:")
        for m in candidates:
            print(f"  {m.relative_to(coral_dir)}")
        return
    attempt_file = candidates[0]

    data = json.loads(attempt_file.read_text(encoding="utf-8"))
    print(f"Commit:  {data['commit_hash']}")
    print(f"Agent:   {data['agent_id']}")
    print(f"Title:   {data['title']}")
    print(f"Score:   {data.get('score', '—')}")
    print(f"Status:  {data['status']}")
    from coral.types import get_budget_class

    print(f"Budget:  {get_budget_class(data.get('metadata'))}")
    meta = data.get("metadata") or {}
    from coral.agent.meta_evolve import attempt_attribution
    from coral.types import Attempt

    attribution = attempt_attribution(Attempt.from_dict(data))
    if attribution is not None:
        print(f"Meta:    {attribution.operator} / {attribution.mutation}")
    if meta.get("archived") is True:
        reason = meta.get("archive_reason")
        print(f"Archived: yes ({reason})" if reason else "Archived: yes")
    print(f"Time:    {data['timestamp']}")
    if data.get("parent_hash"):
        print(f"Parent:  {data['parent_hash']}")
    if data.get("feedback"):
        print(f"Feedback: {data['feedback']}")

    commit = data["commit_hash"]
    git_args = ["git", "show", commit]
    if not getattr(args, "diff", False):
        git_args.insert(2, "--stat")
    result = subprocess.run(
        git_args,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        label = "Diff" if getattr(args, "diff", False) else "Summary"
        print(f"\n--- {label} ---\n{result.stdout}")


def cmd_notes(args: argparse.Namespace) -> None:
    """Browse shared notes.

    Examples:
      coral notes                   List all notes
      coral notes -n 5              Last 5 notes
      coral notes --search "idea"   Search notes
      coral notes --read 3          Read note #3
    """
    from coral.hub.notes import (
        format_notes_list,
        get_recent_notes,
        list_notes,
        read_all_notes,
        read_note,
        search_notes,
    )

    coral_dir, island_id = find_coral_dir_and_island(
        getattr(args, "task", None),
        getattr(args, "run", None),
    )
    status = getattr(args, "status", None)

    if getattr(args, "audit", False):
        from coral.hub.notes_audit import format_audits

        if any(getattr(args, key, None) for key in ("history", "diff", "read")):
            raise SystemExit("--audit cannot be combined with --read, --history, or --diff")
        entries = list_notes(str(coral_dir), island_id=island_id, status=status, audit=True)
        if getattr(args, "search", None):
            query = args.search.lower()
            entries = [e for e in entries if query in (e["title"] + "\n" + e["body"]).lower()]
        if getattr(args, "recent", None) is not None:
            if args.recent < 1:
                raise SystemExit("--recent must be positive with --audit")
            entries = entries[-args.recent :]
        print(format_audits(entries))
        return

    if getattr(args, "history", False):
        from coral.hub.checkpoint import checkpoint_history

        entries = checkpoint_history(str(coral_dir), island_id=island_id)
        if not entries:
            print("No checkpoint history.")
            return
        print(f"{'HASH':<12} {'DATE':<26} MESSAGE")
        print("-" * 72)
        for e in entries:
            print(f"{e['hash'][:10]}   {e['date']:<26} {e['message']}")
        return

    if getattr(args, "diff", None):
        from coral.hub.checkpoint import checkpoint_diff

        print(checkpoint_diff(str(coral_dir), args.diff, island_id=island_id))
        return

    if args.read:
        try:
            idx = int(args.read)
            entry = read_note(str(coral_dir), idx, island_id=island_id)
            if entry:
                print(entry)
            else:
                print(f"Note #{idx} not found.")
        except ValueError:
            print(read_all_notes(str(coral_dir), island_id=island_id))
    elif args.search:
        results = search_notes(
            str(coral_dir),
            args.search,
            island_id=island_id,
            status=status,
        )
        if results:
            print(f"Notes matching '{args.search}':")
            print(format_notes_list(results))
        else:
            print(f"No notes matching '{args.search}'.")
    elif args.recent:
        entries = get_recent_notes(
            str(coral_dir),
            n=args.recent,
            island_id=island_id,
            status=status,
        )
        print(f"Recent notes ({len(entries)}):")
        print(format_notes_list(entries))
    else:
        entries = list_notes(str(coral_dir), island_id=island_id, status=status)
        print(f"Notes ({len(entries)}):")
        print(format_notes_list(entries))


def cmd_skills(args: argparse.Namespace) -> None:
    """Browse shared skills.

    Examples:
      coral skills                  List all skills
      coral skills --read optim     Show skill by name (or prefix)
    """
    from coral.hub._island import all_view_roots, island_root
    from coral.hub.skills import format_skills_list, list_skills, read_skill

    coral_dir, island_id = find_coral_dir_and_island(
        getattr(args, "task", None),
        getattr(args, "run", None),
    )

    if args.read:
        # Same scoping as cmd_show: a worktree caller can only see its own
        # island's skills; a user outside a worktree sees the union.
        if island_id is not None:
            search_roots = [island_root(coral_dir, island_id) / "skills"]
        else:
            search_roots = [r / "skills" for r in all_view_roots(coral_dir)]
        skill_dirs: list[Path] = []
        for skills_dir in search_roots:
            if not skills_dir.is_dir():
                continue
            direct = skills_dir / args.read
            if direct.is_dir():
                skill_dirs.append(direct)
            else:
                skill_dirs.extend(
                    d for d in skills_dir.iterdir() if d.is_dir() and d.name.startswith(args.read)
                )

        if not skill_dirs:
            print(f"Skill '{args.read}' not found.")
            return
        # Dedup by absolute path (same skill on multiple islands) and tag
        # the island for the user.
        seen_paths: set[Path] = set()
        unique: list[Path] = []
        for d in skill_dirs:
            if d in seen_paths:
                continue
            seen_paths.add(d)
            unique.append(d)
        if len(unique) > 1:
            print(f"Ambiguous name '{args.read}'. Matches:")
            for m in unique:
                print(f"  {m}")
            return
        skill_dir = unique[0]
        info = read_skill(skill_dir)
        print(info["content"])
        if info["files"]:
            print(f"\nFiles: {', '.join(info['files'])}")
    else:
        skills = list_skills(str(coral_dir), island_id=island_id)
        print(f"Skills ({len(skills)}):")
        print(format_skills_list(skills))


def _find_results_dir() -> Path:
    """Walk up from cwd to find the results/ directory."""
    current = Path.cwd()
    while True:
        candidate = current / "results"
        if candidate.is_dir():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    print("No results/ directory found.", file=sys.stderr)
    sys.exit(1)


def _relative_time(timestamp_str: str) -> str:
    """Convert a run timestamp like '2026-03-11_163000' to a relative time string."""
    from datetime import datetime

    try:
        dt = datetime.strptime(timestamp_str, "%Y-%m-%d_%H%M%S")
    except ValueError:
        return timestamp_str
    delta = datetime.now() - dt
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        m = seconds // 60
        return f"{m}m ago"
    if seconds < 86400:
        h = seconds // 3600
        return f"{h}h ago"
    d = seconds // 86400
    return f"{d}d ago"


def _collect_runs(results_dir: Path) -> list[dict]:
    """Scan results/ and collect metadata for all runs."""
    runs = []
    for task_dir in sorted(results_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name

        latest_link = task_dir / "latest"
        latest_resolved = None
        if latest_link.is_symlink():
            try:
                latest_resolved = latest_link.resolve()
            except OSError:
                pass

        for run_dir in sorted(task_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.is_symlink():
                continue
            coral_dir = run_dir / ".coral"
            if not coral_dir.is_dir():
                continue

            pid_file = coral_dir / "public" / "manager.pid"
            status = "stopped"
            manager_pid = None

            # Check Docker container first — PIDs in manager.pid are
            # container-internal and meaningless on the host.
            docker_marker = run_dir / ".coral_docker_container"
            if docker_marker.exists():
                container_name = docker_marker.read_text(encoding="utf-8").strip()
                if container_name and is_docker_container_running(container_name):
                    status = "running"
            elif pid_file.exists():
                try:
                    manager_pid = int(pid_file.read_text(encoding="utf-8").strip())
                except ValueError:
                    status = "stopped"
                else:
                    status = "running" if is_process_alive(manager_pid) else "stopped"

            logs_dir = coral_dir / "public" / "logs"
            agent_names: set[str] = set()
            if logs_dir.exists():
                for lf in logs_dir.glob("*.log"):
                    parts = lf.stem.rsplit(".", 1)
                    agent_names.add(parts[0] if len(parts) == 2 else lf.stem)

            # Load config for model/runtime/grader/direction info
            config_file = coral_dir / "config.yaml"
            model = ""
            runtime = ""
            direction = "maximize"
            if config_file.exists():
                try:
                    import yaml

                    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
                    agents_cfg = cfg.get("agents", {})
                    model = agents_cfg.get("model", "")
                    runtime = agents_cfg.get("runtime", "")
                    grader_cfg = cfg.get("grader", {})
                    direction = grader_cfg.get("direction", "maximize")
                except Exception:
                    pass

            # Aggregate attempts across every view root. In multi-island mode
            # the JSONs live in islands/<id>/attempts/ — public/attempts is
            # empty — so a single-dir glob would undercount every run.
            # BEST mirrors format_status_summary: hide tune attempts (they're
            # sweeps, not submissions) so the headline number matches what
            # `coral log` and the body summary show by default.
            from coral.hub._island import all_view_roots
            from coral.types import BUDGET_CLASS_REAL, Attempt

            attempt_count = 0
            best_score = None
            for view_root in all_view_roots(coral_dir):
                attempts_dir = view_root / "attempts"
                if not attempts_dir.is_dir():
                    continue
                for af in attempts_dir.glob("*.json"):
                    try:
                        adata = json.loads(af.read_text(encoding="utf-8"))
                        attempt_count += 1
                        # `budget_class` is a derived field on Attempt
                        # (from metadata), not a top-level JSON key — go
                        # through Attempt.from_dict so the classification
                        # logic stays in one place.
                        if Attempt.from_dict(adata).budget_class != BUDGET_CLASS_REAL:
                            continue
                        s = adata.get("score")
                        if s is not None:
                            if best_score is None:
                                best_score = s
                            elif direction == "maximize" and s > best_score:
                                best_score = s
                            elif direction == "minimize" and s < best_score:
                                best_score = s
                    except (json.JSONDecodeError, KeyError, OSError):
                        attempt_count += 1

            is_latest = latest_resolved is not None and (latest_resolved == run_dir.resolve())

            runs.append(
                {
                    "task": task_name,
                    "run": run_dir.name,
                    "status": status,
                    "pid": manager_pid,
                    "agents": len(agent_names),
                    "attempts": attempt_count,
                    "best": best_score,
                    "model": model,
                    "runtime": runtime,
                    "latest": is_latest,
                    "path": str(run_dir),
                }
            )
    return runs


def cmd_runs(args: argparse.Namespace) -> None:
    """List CORAL runs.

    Examples:
      coral runs                    Active runs only
      coral runs --all              Include stopped runs
      coral runs --task my-task     Filter by task
      coral runs -n 5              Show at most 5 runs
    """
    results_dir = _find_results_dir()
    show_all = getattr(args, "all", False)
    task_filter = getattr(args, "task", None)
    count = getattr(args, "count", None) or 20
    verbose = getattr(args, "verbose", False)

    runs = _collect_runs(results_dir)

    # Filter by task name
    if task_filter:
        runs = [r for r in runs if task_filter in r["task"]]

    # Filter: active only unless --all
    if not show_all:
        runs = [r for r in runs if r["status"] == "running"]

    # Sort: running first, then by run timestamp descending (most recent first)
    runs.sort(key=lambda r: (r["status"] != "running", r["run"]), reverse=False)
    # Reverse the run name sort within each group for most-recent-first
    running = [r for r in runs if r["status"] == "running"]
    stopped = [r for r in runs if r["status"] != "running"]
    running.sort(key=lambda r: r["run"], reverse=True)
    stopped.sort(key=lambda r: r["run"], reverse=True)
    runs = running + stopped

    # Apply limit
    total = len(runs)
    runs = runs[:count]

    if not runs:
        if show_all:
            print("No runs found.")
        else:
            print("No active runs. Use --all to see stopped runs.")
        return

    # Compute column widths from data
    tw = max(len("TASK"), max((len(r["task"]) for r in runs), default=4)) + 2
    rw = max(len("RUN"), max((len(r["run"]) + 2 for r in runs), default=3)) + 2  # +2 for " *"
    sw = max(len("STATUS"), 20)
    mw = max(len("MODEL"), max((len(r["model"]) for r in runs), default=5)) + 2
    rtw = max(len("RUNTIME"), max((len(r["runtime"]) for r in runs), default=7)) + 2

    header = (
        f"{'TASK':<{tw}}{'RUN':<{rw}}{'STATUS':<{sw}}"
        f"{'AGENTS':>8}{'EVALS':>8}{'BEST':>10}"
        f"  {'MODEL':<{mw}}{'RUNTIME':<{rtw}}"
    )
    if verbose:
        header += "  PATH"
    print(header)
    print("-" * len(header))

    for r in runs:
        latest_marker = " *" if r["latest"] else ""
        run_col = f"{r['run']}{latest_marker}"
        if r["status"] == "running":
            status_str = f"running (PID {r['pid']})" if r["pid"] else "running"
        else:
            status_str = f"stopped {_relative_time(r['run'])}"

        best_str = f"{r['best']:.4f}" if r["best"] is not None else "-"
        line = (
            f"{r['task']:<{tw}}{run_col:<{rw}}{status_str:<{sw}}"
            f"{r['agents']:>8}{r['attempts']:>8}{best_str:>10}"
            f"  {r['model']:<{mw}}{r['runtime']:<{rtw}}"
        )
        if verbose:
            line += f"  {r['path']}"
        print(line)

    # Summary
    running_count = sum(1 for r in runs if r["status"] == "running")
    print()
    summary = f"{total} run(s)"
    if not show_all:
        summary += " active"
    elif running_count:
        summary += f", {running_count} running"
    if total > count:
        summary += f" (showing {count})"
    summary += "  (* = latest)"
    print(summary)
