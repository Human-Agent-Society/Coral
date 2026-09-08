"""REST API endpoints for the CORAL web dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from starlette.requests import Request
from starlette.responses import JSONResponse

from coral.cli._helpers import is_docker_run_alive, is_process_alive


def _coral_dir(request: Request) -> Path:
    return request.app.state.coral_dir


def _run_is_alive(coral_dir: Path) -> bool:
    pid_file = coral_dir / "public" / "manager.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = 0
        if is_process_alive(pid):
            return True
    return is_docker_run_alive(coral_dir, quiet=True)


async def get_config(request: Request) -> JSONResponse:
    """GET /api/config — return the run configuration."""
    config_path = _coral_dir(request) / "config.yaml"
    if not config_path.exists():
        return JSONResponse({"error": "config.yaml not found"}, status_code=404)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return JSONResponse(config)


async def get_attempts(request: Request) -> JSONResponse:
    """GET /api/attempts — return all attempts sorted by timestamp.

    Aggregates across islands so the dashboard reflects the whole team, not
    just the attempts in ``coral_dir/public/attempts`` (which is empty in
    multi-island mode).
    """
    from coral.hub.attempts import _read_all_island_attempts

    attempts = _read_all_island_attempts(_coral_dir(request))
    attempts.sort(key=lambda a: a.timestamp)
    return JSONResponse([a.to_dict() for a in attempts])


async def get_leaderboard(request: Request) -> JSONResponse:
    """GET /api/leaderboard?top=N — return top N attempts by score."""
    from coral.hub.attempts import get_leaderboard as _get_leaderboard

    top_n = int(request.query_params.get("top", "20"))
    attempts = _get_leaderboard(
        str(_coral_dir(request)), top_n=top_n, direction=_direction(request)
    )
    return JSONResponse([a.to_dict() for a in attempts])


async def get_attempt_detail(request: Request) -> JSONResponse:
    """GET /api/attempts/{hash} — return a single attempt.

    Searches every island's attempts dir (and the legacy ``public/attempts``)
    so a hash from any island resolves. Mirrors the cross-island lookup that
    ``coral show`` already does in the CLI.
    """
    from coral.hub._island import all_view_roots

    commit_hash = request.path_params["hash"]
    coral_dir = _coral_dir(request)

    # Direct hit anywhere first.
    for view_root in all_view_roots(coral_dir):
        candidate = view_root / "attempts" / f"{commit_hash}.json"
        if candidate.exists():
            return JSONResponse(json.loads(candidate.read_text(encoding="utf-8")))

    # Prefix match — ambiguous across islands → 404 rather than guessing.
    matches: list[Path] = []
    for view_root in all_view_roots(coral_dir):
        matches.extend((view_root / "attempts").glob(f"{commit_hash}*.json"))
    if len(matches) == 1:
        return JSONResponse(json.loads(matches[0].read_text(encoding="utf-8")))
    return JSONResponse({"error": "attempt not found"}, status_code=404)


async def get_agent_attempts(request: Request) -> JSONResponse:
    """GET /api/attempts/agent/{id} — return attempts for a specific agent."""
    from coral.hub.attempts import get_agent_attempts as _get_agent_attempts

    agent_id = request.path_params["id"]
    attempts = _get_agent_attempts(str(_coral_dir(request)), agent_id)
    return JSONResponse([a.to_dict() for a in attempts])


async def get_dag(request: Request) -> JSONResponse:
    """GET /api/dag — experiment lineage as nodes + edges.

    The DAG is reconstructed from attempt ``parent_hash`` links (which come from
    git parentage at eval time). Attempts whose parent is not itself an attempt
    — e.g. the pre-run baseline commit — become roots (``parent: null``).
    Aggregates across islands so the whole team's lineage is shown.
    """
    from coral.hub.attempts import _read_all_island_attempts
    from coral.hub.attempts import get_leaderboard as _get_leaderboard

    coral_dir = _coral_dir(request)
    attempts = _read_all_island_attempts(coral_dir)
    known = {a.commit_hash for a in attempts}

    best_hash: str | None = None
    user_best_hash: str | None = None
    top = _get_leaderboard(str(coral_dir), top_n=1, direction=_direction(request))
    if top:
        best_hash = top[0].commit_hash
    for attempt in attempts:
        if attempt.metadata.get("user_best") is True:
            user_best_hash = attempt.commit_hash
            break

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    for a in sorted(attempts, key=lambda x: x.timestamp):
        parent = a.parent_hash if a.parent_hash in known else None
        nodes.append(
            {
                "id": a.commit_hash,
                "parent": parent,
                "is_root": parent is None,
                "agent_id": a.agent_id,
                "score": a.score,
                "status": a.status,
                "title": a.title,
                "timestamp": a.timestamp,
                "is_best": a.commit_hash == (user_best_hash or best_hash),
                "user_best": a.commit_hash == user_best_hash,
            }
        )
        if parent is not None:
            edges.append({"from": parent, "to": a.commit_hash})

    return JSONResponse({"nodes": nodes, "edges": edges})


async def get_steering(request: Request) -> JSONResponse:
    """GET /api/steer — list pending steer-on-resume actions."""
    from coral.hub.steering import read_pending

    actions = [a.to_dict() for a in read_pending(_coral_dir(request))]
    return JSONResponse({"actions": actions, "pending_count": len(actions)})


async def post_steer(request: Request) -> JSONResponse:
    """POST /api/steer — queue or apply dashboard steering actions.

    `mark_best` is just a flag on an attempt and applies immediately.
    `continue_from` is queued under `.coral/public/steering/` and consumed by
    `coral resume`. Both are safe to call while the run is alive — the queued
    `continue_from` action simply waits until the user stops and resumes.
    """
    from coral.hub.attempts import set_user_best
    from coral.hub.steering import ContinueFromAction, enqueue

    coral_dir = _coral_dir(request)

    body = await request.json()
    kind = body.get("kind")
    commit_hash = body.get("hash")
    if not isinstance(commit_hash, str) or not commit_hash.strip():
        return JSONResponse({"error": "hash is required"}, status_code=400)
    commit_hash = commit_hash.strip()

    if kind == "continue_from":
        instruction = body.get("instruction", "")
        if not isinstance(instruction, str):
            return JSONResponse({"error": "instruction must be a string"}, status_code=400)
        action = enqueue(
            coral_dir,
            ContinueFromAction(hash=commit_hash, instruction=instruction.strip()),
        )
        return JSONResponse({"action": action.to_dict()})

    if kind == "mark_best":
        attempt = set_user_best(coral_dir, commit_hash)
        if attempt is None:
            return JSONResponse({"error": "attempt not found"}, status_code=404)
        return JSONResponse(
            {
                "action": {
                    "kind": "mark_best",
                    "hash": commit_hash,
                    "applied": True,
                }
            }
        )

    return JSONResponse({"error": "unknown steering action kind"}, status_code=400)


async def get_notes(request: Request) -> JSONResponse:
    """GET /api/notes — return all notes, including raw/ source captures.

    The dashboard groups by ``category`` and has a "Raw Sources" bucket, so it
    opts into raw here; the CLI/agent-facing callers of list_notes do not.
    """
    from coral.hub.notes import list_notes

    entries = list_notes(str(_coral_dir(request)), include_raw=True, audit=True)
    for i, entry in enumerate(entries):
        entry["index"] = i
    return JSONResponse(entries)


async def get_notes_graph(request: Request) -> JSONResponse:
    """GET /api/notes/graph — notes as a connection graph (nodes + edges)."""
    from coral.hub.notes import notes_graph

    return JSONResponse(notes_graph(str(_coral_dir(request))))


async def get_skills(request: Request) -> JSONResponse:
    """GET /api/skills — return all skills."""
    from coral.hub.skills import list_skills

    skills = list_skills(str(_coral_dir(request)))
    # Convert any non-string values (e.g. datetime from YAML) to strings
    for sk in skills:
        for key in ("created", "updated"):
            if sk.get(key) and not isinstance(sk[key], str):
                sk[key] = str(sk[key])
    return JSONResponse(skills)


async def get_skill_detail(request: Request) -> JSONResponse:
    """GET /api/skills/{name} — return a specific skill."""
    from coral.hub._island import all_view_roots
    from coral.hub.skills import read_skill

    name = request.path_params["name"]
    coral_dir = _coral_dir(request)
    skill_dir = None
    for view_root in all_view_roots(coral_dir):
        candidate = view_root / "skills" / name
        if candidate.is_dir():
            skill_dir = candidate
            break
    if skill_dir is None:
        return JSONResponse({"error": "skill not found"}, status_code=404)

    info = read_skill(skill_dir)
    return JSONResponse(info)


async def get_logs(request: Request) -> JSONResponse:
    """GET /api/logs/{agent_id} — return parsed log turns for an agent."""
    from coral.web.logs import list_log_files, parse_log_file

    agent_id = request.path_params["agent_id"]
    coral_dir = _coral_dir(request)
    agent_logs = list_log_files(coral_dir)

    if agent_id not in agent_logs:
        return JSONResponse({"error": "agent not found"}, status_code=404)

    # Parse all log files for this agent, grouped by session
    sessions: list[dict[str, Any]] = []
    all_session_metas: list[dict[str, Any]] = []
    global_turn_idx = 0
    for log_info in sorted(agent_logs[agent_id], key=lambda x: x["index"]):
        turns, _, session_meta = parse_log_file(Path(log_info["path"]))
        session_turns = []
        for t in turns:
            td = t.to_dict()
            td["index"] = global_turn_idx
            global_turn_idx += 1
            session_turns.append(td)
        session_data: dict[str, Any] = {
            "session_index": log_info["index"],
            "turns": session_turns,
        }
        if session_meta:
            session_data["meta"] = session_meta.to_dict()
            all_session_metas.append(session_meta.to_dict())
        sessions.append(session_data)

    # Also flatten for backward compat
    all_turns = [t for s in sessions for t in s["turns"]]

    # Aggregate session-level metadata for the whole agent
    agent_meta: dict[str, Any] | None = None
    if all_session_metas:
        total_cost = sum(m.get("total_cost_usd") or 0 for m in all_session_metas)
        total_duration = sum(m.get("duration_ms") or 0 for m in all_session_metas)
        total_api_duration = sum(m.get("duration_api_ms") or 0 for m in all_session_metas)
        total_turns = sum(m.get("num_turns") or 0 for m in all_session_metas)
        # Aggregate usage across sessions
        agg_usage: dict[str, int] = {}
        for m in all_session_metas:
            for k, v in m.get("usage", {}).items():
                if isinstance(v, int | float):
                    agg_usage[k] = agg_usage.get(k, 0) + int(v)
        agent_meta = {
            "total_cost_usd": total_cost,
            "duration_ms": total_duration,
            "duration_api_ms": total_api_duration,
            "num_turns": total_turns,
            "usage": agg_usage,
        }

    return JSONResponse(
        {
            "agent_id": agent_id,
            "log_files": agent_logs[agent_id],
            "turns": all_turns,
            "sessions": sessions,
            "agent_meta": agent_meta,
        }
    )


async def get_logs_list(request: Request) -> JSONResponse:
    """GET /api/logs — return available agents and their log files."""
    from coral.web.logs import list_log_files

    agent_logs = list_log_files(_coral_dir(request))
    return JSONResponse(agent_logs)


def _direction(request: Request) -> str:
    """Read grader direction from config. Returns 'maximize' or 'minimize'."""
    config_path = _coral_dir(request) / "config.yaml"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        return (config.get("grader") or {}).get("direction", "maximize")
    return "maximize"


def _results_dir(request: Request) -> Path:
    return request.app.state.results_dir


def _catalog_root(request: Request) -> Path:
    return getattr(request.app.state, "catalog_root", _results_dir(request).parent)


def _available_results_dirs(request: Request) -> tuple[Path, ...]:
    from coral.web.run_catalog import discover_results_dirs

    current_results_dir = _coral_dir(request).resolve().parent.parent.parent
    return discover_results_dirs(_catalog_root(request), current_results_dir)


def _enumerate_results_dir(results_dir: Path) -> list[dict[str, Any]]:
    """Return the tasks and runs stored in one results directory."""
    tasks = []
    if not results_dir.is_dir():
        return tasks

    for task_dir in sorted(results_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_slug = task_dir.name

        # Resolve "latest" symlink target
        latest_link = task_dir / "latest"
        latest_target = None
        if latest_link.is_symlink():
            try:
                latest_target = latest_link.resolve()
            except OSError:
                pass

        runs = []
        for run_dir in sorted(task_dir.iterdir(), reverse=True):
            if not run_dir.is_dir() or run_dir.is_symlink():
                continue
            coral_dir = run_dir / ".coral"
            if not coral_dir.is_dir():
                continue

            # Check manager status
            status = "running" if _run_is_alive(coral_dir) else "stopped"

            # Count attempts across every view root. In multi-island mode the
            # attempts live in islands/<id>/attempts/ — public/attempts is
            # empty — so a single-dir glob would undercount every run.
            from coral.hub._island import all_view_roots

            attempt_count = 0
            for view_root in all_view_roots(coral_dir):
                attempts_dir = view_root / "attempts"
                if attempts_dir.is_dir():
                    attempt_count += sum(1 for _ in attempts_dir.glob("*.json"))

            # Check if latest (latest symlink now points to run_dir, not .coral)
            is_latest = latest_target is not None and latest_target == run_dir.resolve()

            runs.append(
                {
                    "timestamp": run_dir.name,
                    "status": status,
                    "attempts": attempt_count,
                    "is_latest": is_latest,
                }
            )

        if runs:
            tasks.append({"slug": task_slug, "runs": runs})

    return tasks


def _enumerate_runs(
    results_dirs: tuple[Path, ...],
    current_coral_dir: Path,
    catalog_root: Path,
) -> dict[str, Any]:
    """Return every run catalog available to this dashboard."""
    from coral.web.run_catalog import results_dir_id, results_dir_label

    current_resolved = current_coral_dir.resolve()
    current_results_dir = current_resolved.parent.parent.parent
    current_root_id = results_dir_id(current_results_dir)
    current_task = current_resolved.parent.parent.name
    current_run = current_resolved.parent.name

    roots = []
    current_tasks: list[dict[str, Any]] = []
    for results_dir in results_dirs:
        tasks = _enumerate_results_dir(results_dir)
        root_id = results_dir_id(results_dir)
        roots.append(
            {
                "id": root_id,
                "label": results_dir_label(results_dir, catalog_root),
                "tasks": tasks,
            }
        )
        if root_id == current_root_id:
            current_tasks = tasks

    return {
        "current": {"root": current_root_id, "task": current_task, "run": current_run},
        # Keep the original field for older dashboard clients. New clients use
        # roots so task slugs can safely repeat in separate results catalogs.
        "tasks": current_tasks,
        "roots": roots,
    }


async def get_runs(request: Request) -> JSONResponse:
    """GET /api/runs — list all tasks and runs."""
    coral_dir = _coral_dir(request)
    data = _enumerate_runs(
        _available_results_dirs(request),
        coral_dir,
        _catalog_root(request),
    )
    return JSONResponse(data)


async def switch_run(request: Request) -> JSONResponse:
    """POST /api/runs/switch — switch to a different run."""
    import asyncio

    from coral.web.events import FileWatcher

    body = await request.json()
    root_id = body.get("root")
    task = body.get("task")
    run = body.get("run")
    if not task or not run:
        return JSONResponse({"error": "task and run required"}, status_code=400)

    from coral.web.run_catalog import results_dir_id

    available_results_dirs = _available_results_dirs(request)
    if root_id:
        results_dir = next(
            (path for path in available_results_dirs if results_dir_id(path) == root_id),
            None,
        )
        if results_dir is None:
            return JSONResponse({"error": "run catalog not found"}, status_code=404)
    else:
        # Backward compatibility for clients created before catalogs existed.
        results_dir = _results_dir(request)

    if Path(task).name != task or Path(run).name != run:
        return JSONResponse({"error": "run not found"}, status_code=404)

    task_dir = results_dir / task
    run_dir = task_dir / run
    new_coral_dir = run_dir / ".coral"
    if (
        not task_dir.is_dir()
        or task_dir.is_symlink()
        or not run_dir.is_dir()
        or run_dir.is_symlink()
        or not new_coral_dir.is_dir()
        or new_coral_dir.is_symlink()
    ):
        return JSONResponse({"error": "run not found"}, status_code=404)
    new_coral_dir = new_coral_dir.resolve()

    app = request.app

    async with app.state._switch_lock:
        # Stop old watcher
        old_watcher = app.state.watcher
        old_watcher.stop()
        app.state._watcher_task.cancel()
        try:
            await app.state._watcher_task
        except asyncio.CancelledError:
            pass

        # Switch coral_dir
        app.state.coral_dir = new_coral_dir
        app.state.results_dir = results_dir

        # Start new watcher, reusing subscriber list
        new_watcher = FileWatcher(
            app.state.coral_dir,
            subscribers=old_watcher._subscribers,
        )
        app.state.watcher = new_watcher
        app.state._watcher_task = asyncio.create_task(new_watcher.run())

        # Broadcast switch event
        new_watcher._broadcast(
            {
                "event": "run:switched",
                "data": {"root": results_dir_id(results_dir), "task": task, "run": run},
            }
        )

    return JSONResponse({"ok": True, "root": results_dir_id(results_dir), "task": task, "run": run})


async def get_status(request: Request) -> JSONResponse:
    """GET /api/status — return overall run status."""
    from coral.web.logs import list_log_files

    coral_dir = _coral_dir(request)

    # Manager liveness
    pid_file = coral_dir / "public" / "manager.pid"
    manager_alive = False
    manager_pid = None
    if pid_file.exists():
        try:
            manager_pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            manager_pid = None
        if manager_pid is not None:
            manager_alive = is_process_alive(manager_pid)
    is_docker = not manager_alive and is_docker_run_alive(coral_dir, quiet=True)
    if is_docker:
        manager_alive = True

    # Eval count
    from coral.hub.attempts import read_eval_count

    eval_count = read_eval_count(coral_dir)

    # Attempts summary — aggregate across islands so the status pane shows
    # the whole team, not just public/attempts (empty in multi-island mode).
    from coral.hub.attempts import _read_all_island_attempts

    attempts = _read_all_island_attempts(coral_dir)
    scored = [a for a in attempts if a.score is not None]
    minimize = _direction(request) == "minimize"
    best_fn = min if minimize else max
    best = best_fn(scored, key=lambda a: a.score or 0.0) if scored else None

    # Per-agent status
    agent_logs = list_log_files(coral_dir)
    agents_status: list[dict[str, Any]] = []

    # Read per-agent PID map for process liveness checks.
    # Skip for Docker runs — container-internal PIDs aren't valid on the host.
    agent_pid_map: dict[str, int] = {}
    pid_map_file = coral_dir / "public" / "agent_pids.json"
    if not is_docker and pid_map_file.exists():
        try:
            agent_pid_map = json.loads(pid_map_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback: read agent.pids (plain PID list) and check if any are alive
    any_agent_alive = False
    if not agent_pid_map:
        pids_file = coral_dir / "public" / "agent.pids"
        if pids_file.exists():
            try:
                for line in pids_file.read_text(encoding="utf-8").strip().splitlines():
                    if is_process_alive(int(line.strip())):
                        any_agent_alive = True
                        break
            except (ValueError, OSError):
                pass

    # If agent processes are alive but manager.pid is missing, treat as alive
    if not manager_alive and (agent_pid_map or any_agent_alive):
        manager_alive = True

    import time

    for agent_id, logs in agent_logs.items():
        latest = max(logs, key=lambda log: log["modified"])
        age = time.time() - latest["modified"]

        agent_pid = agent_pid_map.get(agent_id)
        if agent_pid:
            # Direct PID check — most reliable
            status = "active" if is_process_alive(agent_pid) else "stopped"
        elif any_agent_alive or is_docker:
            # Container or agent.pids says something is running but no per-agent mapping
            status = "active" if age < 300 else "idle"
        else:
            # No PID info — log recency as last resort
            status = "active" if age < 120 else "stopped"

        agent_attempts = [a for a in attempts if a.agent_id == agent_id]
        agent_scored = [a for a in agent_attempts if a.score is not None]
        agent_best = best_fn(agent_scored, key=lambda a: a.score or 0.0) if agent_scored else None

        agents_status.append(
            {
                "agent_id": agent_id,
                "status": status,
                "sessions": len(logs),
                "last_activity": latest["modified"],
                "attempts": len(agent_attempts),
                "best_score": agent_best.score if agent_best else None,
            }
        )

    return JSONResponse(
        {
            "manager_alive": manager_alive,
            "manager_pid": manager_pid,
            "eval_count": eval_count,
            "total_attempts": len(attempts),
            "scored_attempts": len(scored),
            "crashed_attempts": len([a for a in attempts if a.status == "crashed"]),
            "best_score": best.score if best else None,
            "best_title": best.title if best else None,
            "agents": agents_status,
        }
    )
