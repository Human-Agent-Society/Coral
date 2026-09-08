#!/usr/bin/env python3
"""Trusted DiscoveryWorld host for a submitted line-oriented harness."""
from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
from knowledge_eval import evaluate_knowledge

VISIBLE_CASES = [
    {
        "id": "space-visible-0",
        "scenario": "Space Sick",
        "difficulty": "Challenge",
        "seed": 0,
    },
    {
        "id": "space-visible-1",
        "scenario": "Space Sick",
        "difficulty": "Challenge",
        "seed": 1,
    },
    {
        "id": "space-visible-2",
        "scenario": "Space Sick",
        "difficulty": "Challenge",
        "seed": 2,
    },
    {
        "id": "chemistry-visible-0",
        "scenario": "Combinatorial Chemistry",
        "difficulty": "Challenge",
        "seed": 0,
    },
    {
        "id": "chemistry-visible-1",
        "scenario": "Combinatorial Chemistry",
        "difficulty": "Challenge",
        "seed": 1,
    },
    {
        "id": "chemistry-visible-2",
        "scenario": "Combinatorial Chemistry",
        "difficulty": "Challenge",
        "seed": 2,
    },
]


def _send(proc: subprocess.Popen[str], payload: dict[str, Any]) -> None:
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
    proc.stdin.flush()


def _receive(proc: subprocess.Popen[str], timeout: float) -> dict[str, Any]:
    assert proc.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    try:
        ready = selector.select(timeout)
    finally:
        selector.close()
    if not ready:
        raise TimeoutError(f"harness action exceeded {timeout:.0f}s")
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("harness exited without returning an action")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("harness action must be a JSON object")
    return value


def _task_description(observation: dict[str, Any]) -> str:
    return "\n".join(
        item.get("description", "")
        for item in observation["ui"].get("taskProgress", [])
    ).strip()


def _agent_observation(api: DiscoveryWorldAPI) -> dict[str, Any]:
    observation = api.getAgentObservation(agentIdx=0)
    observation.pop("vision", None)
    return observation


def _stop(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=3)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_episode(
    case: dict[str, Any],
    harness_path: str,
    max_steps: int,
    log_dir: Path,
    action_timeout: float = 200.0,
) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    thread_id = 10000 + int(case["seed"]) * 100 + sum(map(ord, case["id"])) % 97
    api = DiscoveryWorldAPI(threadID=thread_id)
    if not api.loadScenario(
        scenarioName=case["scenario"],
        difficultyStr=case.get("difficulty", "Challenge"),
        randomSeed=int(case["seed"]),
        numUserAgents=1,
    ):
        raise RuntimeError(f"failed to load {case['id']}")
    observation = _agent_observation(api)

    stderr_file = (log_dir / "harness.stderr.log").open("w")
    identity: dict[str, Any] = {}
    if (
        os.getuid() == 0
        and os.environ.get("DISCOVERYWORLD_VISIBLE_SELF_CHECK") != "1"
    ):
        # Popen's native credential arguments are safe when episodes launch
        # concurrently; preexec_fn is not safe in a threaded parent.
        identity = {"user": 10001, "group": 10001, "extra_groups": []}
    proc = subprocess.Popen(
        [sys.executable, harness_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_file,
        text=True,
        bufsize=1,
        start_new_session=True,
        cwd="/tmp",
        env=os.environ.copy(),
        **identity,
    )
    trajectory = (log_dir / "trajectory.jsonl").open("w")
    error = None
    steps = 0
    thoughts: list[str] = []
    try:
        _send(
            proc,
            {
                "type": "init",
                "task": _task_description(observation),
                "known_actions": api.listKnownActions(limited=False),
                "action_help": api.additionalActionDescriptionString(),
                "max_steps": max_steps,
            },
        )
        for step in range(max_steps):
            _send(
                proc,
                {
                    "type": "observation",
                    "step": step,
                    "observation": observation,
                    "teleport_locations": api.listTeleportLocationsDict(),
                },
            )
            try:
                action = _receive(proc, action_timeout)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                break
            thought = action.get("thought")
            if isinstance(thought, str) and thought.strip():
                thoughts.append(thought)
            if action.get("action") == "SUBMIT":
                trajectory.write(
                    json.dumps(
                        {"step": step, "action": action, "result": {"submitted": True}}
                    )
                    + "\n"
                )
                break
            result = api.performAgentAction(agentIdx=0, actionJSON=action)
            api.tick()
            steps = step + 1
            observation = _agent_observation(api)
            safe_result = dict(result)
            safe_result["success"] = bool(result.get("success"))
            trajectory.write(
                json.dumps(
                    {"step": step, "action": action, "result": safe_result},
                    ensure_ascii=False,
                )
                + "\n"
            )
            trajectory.flush()
            if api.areTasksComplete():
                break
    finally:
        _stop(proc)
        trajectory.close()
        stderr_file.close()

    scorecard = api.getTaskScorecard()[0]
    procedure = float(scorecard.get("scoreNormalized", 0.0))
    completion = float(bool(scorecard.get("completedSuccessfully", False)))
    knowledge, knowledge_correct, knowledge_total, knowledge_error = (
        evaluate_knowledge(
            task_description=str(scorecard.get("taskDescription", "")),
            critical_questions=[
                str(value) for value in scorecard.get("criticalQuestions", [])
            ],
            thoughts=thoughts,
        )
    )
    result = {
        "id": case["id"],
        "scenario": case["scenario"],
        "difficulty": case.get("difficulty", "Challenge"),
        "procedure": procedure,
        "completion": completion,
        "knowledge": knowledge,
        "knowledge_correct": knowledge_correct,
        "knowledge_total": knowledge_total,
        "knowledge_error": knowledge_error,
        "raw_metric": (procedure + completion + knowledge) / 3.0,
        "steps": steps,
        "wall_sec": time.time() - started,
        "error": error,
    }
    (log_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def evaluate(
    cases: list[dict[str, Any]],
    harness_path: str,
    max_steps: int,
    log_root: str,
) -> dict[str, Any]:
    root = Path(log_root)
    root.mkdir(parents=True, exist_ok=True)
    def evaluate_one(case: dict[str, Any]) -> dict[str, Any]:
        print(f"running {case['id']} ({case['scenario']})", flush=True)
        try:
            return run_episode(
                case,
                harness_path=harness_path,
                max_steps=max_steps,
                log_dir=root / case["id"],
            )
        except Exception as exc:
            return {
                "id": case["id"],
                "scenario": case["scenario"],
                "difficulty": case.get("difficulty", "Challenge"),
                "procedure": 0.0,
                "completion": 0.0,
                "knowledge": 0.0,
                "knowledge_correct": 0,
                "knowledge_total": 0,
                "knowledge_error": "episode failed before knowledge evaluation",
                "raw_metric": 0.0,
                "steps": 0,
                "wall_sec": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }

    workers = max(
        1,
        min(len(cases), int(os.environ.get("DISCOVERYWORLD_EVAL_WORKERS", "1"))),
    )
    if workers == 1:
        instances = [evaluate_one(case) for case in cases]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            instances = list(pool.map(evaluate_one, cases))
    return {
        "mean_raw_metric": sum(item["raw_metric"] for item in instances)
        / len(instances),
        "mean_procedure": sum(item["procedure"] for item in instances)
        / len(instances),
        "mean_completion": sum(item["completion"] for item in instances)
        / len(instances),
        "mean_knowledge": sum(item["knowledge"] for item in instances)
        / len(instances),
        "instances": instances,
    }
