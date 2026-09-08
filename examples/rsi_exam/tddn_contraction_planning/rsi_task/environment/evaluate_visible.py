"""Run an editable planner on every frozen visible case."""
from __future__ import annotations

import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any

from tdd_backend import make_environment


def load_solver(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("submitted_solver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import solver from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "select_edge", None)):
        raise TypeError("solver.py must define callable select_edge(observation)")
    return module


def enabled(observation: dict[str, Any], edge_id: int) -> bool:
    mask = observation["action_mask"]
    return bool(mask.get(str(edge_id), mask.get(edge_id, False)))


def run_case(solver: Any, case: dict[str, Any]) -> dict[str, Any]:
    planning_seconds = 0.0
    with make_environment(case) as environment:
        while not environment.is_terminal:
            observation = environment.observation()
            started = time.perf_counter()
            action = int(solver.select_edge(observation))
            planning_seconds += time.perf_counter() - started
            if not enabled(observation, action):
                raise RuntimeError(f"invalid masked edge_id {action}")
            environment.contract(action)
        result = environment.result()
    contraction_seconds = float(result["contraction_time_seconds"])
    return {
        "case_id": case["case_id"],
        "family": case["family"],
        "category": case["category"],
        "correct": bool(result["correct"]),
        "steps": int(result["steps"]),
        "peak_tdd_nodes": int(result["peak_tdd_nodes"]),
        "planning_time_seconds": planning_seconds,
        "contraction_time_seconds": contraction_seconds,
        "total_time_seconds": planning_seconds + contraction_seconds,
    }


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def evaluate(methods_dir: Path, cases_path: Path) -> dict[str, Any]:
    solver = load_solver(methods_dir / "solver.py")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for case in cases:
        try:
            rows.append(run_case(solver, case))
        except Exception as error:  # one bad case must not hide the others
            rows.append({
                "case_id": case["case_id"],
                "family": case["family"],
                "category": case["category"],
                "correct": False,
                "error": f"{type(error).__name__}: {error}",
            })

    correct = all(row.get("correct") is True for row in rows)
    good = [row for row in rows if row.get("correct") is True]
    return {
        "correct": correct,
        "cases_passed": len(good),
        "cases_total": len(rows),
        "peak_tdd_nodes_geomean": geometric_mean(
            [float(row["peak_tdd_nodes"]) for row in good]
        ),
        "total_time_seconds_geomean": geometric_mean(
            [float(row["total_time_seconds"]) for row in good]
        ),
        "cases": rows,
    }
