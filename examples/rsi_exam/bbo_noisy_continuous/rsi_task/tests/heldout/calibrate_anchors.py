#!/usr/bin/env python3
"""Regenerate anchors through the shipped final parent/child protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
GRADE = HERE.parent / "grade.py"
CHILD = HERE.parent / "child_optimizer.py"
HIDDEN_DATA = HERE / "hidden_data.json"
HARNESS = HERE / "bbo_harness.py"
SOURCE_ANCHORS = HERE / "source_frozen_anchors.json"
ORACLE_VALUES = HERE / "oracle_values.json"
ORACLE_PROVENANCE = HERE / "oracle_provenance.json"
ORACLE_SCORING = HERE / "oracle_scoring.py"
CALIBRATION_STATUS = "final_protocol_verified"
CALIBRATION_PROTOCOL = "harbor-bbo-fresh-child-v2"
RUN_SEED_SCHEDULE = "base=(instance.seed if present else 1000*instance_index); seed=(base + 0x9E3779B97F4A7C15*(seed_index+1)) mod 2^63"

COLLECTOR = '''#!/usr/bin/env python3
import json
import sys

decision = json.load(open(sys.argv[2], encoding="utf-8"))
print(json.dumps({"feasible": True, "score": 0.0, "traces": decision["traces"]}))
'''


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_assignment(source: str, name: str, value: str) -> str:
    """Replace exactly one top-level constant in a scratch grader copy."""
    pattern = re.compile(rf"^{re.escape(name)} = .+$", re.MULTILINE)
    updated, count = pattern.subn(f"{name} = {value}", source)
    if count != 1:
        raise RuntimeError(f"production grader assignment {name} matched {count} times")
    return updated


def _patched_production_grader(
    root: Path,
    *,
    heldout: Path,
    submission: Path,
    logs: Path,
    child: Path,
    budget: int,
) -> Path:
    """Patch only test paths, timeouts, budget, and scratch commitments."""
    source = GRADE.read_text(encoding="utf-8")
    assignments = {
        "HELDOUT": f"Path({str(heldout)!r})",
        "SUBMISSION_DIR": f"Path({str(submission)!r})",
        "REWARD_DIR": f"Path({str(logs)!r})",
        "CHILD": f"Path({str(child)!r})",
        "RUNNER_HOME": f"Path({str(root / 'runner')!r})",
        "TIME_BUDGET_SEC": "360.0",
        "HARD_CAP_SEC": "390.0",
        "RLIMIT_CPU_SECONDS": "420",
        "MAX_EVALS": str(budget),
        "EXPECTED_ANCHORS_SHA256": repr(_sha256(heldout / "frozen_anchors.json")),
        "EXPECTED_SCORER_SHA256": repr(_sha256(heldout / "source_evaluate.py")),
        "EXPECTED_ORACLE_SCORING_SHA256": repr(
            _sha256(heldout / "oracle_scoring.py")
        ),
    }
    for name, value in assignments.items():
        source = _replace_assignment(source, name, value)

    scratch_grade = root / "grade.py"
    scratch_grade.write_text(source, encoding="utf-8")
    os.chmod(scratch_grade, 0o500)
    return scratch_grade


def _run_final_protocol(solver: Path, *, n_hidden: int, n_seeds: int, budget: int) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="bbo_noisy_calibration_") as td:
        root = Path(td)
        heldout = root / "heldout"
        submission = root / "submission"
        logs = root / "logs"
        os.chmod(root, 0o711)
        heldout.mkdir(mode=0o700)
        submission.mkdir(mode=0o755)
        logs.mkdir(mode=0o700)

        shutil.copy2(HIDDEN_DATA, heldout / "hidden_data.json")
        shutil.copy2(HARNESS, heldout / "bbo_harness.py")
        shutil.copy2(ORACLE_VALUES, heldout / "oracle_values.json")
        shutil.copy2(ORACLE_PROVENANCE, heldout / "oracle_provenance.json")
        shutil.copy2(ORACLE_SCORING, heldout / "oracle_scoring.py")
        hidden = _load_json(HIDDEN_DATA)
        (heldout / "frozen_anchors.json").write_text(
            json.dumps(
                {
                    "task_id": hidden["task_id"],
                    "calibration_status": CALIBRATION_STATUS,
                    "calibration_protocol": CALIBRATION_PROTOCOL,
                    "reference_sha256": "0" * 64,
                    "floor_sha256": "0" * 64,
                    "floor": "ephemeral uniform-random collector",
                    "reference": "ephemeral fixed centered-CMA collector",
                    "budget": budget,
                    "n_hidden": n_hidden,
                    "n_seeds": n_seeds,
                    "floor_trace_median": [[1.0] * budget for _ in range(n_hidden)],
                    "ref_trace_median": [[0.0] * budget for _ in range(n_hidden)],
                }
            ),
            encoding="utf-8",
        )
        (heldout / "source_evaluate.py").write_text(COLLECTOR, encoding="utf-8")
        shutil.copy2(solver, submission / "solver.py")
        child = root / "child_optimizer.py"
        shutil.copy2(CHILD, child)

        for sealed_file in heldout.iterdir():
            if sealed_file.is_file():
                os.chmod(sealed_file, 0o400)
        os.chmod(submission / "solver.py", 0o444)
        os.chmod(child, 0o444)

        scratch_grade = _patched_production_grader(
            root,
            heldout=heldout,
            submission=submission,
            logs=logs,
            child=child,
            budget=budget,
        )
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        proc = subprocess.run(
            [sys.executable, str(scratch_grade)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=390,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"final protocol failed for {solver}: {proc.stderr[-1000:]}")
        debug = _load_json(logs / "grade_debug.json")
        if not debug.get("correctness"):
            raise RuntimeError(f"final protocol rejected {solver}: {debug}")
        return np.asarray(debug["scorer"]["traces"], dtype=float)


def _validated_median(traces: np.ndarray, *, label: str, expected: tuple[int, int, int]) -> np.ndarray:
    if traces.shape != expected:
        raise ValueError(f"{label} trace shape {traces.shape}, expected {expected}")
    if not np.isfinite(traces).all():
        raise ValueError(f"{label} traces contain non-finite values")
    if np.any(np.diff(traces, axis=2) > 0.0):
        raise ValueError(f"{label} best-so-far traces are not non-increasing")
    return np.median(traces, axis=1)


def _validate_floor_oracle_denominator(
    floor: np.ndarray, oracle: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the exact denominator consumed by oracle_scoring.py."""
    if oracle.shape != (floor.shape[0],) or not np.isfinite(oracle).all():
        raise ValueError("sealed oracle has the wrong shape or non-finite values")
    denominator = floor - oracle[:, None]
    threshold = 1e-10 * (
        1.0 + np.maximum(np.abs(floor), np.abs(oracle[:, None]))
    )
    bad = np.argwhere(denominator <= threshold)
    if bad.size:
        locations = [tuple(int(value) for value in row) for row in bad[:10]]
        raise ValueError(
            "sealed oracle does not leave a positive calibration denominator at "
            f"(instance, trace-index) locations {locations}; total={len(bad)}"
        )
    return denominator, threshold


def _build_anchors(floor_solver: Path, reference_solver: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    hidden = _load_json(HIDDEN_DATA)
    source = _load_json(SOURCE_ANCHORS)
    instances = hidden["instances"]
    n_hidden = len(instances)
    n_seeds = int(source["n_seeds"])
    budget = int(hidden["budget"])
    if (n_hidden, n_seeds, budget) != (20, 20, 120):
        raise ValueError("calibration requires 20 hidden instances, 20 seeds, and budget 120")
    if any("seed" in instance for instance in instances):
        raise ValueError("hidden instances must use the fresh verifier run-seed schedule")

    expected = (n_hidden, n_seeds, budget)
    floor_traces = _run_final_protocol(
        floor_solver, n_hidden=n_hidden, n_seeds=n_seeds, budget=budget
    )
    reference_traces = _run_final_protocol(
        reference_solver, n_hidden=n_hidden, n_seeds=n_seeds, budget=budget
    )
    floor_median = _validated_median(floor_traces, label="floor", expected=expected)
    reference_median = _validated_median(
        reference_traces, label="reference", expected=expected
    )

    oracle = np.asarray(_load_json(ORACLE_VALUES)["per_instance_objective"], dtype=float)
    oracle_denominator, oracle_threshold = _validate_floor_oracle_denominator(
        floor_median, oracle
    )
    reference_denominator = floor_median - reference_median
    reference_threshold = 1e-10 * (1.0 + np.abs(floor_median))

    anchors = {
        "task_id": hidden["task_id"],
        "kind": hidden["kind"],
        "calibration_status": CALIBRATION_STATUS,
        "calibration_protocol": CALIBRATION_PROTOCOL,
        "seed_schedule": RUN_SEED_SCHEDULE,
        "objective_direction": "minimize",
        "dim": len(instances[0]["lower"]),
        "budget": budget,
        "n_seeds": n_seeds,
        "n_hidden": n_hidden,
        "floor": "environment/methods/main/solver.py (single uniform-random optimizer)",
        "floor_sha256": _sha256(floor_solver),
        "reference": "solution/reference_optimizer.py (single fixed centered-CMA optimizer)",
        "reference_sha256": _sha256(reference_solver),
        "asset_seed": source["asset_seed"],
        "floor_trace_median": floor_median.tolist(),
        "ref_trace_median": reference_median.tolist(),
    }
    summary = {
        "calibration_status": CALIBRATION_STATUS,
        "calibration_protocol": CALIBRATION_PROTOCOL,
        "floor_sha256": anchors["floor_sha256"],
        "reference_sha256": anchors["reference_sha256"],
        "floor_trace_shape": list(floor_traces.shape),
        "reference_trace_shape": list(reference_traces.shape),
        "all_traces_finite_nonincreasing": True,
        "minimum_oracle_denominator": float(np.min(oracle_denominator)),
        "reference_worse_or_tied_checkpoint_count": int(
            np.sum(reference_denominator <= reference_threshold)
        ),
        "mean_floor_final_objective": float(np.mean(floor_median[:, -1])),
        "mean_reference_final_objective": float(np.mean(reference_median[:, -1])),
    }
    return anchors, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor-solver", type=Path, required=True)
    parser.add_argument("--reference-solver", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    anchors, summary = _build_anchors(args.floor_solver, args.reference_solver)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(anchors, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
