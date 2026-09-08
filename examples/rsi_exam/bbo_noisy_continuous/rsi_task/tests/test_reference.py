"""Regression tests for the fixed noisy-continuous reference and anchors."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import shutil
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path

import numpy as np


TASK_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = TASK_ROOT / "tests"
HELDOUT_ROOT = TESTS_ROOT / "heldout"
FLOOR_SOLVER_DIR = TASK_ROOT / "environment" / "methods" / "main"
REFERENCE_SOLVER = TASK_ROOT / "solution" / "reference_optimizer.py"
ANCHORS = TESTS_ROOT / "heldout" / "frozen_anchors.json"
STRICT_TOLERANCE = 1e-8
REFERENCE_DIAGNOSTIC_TOLERANCE = 0.01


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_optimizer():
    if not REFERENCE_SOLVER.is_file():
        raise AssertionError("fixed reference optimizer has not been created")
    spec = importlib.util.spec_from_file_location("noisy_reference", REFERENCE_SOLVER)
    if spec is None or spec.loader is None:
        raise AssertionError("reference optimizer is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Optimizer


class TestReferenceAPI(unittest.TestCase):
    def test_reference_is_one_importable_final_protocol_optimizer(self) -> None:
        optimizer = _load_optimizer()
        self.assertEqual(optimizer.__name__, "Optimizer")
        self.assertEqual(
            list(inspect.signature(optimizer).parameters),
            ["dim", "lower", "upper", "budget", "seed", "rng"],
        )
        instance = optimizer(
            dim=10,
            lower=np.full(10, -5.0),
            upper=np.full(10, 5.0),
            budget=120,
            seed=7,
            rng=np.random.default_rng(7),
        )
        points = np.asarray(instance.ask(8), dtype=float)
        self.assertEqual(points.shape, (7, 10))
        instance.tell(points, np.arange(len(points), dtype=float))


class FinalProtocolReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image = f"bbo-noisy-reference:{uuid.uuid4().hex}"
        proc = subprocess.run(
            [
                "docker",
                "build",
                "--quiet",
                "--tag",
                cls.image,
                "--file",
                str(TESTS_ROOT / "Dockerfile"),
                str(TESTS_ROOT),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"verifier image build failed:\n{proc.stdout}\n{proc.stderr}")

    @classmethod
    def tearDownClass(cls) -> None:
        subprocess.run(
            ["docker", "image", "rm", "--force", cls.image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def _run_verifier(self, solver_dir: Path) -> tuple[dict, dict]:
        with tempfile.TemporaryDirectory(prefix=".reference-logs-", dir=TASK_ROOT) as td:
            root = Path(td)
            submission = root / "submission"
            submission.mkdir(mode=0o755)
            submission.chmod(0o755)
            staged_solver = submission / "solver.py"
            shutil.copy2(solver_dir / "solver.py", staged_solver)
            staged_solver.chmod(0o444)
            logs = root / "logs"
            logs.mkdir()
            proc = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--cpus",
                    "1",
                    "--memory",
                    "512m",
                    "--pids-limit",
                    "256",
                    "--volume",
                    f"{submission.resolve()}:/app/methods/main:ro",
                    "--volume",
                    f"{logs.resolve()}:/logs/verifier",
                    self.image,
                    "/bin/bash",
                    "/tests/test.sh",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
            reward = json.loads((logs / "reward.json").read_text(encoding="utf-8"))
            debug = json.loads((logs / "grade_debug.json").read_text(encoding="utf-8"))
            return reward, debug

    def test_replays_are_identical_and_anchor_metadata_hold(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".reference-solver-", dir=TASK_ROOT) as td:
            reference_dir = Path(td)
            shutil.copy2(REFERENCE_SOLVER, reference_dir / "solver.py")
            floor_reward, floor_debug = self._run_verifier(FLOOR_SOLVER_DIR)
            ref_reward_1, ref_debug_1 = self._run_verifier(reference_dir)
            ref_reward_2, ref_debug_2 = self._run_verifier(reference_dir)

        for name, debug in (
            ("floor", floor_debug),
            ("reference replay 1", ref_debug_1),
            ("reference replay 2", ref_debug_2),
        ):
            self.assertTrue(debug["correctness"], f"{name}: {debug}")
            self.assertEqual(debug["trace_shape"], [20, 20, 120], name)
            self.assertEqual(debug["num_evals"], 48_000, name)
            self.assertTrue(debug["scorer"]["feasible"], name)

        self.assertEqual(ref_reward_1, ref_reward_2)
        self.assertEqual(ref_debug_1["scorer"], ref_debug_2["scorer"])
        self.assertLessEqual(abs(float(floor_reward["reward"])), STRICT_TOLERANCE)
        self.assertGreater(float(ref_reward_1["reward"]), 0.0)
        self.assertLess(float(ref_reward_1["reward"]), 0.95)
        reference_diagnostic = float(
            ref_debug_1["scorer"]["reference_score_diagnostic"]
        )
        self.assertGreater(reference_diagnostic, 0.0)
        self.assertLess(reference_diagnostic, 0.95)
        self.assertAlmostEqual(
            float(ref_reward_1["reward"]),
            reference_diagnostic,
            delta=REFERENCE_DIAGNOSTIC_TOLERANCE,
        )

        anchors = json.loads(ANCHORS.read_text(encoding="utf-8"))
        self.assertEqual(anchors["calibration_status"], "final_protocol_verified")
        self.assertEqual(anchors["calibration_protocol"], "harbor-bbo-fresh-child-v2")
        self.assertEqual(
            anchors["floor_sha256"], _sha256(FLOOR_SOLVER_DIR / "solver.py")
        )
        identity = f"{anchors['frontier']} {anchors['floor']}".lower()
        self.assertNotIn("portfolio", identity)
        self.assertNotIn("hindsight", identity)
        self.assertIn("sealed hidden run", anchors["frontier"].lower())
        self.assertEqual(len(anchors["frontier_combined"]), anchors["n_hidden"])

    def test_calibration_utility_regenerates_valid_bundle_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".calibration-output-", dir=TASK_ROOT) as td:
            output_dir = Path(td)
            proc = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--cpus",
                    "1",
                    "--memory",
                    "512m",
                    "--pids-limit",
                    "256",
                    "--volume",
                    f"{(FLOOR_SOLVER_DIR / 'solver.py').resolve()}:/calibration/floor_solver.py:ro",
                    "--volume",
                    f"{REFERENCE_SOLVER.resolve()}:/calibration/reference_optimizer.py:ro",
                    "--volume",
                    f"{output_dir.resolve()}:/calibration/output",
                    "--volume",
                    f"{HELDOUT_ROOT.resolve()}:/calibration/heldout:ro",
                    "--volume",
                    f"{(TESTS_ROOT / 'grade.py').resolve()}:/calibration/grade.py:ro",
                    "--volume",
                    f"{(TESTS_ROOT / 'child_optimizer.py').resolve()}:/calibration/child_optimizer.py:ro",
                    self.image,
                    "python3",
                    "/calibration/heldout/calibrate_anchors.py",
                    "--floor-solver",
                    "/calibration/floor_solver.py",
                    "--reference-solver",
                    "/calibration/reference_optimizer.py",
                    "--output",
                    "/calibration/output/frozen_anchors.json",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=600,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
            summary = json.loads(proc.stdout)
            regenerated = json.loads(
                (output_dir / "frozen_anchors.json").read_text(encoding="utf-8")
            )

        self.assertEqual(summary["floor_trace_shape"], [20, 20, 120])
        self.assertEqual(summary["reference_trace_shape"], [20, 20, 120])
        self.assertTrue(summary["all_traces_finite_nonincreasing"])
        frozen = json.loads(ANCHORS.read_text(encoding="utf-8"))
        trace_fields = {"floor_trace_median", "ref_trace_median"}
        # the frontier anchor comes from a sealed model run, not from calibration,
        # so compare only the keys the calibration utility itself produces
        shared = set(regenerated) & set(frozen) - trace_fields
        self.assertEqual(
            {key: regenerated[key] for key in shared},
            {key: frozen[key] for key in shared},
        )
        for field in trace_fields:
            traces = np.asarray(regenerated[field], dtype=float)
            self.assertEqual(traces.shape, (20, 120), field)
            self.assertTrue(np.isfinite(traces).all(), field)
            self.assertTrue(np.all(np.diff(traces, axis=1) <= 0.0), field)


if __name__ == "__main__":
    unittest.main()
