"""Final-protocol boundary checks for the noisy continuous BBO task."""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path
from statistics import mean


TASK_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = TASK_ROOT / "tests"
PUBLIC_SOLVER_DIR = TASK_ROOT / "environment" / "methods" / "main"
HELDOUT_ROOT = TESTS_ROOT / "heldout"
SCORE_DETAILS_METRIC = "best_latent_objective_at_final_query"
SCORE_DETAILS_AGGREGATION = (
    "median over seeds per instance; leaderboard reward uses "
    "mean(0.70*anytime_per_instance+0.30*final_per_instance)"
)


def _empty_score_details() -> dict:
    return {
        "metric": SCORE_DETAILS_METRIC,
        "direction": "lower",
        "aggregation": SCORE_DETAILS_AGGREGATION,
        "instances": [],
        "aggregate": {
            "raw_metric": 0.0,
            "floor": 0.0,
            "upper_bound": 0.0,
            "reward": 0.0,
        },
    }


def _load_grader_module():
    spec = importlib.util.spec_from_file_location(
        f"bbo_noisy_grade_test_{uuid.uuid4().hex}",
        TESTS_ROOT / "grade.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError("could not load noisy continuous grader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPublicContract(unittest.TestCase):
    def test_selfcheck_uses_only_the_final_six_argument_constructor(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bbo-noisy-public-") as td:
            environment = Path(td) / "environment"
            shutil.copytree(TASK_ROOT / "environment", environment)
            (environment / "methods" / "main" / "solver.py").write_text(
                """import numpy as np

class FinalConstructorOnly(type):
    def __call__(cls, *args, **kwargs):
        if "task_info" in kwargs:
            raise AssertionError("selfcheck supplied obsolete task_info")
        return super().__call__(*args, **kwargs)

class Optimizer(metaclass=FinalConstructorOnly):
    def __init__(self, dim, lower, upper, budget, seed, rng):
        self.dim = dim
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.rng = rng

    def ask(self, n):
        return self.rng.uniform(self.lower, self.upper, size=(n, self.dim))

    def tell(self, X, y, metadata=None):
        pass
""",
                encoding="utf-8",
            )
            proc = subprocess.run(
                ["python3", str(environment / "selfcheck.py")],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                check=False,
            )

        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        self.assertIn("visible oracle-normalized score", proc.stdout)

    def test_instructions_require_a_self_contained_solver_file(self) -> None:
        instruction = (TASK_ROOT / "instruction.md").read_text(encoding="utf-8")
        self.assertNotIn("modules may live beside it", instruction)
        self.assertIn(
            "entire submitted optimizer must be self-contained in "
            "`/app/methods/main/solver.py`",
            instruction,
        )


class TestFailureScoreDetails(unittest.TestCase):
    def test_missing_sealed_assets_emit_valid_empty_schema(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bbo-noisy-failure-") as td:
            root = Path(td)
            heldout = root / "heldout"
            logs = root / "logs"
            heldout.mkdir()

            grader = _load_grader_module()
            grader.HELDOUT = heldout
            grader.REWARD_DIR = logs
            grader._assert_submission_readable_by_runner = lambda: None
            sealed_names = {
                "HIDDEN_DATA_PATH": "hidden_data.json",
                "ANCHORS_PATH": "frozen_anchors.json",
                "HARNESS_PATH": "bbo_harness.py",
                "SCORER_PATH": "source_evaluate.py",
                "ORACLE_PATH": "oracle_values.json",
                "ORACLE_PROVENANCE_PATH": "oracle_provenance.json",
                "ORACLE_SCORING_PATH": "oracle_scoring.py",
            }
            for attribute, filename in sealed_names.items():
                setattr(grader, attribute, heldout / filename)

            grader.main()

            self.assertEqual(
                json.loads((logs / "score_details.json").read_text(encoding="utf-8")),
                _empty_score_details(),
            )
            reward_doc = json.loads((logs / "reward.json").read_text(encoding="utf-8"))
            self.assertEqual(reward_doc, {"reward": 0.0})
            self.assertEqual((logs / "reward.txt").read_text(encoding="utf-8"), "0.0\n")
            debug = json.loads((logs / "grade_debug.json").read_text(encoding="utf-8"))
            self.assertFalse(debug["correctness"])
            self.assertTrue(
                any("grader failed:" in error for error in debug["errors"]),
                debug,
            )


class TestScoreDetailsBuilder(unittest.TestCase):
    def test_raw_metrics_are_seed_medians_at_the_final_query(self) -> None:
        grader = _load_grader_module()
        traces = [
            [[9.0, 8.0, 7.0, 4.0], [13.0, 12.0, 11.0, 8.0]],
            [[24.0, 23.0, 22.0, 20.0], [14.0, 13.0, 12.0, 10.0]],
            [[34.0, 33.0, 32.0, 30.0], [44.0, 43.0, 42.0, 40.0]],
        ]
        anchors = {
            "n_hidden": 3,
            "n_seeds": 2,
            "budget": 4,
            "floor_trace_median": [[0.0, 0.0, 0.0, 50.0]] * 3,
        }
        reward_doc = {
            "reward_per_inst": [0.1, 0.2, 0.3],
            "anytime_per_inst": [0.2, 0.3, 0.4],
            "final_per_inst": [0.0, 0.1, 0.2],
            "kpi": 56.0 / 3.0,
        }

        details = grader._build_score_details(
            traces,
            anchors,
            {"per_instance_objective": [0.0, 1.0, 2.0]},
            reward_doc,
            0.25,
        )

        self.assertEqual(
            [instance["raw_metric"] for instance in details["instances"]],
            [6.0, 15.0, 35.0],
        )
        self.assertEqual(details["aggregate"]["raw_metric"], 56.0 / 3.0)

    def test_explicit_trace_collector_emits_empty_details(self) -> None:
        grader = _load_grader_module()
        traces = [
            [[5.0, 4.0, 3.0], [6.0, 5.0, 2.0]],
            [[9.0, 8.0, 7.0], [10.0, 8.0, 6.0]],
        ]
        anchors = {"n_hidden": 2, "n_seeds": 2, "budget": 3}
        reward_doc = {"feasible": True, "score": 0.0, "traces": traces}

        details = grader._build_score_details(traces, anchors, {}, reward_doc, 0.0)

        self.assertEqual(details, grader._empty_score_details())

    def test_malformed_formal_scorer_document_still_fails_closed(self) -> None:
        grader = _load_grader_module()
        traces = [[[5.0, 4.0, 3.0]]]
        anchors = {"n_hidden": 1, "n_seeds": 1, "budget": 3}
        malformed = {
            "feasible": True,
            "score": 0.0,
            "traces": traces,
            "metric": "source_score",
        }

        with self.assertRaises(KeyError):
            grader._build_score_details(traces, anchors, {}, malformed, 0.0)


class TestSubmissionPreflight(unittest.TestCase):
    def test_missing_and_symlinked_solver_are_rejected_before_execution(self) -> None:
        grader = _load_grader_module()
        with tempfile.TemporaryDirectory(prefix="bbo-noisy-preflight-") as td:
            submission = Path(td) / "submission"
            submission.mkdir()
            grader.SUBMISSION_DIR = submission

            with self.assertRaisesRegex(
                grader.SubmissionError,
                "solver.py is missing",
            ):
                grader._assert_submission_readable_by_runner()

            target = Path(td) / "target.py"
            target.write_text("raise AssertionError('must not execute')\n", encoding="utf-8")
            (submission / "solver.py").symlink_to(target)
            with self.assertRaisesRegex(
                grader.SubmissionError,
                "regular non-symlink file",
            ):
                grader._assert_submission_readable_by_runner()

    def test_cpu_rlimit_cannot_preempt_the_one_cpu_global_timeout(self) -> None:
        grader = _load_grader_module()
        self.assertEqual(grader.VERIFIER_CPUS, 1)
        self.assertGreater(
            float(grader.RLIMIT_CPU_SECONDS),
            float(grader.HARD_CAP_SEC),
        )


class TestRunnerBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image = f"bbo-noisy-boundary:{uuid.uuid4().hex}"
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

    def _run_verifier(
        self, solver_dir: Path
    ) -> tuple[subprocess.CompletedProcess[str], dict, dict, dict]:
        with tempfile.TemporaryDirectory(prefix=".boundary-logs-", dir=TASK_ROOT) as td:
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
            self.assertTrue((logs / "reward.json").is_file(), proc.stderr or proc.stdout)
            self.assertTrue((logs / "grade_debug.json").is_file(), proc.stderr or proc.stdout)
            self.assertTrue((logs / "score_details.json").is_file(), proc.stderr or proc.stdout)
            reward = json.loads((logs / "reward.json").read_text(encoding="utf-8"))
            debug = json.loads((logs / "grade_debug.json").read_text(encoding="utf-8"))
            details = json.loads((logs / "score_details.json").read_text(encoding="utf-8"))
            return proc, reward, debug, details

    def _assert_score_details(
        self,
        details: dict,
        reward: dict,
        debug: dict,
    ) -> None:
        anchors = json.loads((HELDOUT_ROOT / "frozen_anchors.json").read_text(encoding="utf-8"))
        oracle = json.loads((HELDOUT_ROOT / "oracle_values.json").read_text(encoding="utf-8"))
        hidden = json.loads((HELDOUT_ROOT / "hidden_data.json").read_text(encoding="utf-8"))
        expected_floor = [float(trace[-1]) for trace in anchors["floor_trace_median"]]
        expected_upper = [float(value) for value in oracle["per_instance_objective"]]

        self.assertEqual(
            set(details),
            {"metric", "direction", "aggregation", "instances", "aggregate"},
        )
        self.assertEqual(details["metric"], SCORE_DETAILS_METRIC)
        self.assertEqual(details["direction"], "lower")
        self.assertEqual(details["aggregation"], SCORE_DETAILS_AGGREGATION)
        self.assertEqual(len(details["instances"]), 20)
        expected_instance_keys = {
            "id",
            "raw_metric",
            "floor",
            "upper_bound",
            "score",
            "anytime_score",
            "final_score",
        }
        hidden_parameter_keys = set().union(
            *(set(instance) for instance in hidden["instances"])
        )
        self.assertTrue(expected_instance_keys.isdisjoint(hidden_parameter_keys))
        for index, instance in enumerate(details["instances"]):
            self.assertEqual(set(instance), expected_instance_keys)
            self.assertEqual(instance["id"], f"instance_{index:03d}")
        self.assertNotIn("hidden_", json.dumps(details, sort_keys=True))

        runtime_raw = [float(instance["raw_metric"]) for instance in details["instances"]]
        self.assertTrue(all(math.isfinite(value) for value in runtime_raw))
        self.assertEqual(
            [instance["floor"] for instance in details["instances"]],
            expected_floor,
        )
        self.assertEqual(
            [instance["upper_bound"] for instance in details["instances"]],
            expected_upper,
        )
        self.assertEqual(
            [instance["score"] for instance in details["instances"]],
            debug["scorer"]["reward_per_inst"],
        )
        self.assertEqual(
            [instance["anytime_score"] for instance in details["instances"]],
            debug["scorer"]["anytime_per_inst"],
        )
        self.assertEqual(
            [instance["final_score"] for instance in details["instances"]],
            debug["scorer"]["final_per_inst"],
        )

        self.assertEqual(
            details["aggregate"],
            {
                "raw_metric": float(mean(runtime_raw)),
                "floor": float(mean(expected_floor)),
                "upper_bound": float(mean(expected_upper)),
                "reward": reward["reward"],
            },
        )
        self.assertEqual(details["aggregate"]["raw_metric"], debug["scorer"]["kpi"])
        self.assertEqual(details["aggregate"]["reward"], debug["reward"])
        self.assertEqual(details["aggregate"]["reward"], debug["scorer"]["score"])

    def test_public_floor_runs_through_final_child_protocol(self) -> None:
        self.assertTrue((TESTS_ROOT / "child_optimizer.py").is_file())
        proc, reward, debug, details = self._run_verifier(PUBLIC_SOLVER_DIR)
        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        self.assertTrue(debug["correctness"], debug)
        self.assertEqual(debug["trace_shape"], [20, 20, 120])
        self.assertEqual(debug["num_evals"], 48_000)
        self.assertLessEqual(abs(float(reward["reward"])), 1e-8, reward)
        self._assert_score_details(details, reward, debug)

    def test_fixed_reference_has_runtime_consistent_score_details(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".reference-solver-", dir=TASK_ROOT) as td:
            solver_dir = Path(td)
            shutil.copy2(TASK_ROOT / "solution" / "reference_optimizer.py", solver_dir / "solver.py")
            proc, reward, debug, details = self._run_verifier(solver_dir)

        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        self.assertTrue(debug["correctness"], debug)
        self.assertGreater(float(reward["reward"]), 0.0)
        self.assertLess(float(reward["reward"]), 0.95)
        self.assertEqual(float(reward["reward"]), float(debug["scorer"]["score"]))
        self._assert_score_details(details, reward, debug)

    def test_malformed_optimizer_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".boundary-solver-", dir=TASK_ROOT) as td:
            solver_dir = Path(td)
            (solver_dir / "solver.py").write_text(
                """import numpy as np

class Optimizer:
    def __init__(self, dim, lower, upper, budget, seed, rng):
        self.dim = dim

    def ask(self, n):
        return np.zeros((n, self.dim + 1))

    def tell(self, X, y, metadata=None):
        pass
""",
                encoding="utf-8",
            )
            proc, reward, debug, details = self._run_verifier(solver_dir)

        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        self.assertEqual(float(reward["reward"]), 0.0, reward)
        self.assertFalse(debug["correctness"], debug)
        self.assertTrue(debug["errors"], debug)
        self.assertEqual(details, _empty_score_details())

    def test_protocol_output_limit_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".boundary-stdout-", dir=TASK_ROOT) as td:
            solver_dir = Path(td)
            (solver_dir / "solver.py").write_text(
                """import os
import numpy as np

class Optimizer:
    batch = 120

    def __init__(self, dim, lower, upper, budget, seed, rng):
        self.dim = dim

    def ask(self, n):
        payload = b"x" * (1024 * 1024 + 1)
        while payload:
            payload = payload[os.write(1, payload):]
        return np.zeros((n, self.dim))

    def tell(self, X, y, metadata=None):
        pass
""",
                encoding="utf-8",
            )
            proc, reward, debug, details = self._run_verifier(solver_dir)

        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        self.assertEqual(reward, {"reward": 0.0})
        self.assertFalse(debug["correctness"], debug)
        self.assertTrue(
            any("protocol output exceeded the byte limit" in error for error in debug["errors"]),
            debug,
        )
        self.assertEqual(details, _empty_score_details())

    def test_real_chain_probe_and_monkeypatch_cannot_reach_verifier(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".boundary-probe-", dir=TASK_ROOT) as td:
            solver_dir = Path(td)
            (solver_dir / "solver.py").write_text(
                """import builtins
import glob
import os
from pathlib import Path
import numpy as np

BREACHES = []
for candidate in ("/tests/heldout/hidden_data.json", "/tests/grade.py", "/proc/self/status"):
    try:
        with builtins.open(candidate, "rb") as handle:
            handle.read(1)
    except OSError:
        pass
    else:
        BREACHES.append("read:" + candidate)
try:
    if os.listdir("/tests/heldout"):
        BREACHES.append("listdir")
except OSError:
    pass
if glob.glob("/tests/heldout/*"):
    BREACHES.append("glob")
try:
    Path("/logs/verifier/reward.json").write_text('{"reward": 1.0}', encoding="utf-8")
except OSError:
    pass
else:
    BREACHES.append("reward-write")

class Optimizer:
    batch = 120

    def __init__(self, dim, lower, upper, budget, seed, rng):
        if BREACHES:
            raise RuntimeError("isolation breach: " + ",".join(BREACHES))
        self.dim = dim

    def ask(self, n):
        builtins.open = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("patched"))
        Path.open = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("patched"))
        return np.zeros((n, self.dim + 1))

    def tell(self, X, y, metadata=None):
        pass
""",
                encoding="utf-8",
            )
            proc, reward, debug, details = self._run_verifier(solver_dir)

        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        self.assertEqual(reward, {"reward": 0.0})
        self.assertFalse(debug["correctness"], debug)
        self.assertFalse(
            any("isolation breach" in error or error.startswith("grader failed:") for error in debug["errors"]),
            debug,
        )
        self.assertTrue(any("shape" in error for error in debug["errors"]), debug)
        self.assertEqual(details, _empty_score_details())

    def test_detached_process_abstract_socket_and_sysv_ipc_are_purged(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".boundary-persistence-", dir=TASK_ROOT) as td:
            solver_dir = Path(td)
            (solver_dir / "solver.py").write_text(
                """import ctypes
import os
import socket
import time
import numpy as np

class Optimizer:
    batch = 120

    def __init__(self, dim, lower, upper, budget, seed, rng):
        libc = ctypes.CDLL(None, use_errno=True)
        self.shmid = libc.shmget(0, 4096, 0o1600)
        if self.shmid < 0:
            raise OSError(ctypes.get_errno(), "shmget failed")
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            os.setsid()
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(chr(0) + "bbo_noisy_attack_" + str(os.getpid()))
            os.write(write_fd, b"1")
            os.close(write_fd)
            while True:
                time.sleep(10)
        os.close(write_fd)
        if os.read(read_fd, 1) != b"1":
            raise RuntimeError("detached child did not initialize")
        os.close(read_fd)
        self.dim = dim

    def ask(self, n):
        return np.zeros((n, self.dim + 1))

    def tell(self, X, y, metadata=None):
        pass
""",
                encoding="utf-8",
            )
            proc, reward, debug, details = self._run_verifier(solver_dir)

        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        self.assertEqual(reward, {"reward": 0.0})
        self.assertFalse(debug["correctness"], debug)
        self.assertGreaterEqual(debug["isolation"]["processes_killed"], 1, debug)
        self.assertGreaterEqual(debug["isolation"]["sysv_ipc_removed"], 1, debug)
        self.assertFalse(
            any(error.startswith("grader failed:") for error in debug["errors"]),
            debug,
        )
        self.assertEqual(details, _empty_score_details())

    def test_scratch_entry_limit_fails_closed_and_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".boundary-scratch-", dir=TASK_ROOT) as td:
            solver_dir = Path(td)
            (solver_dir / "solver.py").write_text(
                """import numpy as np

class Optimizer:
    batch = 120

    def __init__(self, dim, lower, upper, budget, seed, rng):
        self.dim = dim
        for index in range(300):
            with open(f"junk_{index:03d}", "wb"):
                pass

    def ask(self, n):
        return np.zeros((n, self.dim + 1))

    def tell(self, X, y, metadata=None):
        pass
""",
                encoding="utf-8",
            )
            proc, reward, debug, details = self._run_verifier(solver_dir)

        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        self.assertEqual(reward, {"reward": 0.0})
        self.assertFalse(debug["correctness"], debug)
        self.assertTrue(
            any("scratch entry limit exceeded" in error for error in debug["errors"]),
            debug,
        )
        self.assertEqual(details, _empty_score_details())

    def test_address_space_pressure_fails_child_without_killing_verifier(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".boundary-memory-", dir=TASK_ROOT) as td:
            solver_dir = Path(td)
            (solver_dir / "solver.py").write_text(
                """import numpy as np

class Optimizer:
    batch = 120

    def __init__(self, dim, lower, upper, budget, seed, rng):
        self.payload = bytearray(512 * 1024 * 1024)
        self.dim = dim

    def ask(self, n):
        return np.zeros((n, self.dim))

    def tell(self, X, y, metadata=None):
        pass
""",
                encoding="utf-8",
            )
            proc, reward, debug, details = self._run_verifier(solver_dir)

        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        self.assertEqual(reward, {"reward": 0.0})
        self.assertFalse(debug["correctness"], debug)
        self.assertTrue(any("MemoryError" in error for error in debug["errors"]), debug)
        self.assertFalse(
            any(error.startswith("grader failed:") for error in debug["errors"]),
            debug,
        )
        self.assertEqual(details, _empty_score_details())

    def test_shortened_scratch_copy_timeout_scores_partial_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".boundary-timeout-", dir=TASK_ROOT) as td:
            root = Path(td)
            submission = root / "submission"
            submission.mkdir(mode=0o755)
            submission.chmod(0o755)
            solver = submission / "solver.py"
            solver.write_text(
                """import time
import numpy as np

class Optimizer:
    batch = 1

    def __init__(self, dim, lower, upper, budget, seed, rng):
        self.dim = dim
        self.calls = 0

    def ask(self, n):
        self.calls += 1
        if self.calls > 1:
            time.sleep(1.0)
        return np.zeros((n, self.dim))

    def tell(self, X, y, metadata=None):
        pass
""",
                encoding="utf-8",
            )
            solver.chmod(0o444)
            logs = root / "logs"
            logs.mkdir()
            grade_source = (TESTS_ROOT / "grade.py").read_text(encoding="utf-8")
            replacements = {
                "TIME_BUDGET_SEC = 120.0": "TIME_BUDGET_SEC = 0.5",
                "GRACE_SEC = 15.0": "GRACE_SEC = 0.2",
                "HARD_CAP_SEC = TIME_BUDGET_SEC + 30.0": "HARD_CAP_SEC = 3.0",
            }
            for original, replacement in replacements.items():
                self.assertEqual(grade_source.count(original), 1, original)
                grade_source = grade_source.replace(original, replacement)
            scratch_grade = root / "grade.py"
            scratch_grade.write_text(grade_source, encoding="utf-8")
            scratch_grade.chmod(0o444)
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
                    "--volume",
                    f"{scratch_grade.resolve()}:/tests/grade.py:ro",
                    self.image,
                    "/bin/bash",
                    "/tests/test.sh",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
            reward = json.loads((logs / "reward.json").read_text(encoding="utf-8"))
            debug = json.loads((logs / "grade_debug.json").read_text(encoding="utf-8"))
            details = json.loads((logs / "score_details.json").read_text(encoding="utf-8"))

        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        self.assertGreaterEqual(float(reward["reward"]), 0.0, reward)
        self.assertLessEqual(float(reward["reward"]), 1.0, reward)
        self.assertTrue(debug["correctness"], debug)
        self.assertEqual(debug["num_evals"], 1, debug)
        self.assertTrue(debug["runtime"]["truncated"], debug)
        self.assertEqual(debug["runtime"]["partial_runs"], 1, debug)
        self.assertEqual(debug["runtime"]["floor_filled_runs"], 399, debug)
        self.assertTrue(
            any(error.startswith("submission timed out:") for error in debug["errors"]),
            debug,
        )
        self.assertFalse(
            any(error.startswith("grader failed:") for error in debug["errors"]),
            debug,
        )
        self.assertEqual(len(details["instances"]), 20)

    def test_entrypoint_crash_cannot_preserve_planted_reward(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".boundary-fallback-", dir=TASK_ROOT) as td:
            root = Path(td)
            submission = root / "submission"
            submission.mkdir(mode=0o755)
            solver = submission / "solver.py"
            shutil.copy2(PUBLIC_SOLVER_DIR / "solver.py", solver)
            solver.chmod(0o444)
            logs = root / "logs"
            logs.mkdir()
            malicious_grade = root / "grade.py"
            malicious_grade.write_text(
                """import json
from pathlib import Path

root = Path("/logs/verifier")
(root / "reward.txt").write_text("1.0\\n", encoding="utf-8")
(root / "reward.json").write_text(json.dumps({"reward": 1.0}), encoding="utf-8")
(root / "score_details.json").write_text("{}", encoding="utf-8")
(root / "grade_debug.json").write_text(json.dumps({
    "reward": 1.0,
    "correctness": True,
    "errors": ["submission failed: planted"],
}), encoding="utf-8")
raise SystemExit(9)
""",
                encoding="utf-8",
            )
            malicious_grade.chmod(0o444)
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
                    "--volume",
                    f"{malicious_grade.resolve()}:/tests/grade.py:ro",
                    self.image,
                    "/bin/bash",
                    "/tests/test.sh",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
            reward = json.loads((logs / "reward.json").read_text(encoding="utf-8"))
            debug = json.loads((logs / "grade_debug.json").read_text(encoding="utf-8"))
            details = json.loads((logs / "score_details.json").read_text(encoding="utf-8"))

        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        self.assertEqual(reward, {"reward": 0.0})
        self.assertFalse(debug["correctness"], debug)
        self.assertTrue(debug["errors"][0].startswith("grader failed:"), debug)
        self.assertNotIn("planted", json.dumps(debug, sort_keys=True))
        self.assertEqual(details, _empty_score_details())

    def test_entrypoint_crash_preserves_only_trusted_grader_marker(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".boundary-marker-", dir=TASK_ROOT) as td:
            root = Path(td)
            logs = root / "logs"
            logs.mkdir()
            malicious_grade = root / "grade.py"
            malicious_grade.write_text(
                """import json
from pathlib import Path

root = Path("/logs/verifier")
(root / "reward.json").write_text(json.dumps({"reward": 1.0}), encoding="utf-8")
(root / "grade_debug.json").write_text(json.dumps({
    "reward": 1.0,
    "correctness": False,
    "errors": ["grader failed: sentinel"],
}), encoding="utf-8")
raise SystemExit(7)
""",
                encoding="utf-8",
            )
            malicious_grade.chmod(0o444)
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
                    f"{PUBLIC_SOLVER_DIR.resolve()}:/app/methods/main:ro",
                    "--volume",
                    f"{logs.resolve()}:/logs/verifier",
                    "--volume",
                    f"{malicious_grade.resolve()}:/tests/grade.py:ro",
                    self.image,
                    "/bin/bash",
                    "/tests/test.sh",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
            reward = json.loads((logs / "reward.json").read_text(encoding="utf-8"))
            debug = json.loads((logs / "grade_debug.json").read_text(encoding="utf-8"))
            details = json.loads((logs / "score_details.json").read_text(encoding="utf-8"))

        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        self.assertEqual(reward, {"reward": 0.0})
        self.assertEqual(debug["errors"], ["grader failed: sentinel"])
        self.assertEqual(details, _empty_score_details())

    def test_runner_cannot_read_or_write_trusted_assets(self) -> None:
        harmless_probe = (
            "import os,sys; checks=["
            "not os.access('/tests/heldout/hidden_data.json', os.R_OK),"
            "not os.access('/tests/heldout/source_evaluate.py', os.W_OK)"
            "]; sys.exit(0 if all(checks) else 1)"
        )
        proc = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--user",
                "20000:20000",
                self.image,
                "python3",
                "-c",
                harmless_probe,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)


if __name__ == "__main__":
    unittest.main()
