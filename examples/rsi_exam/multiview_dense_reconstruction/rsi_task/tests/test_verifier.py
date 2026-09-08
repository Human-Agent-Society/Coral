#!/usr/bin/env python3
"""Focused active-verifier regressions for dense-cloud v3."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fail_closed as fc  # noqa: E402
import grade  # noqa: E402


TASK = Path(__file__).resolve().parents[1]


def fake_scenes() -> dict[str, dict[str, object]]:
    return {"sealed_case_001": {"view_ids": ["view_001", "view_002"]}}


def valid_payload() -> bytes:
    cloud = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]
    return json.dumps({"sealed_case_001": cloud}).encode()


class DenseVerifierTest(unittest.TestCase):
    def test_anchor_landings(self) -> None:
        assert grade.scene_reward(0.1, 0.1, 0.7, 1.0) == 0.0
        assert grade.scene_reward(0.7, 0.1, 0.7, 1.0) == 0.6
        assert grade.scene_reward(1.0, 0.1, 0.7, 1.0) == 1.0

    def test_perfect_case_metric(self) -> None:
        cloud = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
        metric, details = grade._case_metric(cloud, cloud)
        self.assertEqual(metric, 1.0)
        self.assertEqual([row["fscore"] for row in details], [1.0, 1.0, 1.0])

    def test_valid_prediction_json(self) -> None:
        result = grade.validate_predictions(valid_payload(), fake_scenes())
        self.assertEqual(result["sealed_case_001"].shape, (4, 3))

    def test_missing_case_rejected(self) -> None:
        with self.assertRaises(grade.SubmissionError):
            grade.validate_predictions(b"{}", fake_scenes())

    def test_nonfinite_rejected(self) -> None:
        payload = valid_payload().replace(b"0.1", b"NaN", 1)
        with self.assertRaises(grade.SubmissionError):
            grade.validate_predictions(payload, fake_scenes())

    def test_duplicate_only_cloud_rejected(self) -> None:
        decoded = json.loads(valid_payload())
        decoded["sealed_case_001"] = [[0.0, 0.0, 0.0]] * 4
        with self.assertRaises(grade.SubmissionError):
            grade.validate_predictions(json.dumps(decoded).encode(), fake_scenes())

    def test_active_anchors_are_ordered_and_complete(self) -> None:
        anchors = json.loads((TASK / "tests" / "anchors.json").read_text())
        self.assertEqual(anchors["calibration_status"], "active")
        self.assertEqual(anchors["schema_version"], "dense-calibration-v7")
        self.assertEqual(anchors["mapping_order"], "mean_then_map")
        self.assertNotIn("per_case", anchors)
        self.assertEqual(len(anchors["per_case_diagnostic"]), 9)
        for row in anchors["per_case_diagnostic"].values():
            self.assertLess(row["baseline_fscore"], row["reference_fscore"])
            self.assertLess(row["reference_fscore"], anchors["upper_fscore"])

    def test_scoring_anchors_are_the_per_case_means(self) -> None:
        anchors = json.loads((TASK / "tests" / "anchors.json").read_text())
        rows = anchors["per_case_diagnostic"].values()
        for name in ("baseline", "reference"):
            self.assertAlmostEqual(
                anchors[f"{name}_fscore"],
                float(np.mean([row[f"{name}_fscore"] for row in rows])),
                places=12,
                msg=name,
            )
        self.assertLess(anchors["baseline_fscore"], anchors["reference_fscore"])
        self.assertLess(anchors["reference_fscore"], anchors["upper_fscore"])

    def test_single_mapping_gives_every_case_one_slope(self) -> None:
        """The v3 defect: with upper fixed at 1.0, per-case mapping made the upper-segment
        slope 0.7/(1 - reference_i), steepest on the easiest cases. Under mean_then_map a
        fixed raw-F-score gain must move the reward by the same amount wherever it lands.
        """
        anchors = json.loads((TASK / "tests" / "anchors.json").read_text())
        baseline = anchors["baseline_fscore"]
        reference = anchors["reference_fscore"]
        upper = anchors["upper_fscore"]
        rows = list(anchors["per_case_diagnostic"].values())
        base = [row["reference_fscore"] for row in rows]
        nudge = 0.01
        deltas = []
        for index in range(len(base)):
            bumped = list(base)
            bumped[index] += nudge
            deltas.append(
                grade.scene_reward(float(np.mean(bumped)), baseline, reference, upper)
                - grade.scene_reward(float(np.mean(base)), baseline, reference, upper)
            )
        self.assertAlmostEqual(max(deltas), min(deltas), places=12)

    def test_agent_image_copy_surface(self) -> None:
        path = TASK / "environment" / "Dockerfile"
        if not path.is_file():
            self.skipTest("agent Dockerfile is outside verifier image")
        dockerfile = path.read_text()
        for forbidden in ("tests", "solution", "_dev", "task.toml", "anchors.json"):
            self.assertNotIn(f"COPY {forbidden}", dockerfile)

    def test_runtime_hardening_constants(self) -> None:
        source = (TASK / "tests" / "grade.py").read_text()
        for required in (
            'VERIFIER_THREADS = "1"', "TIME_BUDGET_SEC = 300.0",
            "MAX_CHILD_ADDRESS_SPACE_BYTES = 384 * 1024 * 1024",
            "libc.prctl(38, 1, 0, 0, 0)", "_kill_untrusted_uid_processes",
        ):
            self.assertIn(required, source)
        self.assertNotIn("os.environ.get", source)

    def _with_submission(self, modifier) -> None:
        old_root, old_dir = grade.SUBMISSION_ROOT, grade.SUBMISSION_DIR
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "methods"
            submission = root / "main"
            submission.mkdir(parents=True)
            solver = submission / "solver.py"
            solver.write_text("def predict(export_dir):\n    return {}\n")
            modifier(solver, submission)
            workspace = Path(temporary) / "workspace"
            workspace.mkdir()
            grade.SUBMISSION_ROOT, grade.SUBMISSION_DIR = root, submission
            try:
                with self.assertRaises(grade.SubmissionError):
                    grade.stage_submission(workspace)
            finally:
                grade.SUBMISSION_ROOT, grade.SUBMISSION_DIR = old_root, old_dir

    def test_symlink_source_rejected(self) -> None:
        def modifier(solver: Path, _: Path) -> None:
            solver.unlink()
            solver.symlink_to("missing.py")
        self._with_submission(modifier)

    def test_hardlink_source_rejected(self) -> None:
        self._with_submission(lambda solver, directory: os.link(solver, directory / "twin.py"))

    def test_non_source_artifact_rejected(self) -> None:
        self._with_submission(lambda _solver, directory: (directory / "payload.bin").write_bytes(b"x"))

    def test_fail_closed_roundtrip(self) -> None:
        if os.geteuid() != 0:
            self.skipTest("root ownership contract is image-only")
        old = fc.REWARD_DIR
        with tempfile.TemporaryDirectory() as temporary:
            fc.REWARD_DIR = Path(temporary)
            try:
                fc.write_failure_outputs("grader failed: fixture")
                details = fc.validate_outputs(published=True)
                self.assertEqual(details["reward"], 0.0)
                self.assertTrue(details["error"].startswith("grader failed:"))
            finally:
                fc.REWARD_DIR = old

    def test_split_seeds_are_disjoint(self) -> None:
        path = TASK / "_dev" / "ledger" / "v3_design.json"
        if not path.is_file():
            self.skipTest("author ledger is outside verifier image")
        design = json.loads(path.read_text())
        visible = {row["seed"] for row in design["data"]["visible"]}
        heldout = {row["seed"] for row in design["data"]["heldout"]}
        self.assertTrue(visible.isdisjoint(heldout))


if __name__ == "__main__":
    unittest.main()
