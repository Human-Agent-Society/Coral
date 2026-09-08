from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import fail_closed
import grade
from grade import GradeError, HiddenCase, Prediction
from secure_session import SessionError, run_case, stage_submission, tree_sha256


def synthetic_arrays(q_count: int = 2, frame_count: int = 8):
    query_points = np.zeros((q_count, 3), np.float32)
    query_points[:, 1:] = 32
    tracks = np.empty((q_count, frame_count, 10, 2), np.float16)
    occlusion = np.zeros((q_count, frame_count, 10), np.float16)
    expected = np.zeros_like(occlusion)
    model = np.empty((q_count, 10), np.uint8)
    stage = np.empty((q_count, 10), np.uint8)
    for query in range(q_count):
        for candidate in range(10):
            tracks[query, :, candidate, 0] = candidate
            tracks[query, :, candidate, 1] = query
            model[query, candidate] = candidate // 5
            stage[query, candidate] = candidate % 5
    public = {
        "query_points": query_points,
        "candidate_tracks": tracks,
        "occlusion_logits": occlusion,
        "expected_dist_logits": expected,
        "candidate_model_id": model,
        "candidate_stage": stage,
    }
    labels = {
        "gt_tracks": tracks[:, :, 9].astype(np.float32),
        "gt_occluded": np.zeros((q_count, frame_count), bool),
    }
    return public, labels


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bundle(root: Path, *, extra_public: bool = False) -> str:
    public, labels = synthetic_arrays()
    if extra_public:
        public["gt_tracks"] = labels["gt_tracks"]
    case_id = "0123456789abcdefabcd"
    (root / "inputs").mkdir(parents=True)
    (root / "labels").mkdir()
    input_path = root / "inputs" / f"case_{case_id}.npz"
    label_path = root / "labels" / f"case_{case_id}.npz"
    np.savez_compressed(input_path, **public)
    np.savez_compressed(label_path, **labels)
    manifest = {
        "schema_version": grade.SCHEMA_VERSION,
        "input_schema_name": grade.INPUT_SCHEMA_NAME,
        "partition": "primary_sealed",
        "case_count": 1,
        "input_fields": list(grade.PUBLIC_FIELDS),
        "label_fields": list(grade.LABEL_FIELDS),
        "candidate_permutation": "synthetic test",
        "cases": [
            {
                "case_id": case_id,
                "input_file": f"inputs/{input_path.name}",
                "label_file": f"labels/{label_path.name}",
                "input_sha256": sha256(input_path),
                "label_sha256": sha256(label_path),
                "input_schema": grade._array_schema(public),
                "label_schema": grade._array_schema(labels),
            }
        ],
    }
    manifest_path = root / "index.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return sha256(manifest_path)


class MetricAndRoutingTest(unittest.TestCase):
    def test_official_average_jaccard_perfect(self) -> None:
        public, labels = synthetic_arrays()
        case = HiddenCase("x", public, labels["gt_tracks"], labels["gt_occluded"])
        states = np.full(labels["gt_occluded"].shape, 9, np.uint8)
        prediction = Prediction(
            states, grade.reconstruct_tracks(public, states), labels["gt_occluded"].copy()
        )
        self.assertEqual(grade.average_jaccard(case, prediction), 1.0)

    def test_reward_mapping(self) -> None:
        self.assertEqual(grade.map_reward(0.2, 0.2, 1.0), 0.0)
        self.assertAlmostEqual(grade.map_reward(0.4, 0.2, 1.0), 0.25)
        self.assertAlmostEqual(grade.map_reward(0.6, 0.2, 1.0), 0.5)
        self.assertAlmostEqual(grade.map_reward(0.8, 0.2, 1.0), 0.75)
        self.assertEqual(grade.map_reward(1.0, 0.2, 1.0), 1.0)

    @unittest.skipUnless(os.geteuid() == 0, "root-only anchor-file test")
    def test_configured_anchor_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            anchor_file = Path(temporary) / "anchors.json"
            anchor_file.write_text(
                json.dumps({"BASELINE": 0.2, "UPPER_BOUND": 1.0})
            )
            anchor_file.chmod(0o400)
            with mock.patch.object(grade, "ANCHOR_FILE", anchor_file):
                self.assertEqual(
                    grade.read_configured_anchors(),
                    {
                        "baseline": 0.2,
                        "upper_bound": 1.0,
                    },
                )

    def test_integer_tokens_reconstruct_semantically(self) -> None:
        public, _ = synthetic_arrays()
        states = np.full((2, 8), 9, np.int64)
        result = grade.validate_prediction(
            (states, np.zeros((2, 8), bool)), public, "test"
        )
        self.assertEqual(result.state_token.dtype, np.uint8)
        self.assertTrue(np.all(result.tracks[..., 0] == 9))
        for bad in (
            states.astype(np.float32),
            np.full((2, 8), -1, np.int16),
            np.full((2, 8), 10, np.int16),
            np.zeros((2, 8), bool),
        ):
            with self.assertRaises(GradeError):
                grade.validate_prediction(
                    (bad, np.zeros((2, 8), bool)), public, "test"
                )

    def test_exactly_four_switches_pass_and_five_fail(self) -> None:
        public, _ = synthetic_arrays(1, 8)
        four = np.asarray([[9, 0, 1, 2, 3, 4, 4, 4]], np.int16)
        grade.validate_prediction((four, np.zeros((1, 8), bool)), public, "four")
        five = np.asarray([[9, 0, 1, 2, 3, 4, 5, 5]], np.int16)
        with self.assertRaisesRegex(GradeError, "exceeds"):
            grade.validate_prediction((five, np.ones((1, 8), bool)), public, "five")

    def test_prequery_changes_and_query_boundary_do_not_count(self) -> None:
        public, _ = synthetic_arrays(1, 8)
        public["query_points"][0, 0] = 4
        states = np.asarray([[0, 1, 2, 3, 4, 9, 9, 9]], np.int16)
        grade.validate_prediction((states, np.zeros((1, 8), bool)), public, "boundary")

    def test_query_and_candidate_permutation_preserves_alignment(self) -> None:
        public, _ = synthetic_arrays(3, 4)
        permuted, query_order = grade.permute_public(public, 123)
        self.assertFalse(np.array_equal(query_order, np.arange(3)))
        for query in range(3):
            token = 5 * permuted["candidate_model_id"][query] + permuted[
                "candidate_stage"
            ][query]
            values = permuted["candidate_tracks"][query, 0, :, 0].astype(int)
            self.assertTrue(np.array_equal(token, values))
            self.assertEqual(
                int(permuted["candidate_tracks"][query, 0, 0, 1]),
                int(query_order[query]),
            )
        states = np.full((3, 4), 9, np.uint8)
        original_tracks = grade.reconstruct_tracks(public, states)
        permuted_tracks = grade.reconstruct_tracks(permuted, states)
        self.assertTrue(np.array_equal(permuted_tracks, original_tracks[query_order]))


class ManifestSecurityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "heldout"
        self.root.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_valid_exact_bundle_loads(self) -> None:
        digest = write_bundle(self.root)
        cases, actual, _ = grade.load_hidden_bundle(self.root, digest)
        self.assertEqual(actual, digest)
        self.assertEqual(len(cases), 1)

    def test_public_label_leak_is_rejected(self) -> None:
        digest = write_bundle(self.root, extra_public=True)
        with self.assertRaises(GradeError):
            grade.load_hidden_bundle(self.root, digest)

    def test_symlink_artifact_is_rejected(self) -> None:
        digest = write_bundle(self.root)
        input_path = next((self.root / "inputs").iterdir())
        real = self.root / "real.npz"
        input_path.rename(real)
        input_path.symlink_to(real)
        with self.assertRaises(GradeError):
            grade.load_hidden_bundle(self.root, digest)

    def test_manifest_hash_mismatch_is_rejected(self) -> None:
        write_bundle(self.root)
        with self.assertRaises(GradeError):
            grade.load_hidden_bundle(self.root, "0" * 64)


class SubmissionIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.methods = self.root / "methods"
        self.stage = self.root / "stage"
        self.methods.mkdir()
        self.stage.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_helper_submission(self) -> Path:
        (self.methods / "helper.py").write_text(
            "import numpy as np\n"
            "def solve(q,t,o,e,m,s):\n"
            "    shape=t.shape[:2]\n"
            "    return np.full(shape,9,dtype=np.uint8),np.zeros(shape,dtype=bool)\n"
        )
        (self.methods / "model.json").write_text("{}\n")
        predictor = self.methods / "predict.py"
        predictor.write_text(
            "from helper import solve\n"
            "def predict(query_points,candidate_tracks,occlusion_logits,expected_dist_logits,candidate_model_id,candidate_stage):\n"
            "    return solve(query_points,candidate_tracks,occlusion_logits,expected_dist_logits,candidate_model_id,candidate_stage)\n"
        )
        return predictor

    def test_bundle_staging_preserves_helpers_and_hash(self) -> None:
        predictor = self._write_helper_submission()
        root, staged, digest = stage_submission(predictor, self.stage)
        self.assertTrue((root / "helper.py").is_file())
        self.assertTrue((root / "model.json").is_file())
        self.assertEqual(tree_sha256(root), digest)
        self.assertEqual(staged.stat().st_mode & 0o777, 0o444)

    def test_helper_import_works_in_fresh_child(self) -> None:
        predictor = self._write_helper_submission()
        root, staged, _ = stage_submission(predictor, self.stage)
        public, _ = synthetic_arrays()
        result = run_case(
            staged,
            public,
            stage_parent=self.stage,
            timeout_seconds=20,
            release_mode=False,
        )
        self.assertTrue(np.array_equal(result.state_token, np.full((2, 8), 9)))
        self.assertEqual(result.occluded.dtype, np.bool_)
        shutil.rmtree(root)

    def test_symlink_and_hardlink_are_rejected(self) -> None:
        predictor = self._write_helper_submission()
        target = self.methods / "data.json"
        target.write_text("{}")
        (self.methods / "link.json").symlink_to(target)
        with self.assertRaises(SessionError):
            stage_submission(predictor, self.stage)
        (self.methods / "link.json").unlink()
        os.link(target, self.methods / "hard.json")
        with self.assertRaises(SessionError):
            stage_submission(predictor, self.stage)

    def test_reward_json_has_exactly_one_key(self) -> None:
        original = grade.REWARD_DIR
        try:
            grade.REWARD_DIR = self.root / "logs"
            grade._write_results(
                {
                    "reward": 0.25,
                    "mean_video_AJ": 0.7,
                    "anchor_per_video": [{"case_id": "hidden", "weak_AJ": 0.1}],
                    "errors": ["hidden case identifier"],
                }
            )
            payload = json.loads((grade.REWARD_DIR / "reward.json").read_text())
            self.assertEqual(payload, {"reward": 0.25})
            summary = json.loads((grade.REWARD_DIR / "grade_debug.json").read_text())
            self.assertEqual(summary, {"mean_video_AJ": 0.7, "reward": 0.25})
            self.assertNotIn("anchor_per_video", summary)
            self.assertNotIn("errors", summary)
        finally:
            grade.REWARD_DIR = original


class RewardDirectorySecurityTest(unittest.TestCase):
    @unittest.skipUnless(os.geteuid() == 0, "root-only ownership test")
    def test_root_adopts_preexisting_host_owned_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reward_dir = Path(temporary) / "verifier"
            reward_dir.mkdir(mode=0o777)
            os.chown(reward_dir, 61224, 61224)
            grade._secure_reward_dir(reward_dir)
            info = reward_dir.lstat()
            self.assertEqual((info.st_uid, info.st_gid), (0, 0))
            self.assertEqual(info.st_mode & 0o777, 0o700)

    def test_nonroot_owner_mismatch_still_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reward_dir = Path(temporary) / "verifier"
            reward_dir.mkdir()
            different_uid = reward_dir.lstat().st_uid + 1
            with (
                mock.patch.object(grade.os, "geteuid", return_value=different_uid),
                mock.patch.object(grade.os, "chown") as chown,
                self.assertRaisesRegex(GradeError, "wrong owner"),
            ):
                grade._secure_reward_dir(reward_dir)
            chown.assert_not_called()

    def test_symlink_is_rejected_before_owner_adoption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            link = root / "verifier"
            real.mkdir()
            link.symlink_to(real, target_is_directory=True)
            with (
                mock.patch.object(grade.os, "chown") as chown,
                self.assertRaisesRegex(GradeError, "unsafe"),
            ):
                grade._secure_reward_dir(link)
            chown.assert_not_called()


class FailClosedEntrypointTest(unittest.TestCase):
    def test_grader_marker_survives_while_planted_reward_is_zeroed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = "grader failed: deliberate internal regression"
            (root / "reward.json").write_text('{"reward": 0.999}\n')
            (root / "reward.txt").write_text("0.999\n")
            (root / "grade_debug.json").write_text(
                json.dumps({"error": marker, "reward": 0.999}) + "\n"
            )
            retained = fail_closed.write_zero_results(root)
            self.assertEqual(retained, marker)
            self.assertEqual(json.loads((root / "reward.json").read_text()), {"reward": 0.0})
            self.assertEqual(float((root / "reward.txt").read_text()), 0.0)
            for name in ("grade_debug.json", "score_details.json"):
                payload = json.loads((root / name).read_text())
                self.assertEqual(payload["error"], marker)
                self.assertEqual(payload["reward"], 0.0)

    def test_submission_marker_and_malformed_artifacts_cannot_survive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "reward.json").write_text('{"reward": 1.0}\n')
            (root / "reward.txt").write_text("1.0\n")
            (root / "grade_debug.json").write_text(
                '{"error":"submission failed: planted marker","reward":1.0}\n'
            )
            (root / "score_details.json").write_text("not-json\n")
            retained = fail_closed.write_zero_results(root)
            self.assertEqual(retained, fail_closed.GENERIC_MARKER)
            self.assertEqual(json.loads((root / "reward.json").read_text()), {"reward": 0.0})
            self.assertEqual(float((root / "reward.txt").read_text()), 0.0)
            self.assertTrue(
                json.loads((root / "grade_debug.json").read_text())["error"].startswith(
                    "grader failed:"
                )
            )


if __name__ == "__main__":
    unittest.main()
