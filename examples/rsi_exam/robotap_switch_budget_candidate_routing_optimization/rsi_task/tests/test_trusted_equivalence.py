from __future__ import annotations

import importlib.util
import itertools
import unittest
from pathlib import Path

import numpy as np


def load_trusted():
    path = Path(__file__).resolve().parent / "trusted" / "reference_inference.py"
    spec = importlib.util.spec_from_file_location("trusted_routing_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrustedRoutingDynamicProgramTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_trusted()

    def test_matches_exhaustive_optimum(self) -> None:
        rng = np.random.default_rng(20260719)
        for length in (2, 3, 5):
            scores = rng.normal(size=(1, length, 3)).astype(np.float32)
            query_points = np.asarray([[0.0, 0.0, 0.0]], np.float32)
            budget = 2
            route = self.module.budget_path(scores, query_points, budget)
            chosen = tuple(int(value) for value in route[0, 1:])
            switches = sum(a != b for a, b in zip(chosen, chosen[1:]))
            self.assertLessEqual(switches, budget)
            actual = sum(float(scores[0, frame, state]) for frame, state in enumerate(chosen, 1))
            expected = max(
                sum(float(scores[0, frame, state]) for frame, state in enumerate(candidate, 1))
                for candidate in itertools.product(range(3), repeat=max(length - 1, 0))
                if sum(a != b for a, b in zip(candidate, candidate[1:])) <= budget
            )
            self.assertAlmostEqual(actual, expected, places=6)

    def test_zero_and_one_scored_frame_use_no_switches(self) -> None:
        scores = np.zeros((2, 3, 3), np.float32)
        queries = np.asarray([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float32)
        route = self.module.budget_path(scores, queries, 4)
        self.assertEqual(route.shape, (2, 3))
        self.assertTrue(np.array_equal(route, np.zeros((2, 3), np.int16)))

    def test_ties_prefer_fewer_switches_and_smaller_token(self) -> None:
        scores = np.zeros((1, 7, 4), np.float32)
        queries = np.asarray([[0.0, 0.0, 0.0]], np.float32)
        route = self.module.budget_path(scores, queries, 4)
        self.assertTrue(np.array_equal(route, np.zeros((1, 7), np.int16)))


if __name__ == "__main__":
    unittest.main()
