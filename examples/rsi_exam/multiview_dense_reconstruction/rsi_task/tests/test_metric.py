#!/usr/bin/env python3
"""Trusted dense metric and anchor-landing regressions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grade  # noqa: E402


def main() -> None:
    truth = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.1, 0.0]],
        dtype=np.float64,
    )
    metric, details = grade._case_metric(truth, truth)
    assert metric == 1.0
    assert [row["fscore"] for row in details] == [1.0, 1.0, 1.0]
    shifted = truth + np.array([1.0, 1.0, 1.0])
    metric, _ = grade._case_metric(shifted, truth)
    assert metric == 0.0
    for baseline, reference in ((0.0, 0.5), (0.1, 0.9)):
        assert grade.scene_reward(baseline, baseline, reference, 1.0) == 0.0
        assert grade.scene_reward(reference, baseline, reference, 1.0) == 0.6
        assert grade.scene_reward(1.0, baseline, reference, 1.0) == 1.0

    # Duplicate padding must be worth exactly nothing. Precision is a mean over the
    # returned points, so before validate_predictions deduplicated, repeating an accurate
    # point while dropping nothing raised the F-score for free.
    partial = np.concatenate([truth[:2], truth[2:] + 0.4])
    plain, _ = grade._case_metric(partial, truth)
    padded_raw = np.concatenate([partial, np.repeat(truth[:1], 60, axis=0)])
    inflated, _ = grade._case_metric(padded_raw, truth)
    assert inflated > plain, "the padding leak must be reproducible before the fix"
    deduped, _ = grade._case_metric(np.unique(padded_raw, axis=0), truth)
    assert deduped == plain, f"dedup must neutralise padding: {deduped} != {plain}"

    payload = json.dumps(
        {"case_a": padded_raw.tolist(), "case_b": truth.tolist()}
    ).encode()
    validated = grade.validate_predictions(payload, {"case_a": {}, "case_b": {}})
    assert len(validated["case_a"]) == len(np.unique(partial, axis=0)), (
        "validate_predictions must strip exact duplicates before scoring"
    )
    scored, _ = grade._case_metric(validated["case_a"], truth)
    assert scored == plain

    print("metric tests passed")


if __name__ == "__main__":
    main()
