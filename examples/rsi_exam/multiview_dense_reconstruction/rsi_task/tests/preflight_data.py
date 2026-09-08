#!/usr/bin/env python3
"""Build-time structural preflight for the sealed procedural cloud package."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def finite_points(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 4
        and all(
            isinstance(point, list)
            and len(point) == 3
            and all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                for item in point
            )
            for point in value
        )
    )


def main() -> None:
    root = Path(sys.argv[1])
    observations, truth_root = root / "observations", root / "gt"
    manifest = json.loads((observations / "manifest.json").read_text())
    require(set(manifest) == {"schema_version", "split", "cases"}, "manifest keys")
    require(manifest["schema_version"] == "multiview-cloud-v4", "schema")
    require(manifest["split"] == "sealed", "split")
    case_ids: list[str] = []
    for index, entry in enumerate(manifest["cases"], start=1):
        case_id = f"sealed_case_{index:03d}"
        require(set(entry) == {"case_id", "file", "num_views"}, "entry keys")
        require(entry["case_id"] == case_id and entry["file"] == f"{case_id}.json", "entry identity")
        case = json.loads((observations / entry["file"]).read_text())
        truth = json.loads((truth_root / f"{case_id}_gt.json").read_text())
        require(set(case) == {"case_id", "views"}, "case keys")
        require(set(truth) == {"case_id", "points"}, "truth keys")
        require(case["case_id"] == truth["case_id"] == case_id, "case identity")
        require(len(case["views"]) == entry["num_views"], "view count")
        require(finite_points(truth["points"]), "truth points")
        for view_index, view in enumerate(case["views"], start=1):
            require(set(view) == {"view_id", "points", "confidence"}, "view keys")
            require(view["view_id"] == f"view_{view_index:03d}", "view identity")
            require(finite_points(view["points"]), "view points")
            require(len(view["confidence"]) == len(view["points"]), "confidence length")
            require(
                all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and 0.0 <= float(value) <= 1.0
                    for value in view["confidence"]
                ),
                "confidence range",
            )
        case_ids.append(case_id)
    anchor_path = Path("/tests/anchors.json")
    anchors = json.loads((anchor_path if anchor_path.is_file() else Path(__file__).parent / "anchors.json").read_text())
    require(anchors["schema_version"] == "dense-calibration-v7", "anchor schema")
    require(set(anchors["per_case_diagnostic"]) == set(case_ids), "anchor cases")
    require(anchors["thresholds"] == [0.0125, 0.025, 0.05], "thresholds")
    require(
        0.0 <= anchors["baseline_fscore"] < anchors["reference_fscore"] < anchors["upper_fscore"] == 1.0,
        "anchor order",
    )
    for row in anchors["per_case_diagnostic"].values():
        require(0.0 <= row["baseline_fscore"] < row["reference_fscore"] < anchors["upper_fscore"] == 1.0, "anchor order")
    require(
        {path.name for path in observations.iterdir()}
        == {"manifest.json", *(f"{case_id}.json" for case_id in case_ids)},
        "observation extras",
    )
    require(
        {path.name for path in truth_root.iterdir()}
        == {f"{case_id}_gt.json" for case_id in case_ids},
        "truth extras",
    )
    print(f"sealed preflight passed: {len(case_ids)} cases")


if __name__ == "__main__":
    main()
