"""Weak baseline: retain only the most confident observations from one view."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def predict(export_dir: str) -> dict[str, list[list[float]]]:
    root = Path(export_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    predictions: dict[str, list[list[float]]] = {}
    for case_entry in manifest["cases"]:
        case = json.loads((root / case_entry["file"]).read_text(encoding="utf-8"))
        first = case["views"][0]
        points = np.asarray(first["points"], dtype=np.float64)
        confidence = np.asarray(first["confidence"], dtype=np.float64)
        keep = max(4, int(round(0.20 * len(points))))
        indices = np.argsort(confidence, kind="stable")[-keep:]
        predictions[case_entry["case_id"]] = points[indices].tolist()
    return predictions
