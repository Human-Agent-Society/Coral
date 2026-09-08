#!/usr/bin/env python3
"""Invoke one submitted switch-budget router on one public case."""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from pathlib import Path

import numpy as np


PUBLIC_FIELDS = (
    "query_points",
    "candidate_tracks",
    "occlusion_logits",
    "expected_dist_logits",
    "candidate_model_id",
    "candidate_stage",
)


def load_predictor(path: Path):
    sibling_path = str(path.parent)
    sys.path.insert(0, sibling_path)
    spec = importlib.util.spec_from_file_location("submitted_candidate_router", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import submitted predict.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predict = getattr(module, "predict", None)
    if not callable(predict):
        raise RuntimeError("predict.py must define callable predict(...)")
    return predict


def main() -> None:
    if len(sys.argv) != 4:
        raise RuntimeError("usage: child_entry.py PREDICTOR INPUT_NPZ OUTPUT_NPZ")
    predictor = Path(sys.argv[1])
    input_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    with np.load(input_path, allow_pickle=False) as archive:
        if tuple(sorted(archive.files)) != tuple(sorted(PUBLIC_FIELDS)):
            raise RuntimeError("sanitized input has an unexpected schema")
        values = [np.asarray(archive[name]) for name in PUBLIC_FIELDS]

    predict = load_predictor(predictor)
    result = predict(*values)
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError("predict(...) must return (state_token, occluded)")
    state_token, occluded = result
    np.savez_compressed(
        output_path,
        pred_state_token=np.asarray(state_token),
        pred_occluded=np.asarray(occluded),
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        os._exit(1)
