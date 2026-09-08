from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def train(
    data_path: str = "/app/data/train.npz",
    output_dir: str = "/app/submission",
    config: dict | None = None,
):
    """Create the initial control-passthrough checkpoint.

    This deliberately weak method is a valid end-to-end starting point. Replace
    it with a learned perturbation model while preserving the public interface.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(data_path, allow_pickle=False) as data:
        summary = {
            "format": "control_passthrough_v1",
            "treated_samples": int(len(data["treated"])),
            "control_samples": int(len(data["control_bank"])),
            "config": config or {},
        }
    (output / "checkpoint.pt").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def predict(
    inputs_path: str,
    checkpoint_dir: str = "/app/submission",
    output_path: str = "/tmp/predictions.npz",
):
    checkpoint = Path(checkpoint_dir) / "checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    with np.load(inputs_path, allow_pickle=False) as data:
        prediction = data["control"].copy()
        sample_id = data["sample_id"].copy()
    np.savez_compressed(output_path, prediction=prediction, sample_id=sample_id)
    return output_path
