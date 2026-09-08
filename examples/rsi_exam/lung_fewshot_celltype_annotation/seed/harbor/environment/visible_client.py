"""Client for the score-only visible evaluator."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import pandas as pd


MAX_RESPONSE_BYTES = 1024 * 1024


def score(predictions_path: Path, socket_path: Path) -> dict[str, object]:
    frame = pd.read_csv(predictions_path, dtype=str)
    if list(frame.columns) != ["cell_id", "pred_label"]:
        raise ValueError("predictions must have columns cell_id,pred_label")
    if frame.isna().any().any() or frame["cell_id"].duplicated().any():
        raise ValueError("predictions contain missing values or duplicate cell IDs")
    payload = json.dumps(
        {"cell_ids": frame["cell_id"].tolist(), "pred_labels": frame["pred_label"].tolist()},
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(30.0)
        client.connect(str(socket_path))
        client.sendall(payload)
        stream = client.makefile("rb")
        response = stream.readline(MAX_RESPONSE_BYTES + 1)
    if len(response) > MAX_RESPONSE_BYTES:
        raise RuntimeError("visible evaluator response was too large")
    result = json.loads(response)
    if not result.get("ok"):
        raise RuntimeError(str(result.get("error", "visible evaluation failed")))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--socket", type=Path, default=Path("/app/.visible-evaluator/evaluator.sock"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = score(args.predictions, args.socket)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
