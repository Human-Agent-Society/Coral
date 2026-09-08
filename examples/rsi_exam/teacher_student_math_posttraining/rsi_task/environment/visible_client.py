#!/usr/bin/env python3
"""Submit one complete prediction vector to the aggregate-only development scorer."""
from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

SOCKET = Path("/app/.visible-evaluator/evaluator.sock")


def score(predictions: dict[str, int]) -> dict[str, object]:
    payload = (json.dumps({"predictions": predictions}, separators=(",", ":")) + "\n").encode()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(SOCKET))
        connection.sendall(payload)
        response = connection.makefile("rb").readline(64 * 1024)
    result = json.loads(response)
    if not isinstance(result, dict):
        raise RuntimeError("invalid visible-evaluator response")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, help="JSON object mapping every dev ID to one integer")
    args = parser.parse_args()
    raw = json.loads(args.predictions.read_text())
    print(json.dumps(score({str(key): int(value) for key, value in raw.items()}), sort_keys=True))


if __name__ == "__main__":
    main()
