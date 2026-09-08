#!/usr/bin/env python3
"""Aggregate-only development scorer; answer data never enters the agent container."""
from __future__ import annotations

import json
import os
import socket
from pathlib import Path

SOCKET = Path("/run/math-visible/evaluator.sock")
ANSWERS = Path("/opt/visible/answers.json")
QUERY_BUDGET = 16
MIN_CHANGED = 10
MAX_BYTES = 64 * 1024


def line(value: dict[str, object]) -> bytes:
    return (json.dumps(value, separators=(",", ":")) + "\n").encode()


def main() -> None:
    truth = {str(k): int(v) for k, v in json.loads(ANSWERS.read_text()).items()}
    expected = sorted(truth)
    SOCKET.parent.mkdir(parents=True, exist_ok=True)
    SOCKET.unlink(missing_ok=True)
    history: list[tuple[int, ...]] = []
    used = 0
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(SOCKET))
        os.chmod(SOCKET, 0o666)
        server.listen(8)
        while True:
            connection, _ = server.accept()
            with connection:
                payload = connection.makefile("rb").readline(MAX_BYTES + 1)
                used += 1
                try:
                    if len(payload) > MAX_BYTES:
                        raise ValueError("request too large")
                    request = json.loads(payload)
                    if not isinstance(request, dict) or set(request) != {"predictions"}:
                        raise ValueError("request must contain only predictions")
                    predictions = request["predictions"]
                    if not isinstance(predictions, dict) or sorted(predictions) != expected:
                        raise ValueError("prediction IDs must exactly match the development set")
                    vector = tuple(int(predictions[key]) for key in expected)
                    if any(value < -1 or value > 999 for value in vector):
                        raise ValueError("predictions must be integers in [-1,999]")
                    if any(sum(a != b for a, b in zip(vector, prior)) < MIN_CHANGED for prior in history):
                        raise ValueError(f"each scored vector must change at least {MIN_CHANGED} predictions")
                    if used > QUERY_BUDGET:
                        raise ValueError("development score query budget exhausted")
                    history.append(vector)
                    correct = sum(value == truth[key] for key, value in zip(expected, vector))
                    response = {
                        "ok": True,
                        "accuracy": round(correct / len(expected), 4),
                        "queries_used": used,
                        "queries_remaining": QUERY_BUDGET - used,
                    }
                except Exception as exc:  # malformed attempts consume budget
                    response = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "queries_used": used,
                        "queries_remaining": max(0, QUERY_BUDGET - used),
                    }
                connection.sendall(line(response))


if __name__ == "__main__":
    main()
