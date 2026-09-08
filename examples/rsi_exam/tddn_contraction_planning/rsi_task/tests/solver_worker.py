"""Minimal JSON-lines adapter around an untrusted submitted solver."""
from __future__ import annotations

import importlib.util
import json
import os
import resource
import sys
from pathlib import Path
from typing import Any


def restrict_process() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))


def load_solver(path: Path) -> Any:
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("planner_submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import solver from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "select_edge", None)):
        raise TypeError("solver.py must define callable select_edge(observation)")
    return module


def emit(stream: Any, value: dict[str, Any]) -> None:
    stream.write((json.dumps(value, separators=(",", ":")) + "\n").encode())


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: solver_worker.py /path/to/solver.py")

    # Preserve one private protocol FD, then silence submission stdout/stderr.
    protocol_fd = os.dup(1)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null_fd, 1)
    os.dup2(null_fd, 2)
    os.close(null_fd)
    protocol = os.fdopen(protocol_fd, "wb", buffering=0)

    restrict_process()
    try:
        solver = load_solver(Path(sys.argv[1]))
        emit(protocol, {"type": "ready"})
    except Exception as error:
        emit(protocol, {
            "type": "error",
            "error": f"{type(error).__name__}: {error}",
        })
        return

    for raw in sys.stdin.buffer:
        try:
            request = json.loads(raw)
            nonce = str(request["nonce"])
            action = int(solver.select_edge(request["observation"]))
            emit(protocol, {"type": "action", "nonce": nonce, "action": action})
        except Exception as error:
            emit(protocol, {
                "type": "error",
                "error": f"{type(error).__name__}: {error}",
            })
            return


if __name__ == "__main__":
    main()
