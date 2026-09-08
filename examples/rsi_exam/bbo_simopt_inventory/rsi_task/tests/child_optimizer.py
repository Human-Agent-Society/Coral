#!/usr/bin/env python3
"""Untrusted BBO child process.

Protocol: newline-delimited JSON frames over stdin/stdout.
Commands:
- init  -> build Optimizer(dim=..., lower=np.array(...), upper=np.array(...), budget=..., seed=..., rng=np.random.default_rng(seed))
- ask   -> call opt.ask(n) and return finite float matrix X with shape [n, dim]
- tell  -> call opt.tell(X, y, metadata)
- close -> acknowledge and exit

The child reserves original stdout for protocol frames only. Any submission prints are diverted to stderr.
"""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ORIGINAL_STDOUT = sys.stdout
sys.stdout = sys.stderr


class ProtocolError(ValueError):
    """Raised when a parent frame is malformed."""


def _emit(obj: dict[str, Any]) -> None:
    ORIGINAL_STDOUT.write(json.dumps(obj) + "\n")
    ORIGINAL_STDOUT.flush()


def _read_frame(line: str) -> dict[str, Any]:
    try:
        frame = json.loads(line)
    except json.JSONDecodeError as exc:  # noqa: PERF203
        raise ProtocolError(f"invalid json: {exc.msg}") from exc
    if not isinstance(frame, dict):
        raise ProtocolError("frame must be a JSON object")
    return frame


def _require(frame: dict[str, Any], key: str) -> Any:
    if key not in frame:
        raise ProtocolError(f"missing field: {key}")
    return frame[key]


def _as_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{name} must be an integer")
    return value


def _as_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be numeric")
    out = float(value)
    if not np.isfinite(out):
        raise ProtocolError(f"{name} must be finite")
    return out


def _as_vector(value: Any, *, name: str) -> np.ndarray:
    if not isinstance(value, list):
        raise ProtocolError(f"{name} must be a list")
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ProtocolError(f"{name} must be 1-D")
    if not np.isfinite(arr).all():
        raise ProtocolError(f"{name} must contain only finite values")
    return arr


def _as_matrix(value: Any, *, name: str, dim: int | None = None) -> np.ndarray:
    if not isinstance(value, list):
        raise ProtocolError(f"{name} must be a list")
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ProtocolError(f"{name} must be 2-D")
    if dim is not None and arr.shape[1] != dim:
        raise ProtocolError(f"{name} must have shape [n, {dim}]")
    if not np.isfinite(arr).all():
        raise ProtocolError(f"{name} must contain only finite values")
    return arr


class _State:
    def __init__(self) -> None:
        self.optimizer: Any | None = None
        self.dim: int | None = None
        self.budget: int = 1


def _safe_batch_hint(state: _State) -> int:
    budget_cap = max(1, int(state.budget))
    if state.optimizer is None:
        return 1
    try:
        raw = getattr(state.optimizer, 'batch', 1)
        if isinstance(raw, bool):
            raise TypeError('bool is not a valid batch size')
        if isinstance(raw, (int, np.integer)):
            batch = int(raw)
        elif isinstance(raw, float) and np.isfinite(raw) and raw.is_integer():
            batch = int(raw)
        else:
            raise TypeError('batch must be an integer')
    except Exception:  # noqa: BLE001
        batch = 1
    if batch < 1:
        batch = 1
    return min(batch, budget_cap)


def _load_optimizer(submission_solver: Path):
    sys.argv = [str(submission_solver)]
    spec = importlib.util.spec_from_file_location("submitted_solver", submission_solver)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import solver from {submission_solver}")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
    if not hasattr(module, "Optimizer"):
        raise RuntimeError("solver.py must define Optimizer")
    return module.Optimizer


def _handle_init(state: _State, frame: dict[str, Any]) -> dict[str, Any]:
    if state.optimizer is not None:
        raise ProtocolError("second init is not permitted; each child serves one run")

    payload = _require(frame, "payload")
    if not isinstance(payload, dict):
        raise ProtocolError("payload must be an object")

    submission_solver = Path(_require(payload, "solver_path"))
    dim = _as_int(_require(payload, "dim"), name="dim")
    budget = _as_int(_require(payload, "budget"), name="budget")
    seed = _as_int(_require(payload, "seed"), name="seed")
    lower = _as_vector(_require(payload, "lower"), name="lower")
    upper = _as_vector(_require(payload, "upper"), name="upper")
    if len(lower) != dim or len(upper) != dim:
        raise ProtocolError("lower/upper length must match dim")

    Optimizer = _load_optimizer(submission_solver)
    rng = np.random.default_rng(seed)
    state.optimizer = Optimizer(dim=dim, lower=lower, upper=upper, budget=budget, seed=seed, rng=rng)
    state.dim = dim
    state.budget = max(1, budget)
    return {"ok": True, "event": "init", "dim": dim, "batch_size": _safe_batch_hint(state)}


def _handle_ask(state: _State, frame: dict[str, Any]) -> dict[str, Any]:
    if state.optimizer is None or state.dim is None:
        raise ProtocolError("init required before ask")
    payload = _require(frame, "payload")
    if not isinstance(payload, dict):
        raise ProtocolError("payload must be an object")
    batch_size = _as_int(_require(payload, "batch_size"), name="batch_size")
    X = np.asarray(state.optimizer.ask(batch_size), dtype=float)
    if X.ndim != 2 or X.shape[1] != state.dim:
        raise ProtocolError(f"ask must return shape [n, {state.dim}]")
    if not np.isfinite(X).all():
        raise ProtocolError("ask must return only finite values")
    return {"ok": True, "event": "ask", "X": X.tolist(), "batch_size": _safe_batch_hint(state)}


def _handle_tell(state: _State, frame: dict[str, Any]) -> dict[str, Any]:
    if state.optimizer is None or state.dim is None:
        raise ProtocolError("init required before tell")
    payload = _require(frame, "payload")
    if not isinstance(payload, dict):
        raise ProtocolError("payload must be an object")
    X = _as_matrix(_require(payload, "X"), name="X", dim=state.dim)
    y_raw = _require(payload, "y")
    if not isinstance(y_raw, list):
        raise ProtocolError("y must be a list")
    y = np.asarray([_as_float(v, name="y") for v in y_raw], dtype=float)
    if y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ProtocolError("X and y batch sizes must match")
    metadata = payload.get("metadata")
    tell = state.optimizer.tell
    try:
        sig = inspect.signature(tell)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"unable to inspect optimizer tell signature: {exc}") from exc

    if metadata is None:
        try:
            sig.bind(X, y, metadata)
        except TypeError:
            try:
                sig.bind(X, y)
            except TypeError as exc:
                raise ProtocolError(f"incompatible optimizer tell signature: {exc}") from exc
            tell(X, y)
        else:
            tell(X, y, metadata)
    else:
        try:
            sig.bind(X, y, metadata)
        except TypeError as exc:
            raise ProtocolError("optimizer tell() must accept metadata for non-null metadata payloads") from exc
        tell(X, y, metadata)
    return {"ok": True, "event": "tell", "n": int(X.shape[0]), "metadata": metadata, "batch_size": _safe_batch_hint(state)}


def main() -> int:
    state = _State()
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            frame = _read_frame(raw)
            command = _require(frame, "command")
            if command == "init":
                reply = _handle_init(state, frame)
            elif command == "ask":
                reply = _handle_ask(state, frame)
            elif command == "tell":
                reply = _handle_tell(state, frame)
            elif command == "close":
                _emit({"ok": True, "event": "close"})
                return 0
            else:
                raise ProtocolError(f"unknown command: {command}")
        except Exception as exc:  # noqa: BLE001
            _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return 1
        _emit(reply)
    _emit({"ok": False, "error": "ProtocolError: stdin closed before close"})
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
