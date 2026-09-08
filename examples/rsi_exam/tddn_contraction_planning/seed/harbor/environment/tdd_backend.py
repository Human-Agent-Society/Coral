"""Streaming client for the trusted exact TDD contraction engine."""
from __future__ import annotations

import json
import os
import select
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_ENGINE = Path("/usr/local/bin/tddn_engine")


class TDDEnvironment:
    """One exact contraction case backed by an isolated native process."""

    def __init__(
        self,
        case: dict[str, Any],
        *,
        engine: Path = DEFAULT_ENGINE,
        step_timeout: float = 120.0,
    ) -> None:
        self._timeout = step_timeout
        self._pending = b""
        self._observation: dict[str, Any] | None = None
        self._result: dict[str, Any] | None = None
        self._process = subprocess.Popen(
            [str(engine)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        self._send(case)
        message = self._read_message()
        if message.get("type") != "ready":
            self.close()
            raise RuntimeError(f"TDD engine did not become ready: {message}")
        self._observation = message["observation"]
        if not any(self._observation.get("action_mask", {}).values()):
            terminal = self._read_message()
            if terminal.get("type") != "result":
                self.close()
                raise RuntimeError(f"Expected terminal TDD result: {terminal}")
            self._result = terminal
            self._observation = None

    def _send(self, value: dict[str, Any]) -> None:
        if self._process.stdin is None:
            raise RuntimeError("TDD engine input is closed")
        data = (json.dumps(value, separators=(",", ":")) + "\n").encode()
        self._process.stdin.write(data)
        self._process.stdin.flush()

    def _read_message(self) -> dict[str, Any]:
        if self._process.stdout is None:
            raise RuntimeError("TDD engine output is closed")
        deadline = time.monotonic() + self._timeout
        fd = self._process.stdout.fileno()
        while True:
            while b"\n" in self._pending:
                raw, self._pending = self._pending.split(b"\n", 1)
                try:
                    value = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict) and value.get("type") in {
                    "ready",
                    "step",
                    "result",
                    "error",
                }:
                    return value

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("TDD engine step timed out")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                raise TimeoutError("TDD engine step timed out")
            chunk = os.read(fd, 65536)
            if not chunk:
                code = self._process.poll()
                raise RuntimeError(f"TDD engine exited early with status {code}")
            self._pending += chunk

    @property
    def is_terminal(self) -> bool:
        return self._result is not None

    @property
    def latest_tdd_size(self) -> int:
        if self._result is not None:
            return int(self._result["final_tdd_nodes"])
        assert self._observation is not None
        return int(self._observation["global_features"]["latest_tdd_size"])

    def observation(self) -> dict[str, Any]:
        if self._observation is None or self.is_terminal:
            raise RuntimeError("No non-terminal observation is available")
        return self._observation

    def contract(self, edge_id: int) -> None:
        if self.is_terminal:
            raise RuntimeError("Cannot contract a terminal environment")
        self._send({"action": int(edge_id)})
        message = self._read_message()
        if message["type"] == "error":
            raise RuntimeError(str(message.get("error", "TDD engine error")))
        if message["type"] == "step":
            self._observation = message["observation"]
            return
        if message["type"] == "result":
            self._result = message
            self._observation = None
            return
        raise RuntimeError(f"Unexpected TDD engine response: {message}")

    def result(self) -> dict[str, Any]:
        if self._result is None:
            raise RuntimeError("Contraction is not complete")
        return dict(self._result)

    def close(self) -> None:
        if getattr(self, "_process", None) is None:
            return
        if self._process.poll() is None:
            self._process.kill()
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass

    def __enter__(self) -> "TDDEnvironment":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def make_environment(case: dict[str, Any]) -> TDDEnvironment:
    return TDDEnvironment(case)
