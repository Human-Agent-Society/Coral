#!/usr/bin/env python3
"""Run an untrusted game policy behind a small action-only JSON protocol."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import resource
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time


MAX_SOURCE_BYTES = 10_000_000   # keep in step with grade.py and selfcheck.py
MAX_LINE_BYTES = 64_000
ACTION_TIMEOUT_SEC = 5.0
SUITE_CPU_SEC = 3600            # whole sealed suite; see docs/2048-cpu-budget.md
UNTRUSTED_UID = 65534
UNTRUSTED_GID = 65534


def _child_setup() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (SUITE_CPU_SEC, SUITE_CPU_SEC + 20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1_048_576, 1_048_576))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
    except (ValueError, OSError):
        pass
    os.setgroups([])
    os.setgid(UNTRUSTED_GID)
    os.setuid(UNTRUSTED_UID)


def _stage_policy(source_dir: Path, destination: Path) -> Path:
    destination.mkdir(mode=0o755)
    total = 0
    for source in sorted(source_dir.rglob("*")):
        if "__pycache__" in source.parts or source.suffix in {".pyc", ".pyo"}:
            continue
        if source.is_symlink():
            raise ValueError("policy source may not contain symlinks")
        if source.is_dir():
            relative = source.relative_to(source_dir)
            (destination / relative).mkdir(mode=0o755, parents=True, exist_ok=True)
            continue
        if not source.is_file() or source.suffix != ".py":
            raise ValueError("policy source may contain only Python files")
        total += source.stat().st_size
        if total > MAX_SOURCE_BYTES:
            raise ValueError("policy source exceeds 10 MB")
        target = destination / source.relative_to(source_dir)
        target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o644)
    policy = destination / "policy.py"
    if not policy.is_file():
        raise ValueError("missing regular methods/main/policy.py")
    return policy


class PolicyProcess:
    def __init__(self, policy_path: Path, kind: str):
        self._tmp = tempfile.TemporaryDirectory(prefix="game-policy-")
        root = Path(self._tmp.name)
        root.chmod(0o755)
        staged = _stage_policy(policy_path.parent, root / "source")
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": "/tmp",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
        }
        self._process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--child", str(staged), kind],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd="/tmp",
            env=env,
            bufsize=0,
            start_new_session=True,
            preexec_fn=_child_setup,
        )
        self._buffer = bytearray()

    def _readline(self) -> bytes:
        assert self._process.stdout is not None
        fd = self._process.stdout.fileno()
        deadline = time.monotonic() + ACTION_TIMEOUT_SEC
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._buffer[:newline])
                del self._buffer[: newline + 1]
                return line
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("policy action timed out")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                raise TimeoutError("policy action timed out")
            chunk = os.read(fd, 4096)
            if not chunk:
                raise RuntimeError("policy process exited without an action")
            self._buffer.extend(chunk)
            if len(self._buffer) > MAX_LINE_BYTES:
                raise ValueError("policy response exceeds 64 KB")

    def call(self, args: list[object]) -> object:
        if self._process.poll() is not None:
            raise RuntimeError("policy process is not running")
        assert self._process.stdin is not None
        request = json.dumps({"args": args}, separators=(",", ":")).encode() + b"\n"
        self._process.stdin.write(request)
        self._process.stdin.flush()
        response = json.loads(self._readline())
        if not isinstance(response, dict) or set(response) != {"ok", "value"}:
            raise ValueError("malformed policy response")
        if response["ok"] is not True:
            raise RuntimeError(str(response["value"])[:500])
        return response["value"]

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                os.killpg(self._process.pid, signal.SIGTERM)
                self._process.wait(timeout=1)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self._process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self._tmp.cleanup()

    def __enter__(self) -> "PolicyProcess":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _child_main(policy_path: Path, kind: str) -> int:
    protocol_in = sys.stdin.buffer
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    loads, dumps = json.loads, json.dumps
    spec = importlib.util.spec_from_file_location("submitted_policy", policy_path)
    if spec is None or spec.loader is None:
        return 2
    sys.path.insert(0, str(policy_path.parent))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function_name = "choose_move" if kind == "2048" else "choose_action"
    function = getattr(module, function_name, None)
    if not callable(function):
        return 2
    for raw in protocol_in:
        try:
            request = loads(raw)
            args = request["args"]
            if kind == "2048":
                board = tuple(tuple(int(value) for value in row) for row in args[0])
                value = function(board)
                if type(value) is not str:
                    raise TypeError("choose_move must return a string")
            else:
                board = tuple(tuple(int(value) for value in row) for row in args[0])
                incoming = (int(args[6][0]), tuple(int(value) for value in args[6][1]))
                decision = function(
                    board, str(args[1]), tuple(str(value) for value in args[2]),
                    None if args[3] is None else str(args[3]), int(args[4]),
                    bool(args[5]), incoming,
                )
                value = [bool(decision[0]), int(decision[1]), int(decision[2])]
            response = {"ok": True, "value": value}
        except BaseException as exc:
            response = {"ok": False, "value": f"{type(exc).__name__}: {exc}"[:500]}
        protocol_out.write(dumps(response, separators=(",", ":")).encode() + b"\n")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--child":
        raise SystemExit(2)
    raise SystemExit(_child_main(Path(sys.argv[2]), sys.argv[3]))
