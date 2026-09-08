from __future__ import annotations

import contextlib
import ctypes
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterator


@contextlib.contextmanager
def restricted_workspace(
    source: Path,
    guard: Callable[[Path], list[str]],
    *,
    case_index: int,
) -> Iterator[tuple[Path, Path, dict[str, str], int]]:
    failures = guard(source)
    if failures:
        raise RuntimeError("submission rejected:\n" + "\n".join(failures))
    root = Path(tempfile.mkdtemp(prefix=f"gpu-case-{case_index:02d}-", dir="/tmp"))
    candidate, runtime = root / "submission", root / "runtime"
    shutil.copytree(source, candidate)
    runtime.mkdir()
    for path in candidate.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    candidate.chmod(0o555)
    root.chmod(0o711)
    uid = 10001 + case_index
    os.chown(runtime, uid, uid)
    runtime.chmod(0o700)
    env = {
        "HOME": str(runtime),
        "PATH": "/opt/conda/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": "/runner",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_CACHE_PATH": str(runtime / "cuda-cache"),
        "TRITON_CACHE_DIR": str(runtime / "triton-cache"),
        "TORCHINDUCTOR_CACHE_DIR": str(runtime / "torchinductor-cache"),
        "TMPDIR": str(runtime),
    }
    try:
        yield candidate, runtime, env, uid
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run_restricted(command: list[str], *, cwd: Path, env: dict[str, str], uid: int, timeout: int):
    def prepare():
        os.setsid()
        os.umask(0o077)
        try:
            ctypes.CDLL(None).prctl(38, 1, 0, 0, 0)
        except Exception:
            pass
        os.setgroups([])
        os.setgid(uid)
        os.setuid(uid)

    started = time.monotonic()
    process = subprocess.Popen(
        command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, preexec_fn=prepare,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        raise RuntimeError(f"candidate timed out\n{stdout}\n{stderr}")
    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    completed.wall_time_ms = (time.monotonic() - started) * 1000.0
    return completed
