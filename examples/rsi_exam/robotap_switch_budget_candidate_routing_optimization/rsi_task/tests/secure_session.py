"""Low-privilege execution for one switch-budgeted routing case."""

from __future__ import annotations

import functools
import hashlib
import os
import resource
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


MAX_SUBMISSION_FILE_BYTES = 32 * 1024 * 1024
MAX_SUBMISSION_TOTAL_BYTES = 64 * 1024 * 1024
MAX_SUBMISSION_FILES = 64
ALLOWED_SUBMISSION_SUFFIXES = {".py", ".json", ".joblib", ".npz", ".npy"}
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
MEMORY_LIMIT_BYTES = 3 * 1024 * 1024 * 1024
FILE_LIMIT_BYTES = 160 * 1024 * 1024
PIDS_LIMIT = 24
NOFILE_LIMIT = 64
UNTRUSTED_UID = 61224
UNTRUSTED_GID = 61224
CHILD = Path("/tests/child_entry.py")
DEV_CHILD = Path(__file__).with_name("child_entry.py").resolve()
BWRAP = Path("/usr/bin/bwrap")
SETPRIV = Path("/usr/bin/setpriv")
PYTHON_PREFIX = Path(sys.prefix).resolve()


class SessionError(RuntimeError):
    """The submitted predictor or its isolation session failed closed."""


class InfrastructureError(SessionError):
    """The trusted verifier environment cannot enforce its release contract."""

@dataclass(frozen=True)
class PredictionResult:
    state_token: np.ndarray
    occluded: np.ndarray
    elapsed_seconds: float
    stderr_tail: str


def _read_stable_file(path: Path) -> bytes:
    if path.is_symlink() or path.suffix not in ALLOWED_SUBMISSION_SUFFIXES:
        raise SessionError("submission bundle contains an unsafe file")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SessionError("submission bundle file is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise SessionError("submission files must be regular and non-hardlinked")
        if not 1 <= before.st_size <= MAX_SUBMISSION_FILE_BYTES:
            raise SessionError("submission file size is outside limits")
        payload = bytearray()
        while len(payload) <= MAX_SUBMISSION_FILE_BYTES:
            chunk = os.read(
                descriptor,
                min(1 << 20, MAX_SUBMISSION_FILE_BYTES + 1 - len(payload)),
            )
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if identity(before) != identity(after) or len(payload) != before.st_size:
            raise SessionError("submission file changed while being staged")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _hash_entries(entries: list[tuple[Path, bytes]]) -> str:
    digest = hashlib.sha256()
    for relative, payload in entries:
        encoded = relative.as_posix().encode()
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def _read_bundle(root: Path) -> list[tuple[Path, bytes]]:
    if root.is_symlink() or not root.is_dir():
        raise SessionError("submission method root is unsafe")
    entries: list[tuple[Path, bytes]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise SessionError(f"submission bundle contains symlink: {relative}")
        if path.is_dir():
            continue
        if not stat.S_ISREG(info.st_mode):
            raise SessionError(f"submission bundle contains unsupported entry: {relative}")
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        entries.append((relative, _read_stable_file(path)))
    if not entries or Path("predict.py") not in {relative for relative, _ in entries}:
        raise SessionError("submission bundle lacks predict.py")
    if len(entries) > MAX_SUBMISSION_FILES:
        raise SessionError("submission bundle has too many files")
    if sum(len(payload) for _, payload in entries) > MAX_SUBMISSION_TOTAL_BYTES:
        raise SessionError("submission bundle is too large")
    return entries


def stage_submission(source: Path, stage_parent: Path) -> tuple[Path, Path, str]:
    if source.name != "predict.py" or source.is_symlink() or not source.is_file():
        raise SessionError("submission entrypoint must be one real predict.py")
    entries = _read_bundle(source.parent)
    stage_parent.mkdir(parents=True, exist_ok=True)
    os.chmod(stage_parent, 0o711)
    root = Path(tempfile.mkdtemp(prefix="submission-", dir=stage_parent))
    os.chmod(root, 0o755)
    for relative, payload in entries:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(destination.parent, 0o755)
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o444,
        )
        try:
            view = memoryview(payload)
            while view:
                view = view[os.write(descriptor, view) :]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(destination, 0o444)
    return root, root / "predict.py", _hash_entries(entries)


def tree_sha256(root: Path) -> str:
    return _hash_entries(_read_bundle(root))


def processes_for_uid(uid: int) -> list[int]:
    result: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            lines = (entry / "status").read_text().splitlines()
            uid_line = next(line for line in lines if line.startswith("Uid:"))
            real_uid = int(uid_line.split()[1])
        except (OSError, StopIteration, ValueError):
            continue
        if real_uid == uid:
            result.append(int(entry.name))
    return sorted(result)


def kill_uid_processes(uid: int) -> None:
    for pid in processes_for_uid(uid):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass



def preflight_staged_read(staged_submission: Path) -> None:
    """Actually open the staged entrypoint as the fixed solver UID."""
    if os.geteuid() != 0:
        raise InfrastructureError("staged-read preflight requires a root verifier")
    pid = os.fork()
    if pid == 0:
        exit_code = 1
        try:
            os.setgroups([])
            os.setgid(UNTRUSTED_GID)
            os.setuid(UNTRUSTED_UID)
            flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(staged_submission, flags)
            try:
                os.read(descriptor, 1)
            finally:
                os.close(descriptor)
            exit_code = 0
        except BaseException:
            pass
        os._exit(exit_code)
    _, status = os.waitpid(pid, 0)
    if (
        not os.WIFEXITED(status)
        or os.WEXITSTATUS(status) != 0
    ):
        raise InfrastructureError("staged submission is unreadable as UID 61224")

def _setpriv_command(command: list[str]) -> list[str]:
    return [
        str(SETPRIV),
        "--reuid",
        str(UNTRUSTED_UID),
        "--regid",
        str(UNTRUSTED_GID),
        "--clear-groups",
        "--no-new-privs",
        "--bounding-set=-all",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        *command,
    ]


def _base_namespace_command() -> list[str]:
    return [
        str(BWRAP),
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-net",
        "--hostname",
        "ara-routing-predictor",
    ]


@functools.lru_cache(maxsize=1)
def probe_release_support() -> None:
    """Fail closed unless the verifier can create the required namespaces."""

    if sys.platform != "linux" or os.geteuid() != 0:
        raise InfrastructureError("release isolation requires a root Linux verifier")
    for path in (BWRAP, SETPRIV):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise InfrastructureError(f"release isolation helper is unavailable: {path}")
    command = _base_namespace_command()
    command.extend(["--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib"])
    if Path("/lib64").exists():
        command.extend(["--ro-bind", "/lib64", "/lib64"])
    command.extend(
        [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--",
            *_setpriv_command(["/usr/bin/true"]),
        ]
    )
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=5.0,
            check=False,
            env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise InfrastructureError("release namespace probe could not run") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace")[-500:].strip()
        raise InfrastructureError(f"release namespace probe failed: {detail}")


def _release_command(
    staged_submission: Path,
    input_path: Path,
    work: Path,
    private_tmp: Path,
) -> list[str]:
    command = _base_namespace_command()
    command.append("--clearenv")
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "NUMEXPR_NUM_THREADS": "1",
        "NVIDIA_VISIBLE_DEVICES": "void",
    }
    for name, value in environment.items():
        command.extend(["--setenv", name, value])
    command.extend(
        [
            "--ro-bind",
            str(PYTHON_PREFIX),
            str(PYTHON_PREFIX),
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/lib",
            "/lib",
        ]
    )
    if Path("/lib64").exists():
        command.extend(["--ro-bind", "/lib64", "/lib64"])
    command.extend(
        [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--dir",
            "/sandbox",
            "--dir",
            "/output",
            "--dir",
            "/tmp",
            "--ro-bind",
            str(CHILD),
            "/sandbox/child_entry.py",
            "--ro-bind",
            str(staged_submission.parent),
            "/sandbox/submission",
            "--ro-bind",
            str(input_path),
            "/sandbox/public_input.npz",
            "--bind",
            str(work),
            "/output",
            "--bind",
            str(private_tmp),
            "/tmp",
            "--chdir",
            "/sandbox",
            "--",
            *_setpriv_command(
                [
                    str(PYTHON_PREFIX / "bin" / "python"),
                    "-I",
                    "-B",
                    "/sandbox/child_entry.py",
                    "/sandbox/submission/predict.py",
                    "/sandbox/public_input.npz",
                    "/output/prediction.npz",
                ]
            ),
        ]
    )
    return command


def _child_setup(timeout_seconds: float) -> None:
    os.setsid()
    cpu = max(1, int(timeout_seconds) + 2)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_LIMIT_BYTES, FILE_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_NOFILE, (NOFILE_LIMIT, NOFILE_LIMIT))
    resource.setrlimit(resource.RLIMIT_NPROC, (PIDS_LIMIT, PIDS_LIMIT))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        process.wait(timeout=2.0)
    except (subprocess.TimeoutExpired, ChildProcessError):
        pass


def _write_public_input(path: Path, public: Mapping[str, np.ndarray]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=".input-", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **public)
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_output(path: Path, release_mode: bool) -> tuple[np.ndarray, np.ndarray]:
    if path.is_symlink() or not path.is_file():
        raise SessionError("predictor produced no safe output archive")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise SessionError("predictor output must be a single regular file")
    if release_mode and info.st_uid != UNTRUSTED_UID:
        raise SessionError("predictor output has an unexpected owner")
    if not 1 <= info.st_size <= MAX_OUTPUT_BYTES:
        raise SessionError("predictor output size is outside limits")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if tuple(sorted(archive.files)) != ("pred_occluded", "pred_state_token"):
                raise SessionError("predictor output fields are not exact")
            state_token = np.asarray(archive["pred_state_token"]).copy()
            occluded = np.asarray(archive["pred_occluded"]).copy()
    except SessionError:
        raise
    except Exception as exc:
        raise SessionError("predictor output is not a valid safe NPZ") from exc
    return state_token, occluded


def run_case(
    staged_submission: Path,
    public: Mapping[str, np.ndarray],
    *,
    stage_parent: Path,
    timeout_seconds: float,
    release_mode: bool = True,
) -> PredictionResult:
    if release_mode and os.geteuid() != 0:
        raise InfrastructureError("release isolation requires a root verifier parent")
    if release_mode:
        probe_release_support()
        kill_uid_processes(UNTRUSTED_UID)
        if processes_for_uid(UNTRUSTED_UID):
            raise InfrastructureError("could not clear stale untrusted processes")
    root = Path(tempfile.mkdtemp(prefix="case-", dir=stage_parent))
    os.chmod(root, 0o755)
    input_path = root / "public_input.npz"
    _write_public_input(input_path, public)
    work = root / "work"
    work.mkdir(mode=0o700)
    private_tmp = root / "tmp"
    private_tmp.mkdir(mode=0o700)
    if release_mode:
        os.chown(work, UNTRUSTED_UID, UNTRUSTED_GID)
        os.chown(private_tmp, UNTRUSTED_UID, UNTRUSTED_GID)
    output_path = work / "prediction.npz"
    stderr_path = root / "stderr.log"
    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    stderr_tail = ""
    try:
        with stderr_path.open("w+b") as stderr_file:
            environment = {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": str(work),
                "TMPDIR": str(work),
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
            command = [
                sys.executable,
                str(CHILD if release_mode else DEV_CHILD),
                str(staged_submission),
                str(input_path),
                str(output_path),
            ]
            cwd = work
            if release_mode:
                command = _release_command(
                    staged_submission,
                    input_path,
                    work,
                    private_tmp,
                )
                cwd = root
                environment = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin"}
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
                close_fds=True,
                preexec_fn=lambda: _child_setup(timeout_seconds),
            )
            try:
                return_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                raise SessionError("predictor timed out") from exc
            stderr_file.flush()
            stderr_file.seek(0, os.SEEK_END)
            size = stderr_file.tell()
            stderr_file.seek(max(0, size - 2000))
            stderr_tail = stderr_file.read().decode("utf-8", "replace")
            if return_code != 0:
                raise SessionError(f"predictor exited with code {return_code}; stderr_tail={stderr_tail!r}")
        state_token, occluded = _read_output(output_path, release_mode)
        return PredictionResult(
            state_token, occluded, time.monotonic() - started, stderr_tail
        )
    finally:
        if process is not None:
            _kill_group(process)
        if release_mode:
            kill_uid_processes(UNTRUSTED_UID)
            if processes_for_uid(UNTRUSTED_UID):
                raise InfrastructureError("untrusted processes survived case cleanup")
        shutil.rmtree(root, ignore_errors=True)
