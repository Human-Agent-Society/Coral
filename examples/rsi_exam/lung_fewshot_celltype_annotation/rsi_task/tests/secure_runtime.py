#!/usr/bin/env python3
"""Race-resistant staging, privilege separation, and child lifecycle controls."""

from __future__ import annotations

import ctypes
import errno
import glob
import hashlib
import os
import resource
import secrets
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

SOURCE_ROOT = Path("/app/methods/main")
EXPERIMENT_LOG = Path("/app/experiment_log.md")
INPUT_DIR = Path("/tests/inputs")
SEALED_DIR = Path("/tests/sealed")

MAX_SUBMISSION_BYTES = 1_048_576
MAX_SUBMISSION_FILE_BYTES = 1_048_576
MAX_SUBMISSION_FILES = 128
MAX_SUBMISSION_ENTRIES = 256
MAX_SUBMISSION_DEPTH = 8
ALLOWED_SUFFIXES = {".py", ".json", ".toml", ".txt"}
MAX_EXPERIMENT_LOG_BYTES = 256 * 1024
MAX_PREDICTION_BYTES = 10 * 1024 * 1024
MAX_DIAGNOSTIC_BYTES = 4 * 1024 * 1024
DIAGNOSTIC_READ_CHUNK = 64 * 1024
MAX_SCRATCH_BYTES = 4 * 1024 * 1024 * 1024
MAX_SCRATCH_ENTRIES = 20_000
MAX_SCRATCH_DEPTH = 16
CHILD_ADDRESS_SPACE_BYTES = 48 * 1024 * 1024 * 1024
CHILD_FILE_SIZE_BYTES = 1024 * 1024 * 1024
CHILD_PROCESS_LIMIT = 64
CHILD_OPEN_FILE_LIMIT = 256
CHILD_CPU_SECONDS = 5_460
METHOD_HARD_CAP_SEC = 5_430.0
RUN_UID_START = 30_000
RUN_GID_START = 30_000
RUN_IDENTITY_COUNT = 8_192
PR_SET_DUMPABLE = 4
PR_SET_NO_NEW_PRIVS = 38
PR_SET_CHILD_SUBREAPER = 36
IPC_RMID = 0

CHILD_PATH = "/opt/conda/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
CHILD_LD_LIBRARY_PATH = "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/opt/conda/lib"


class SubmissionError(RuntimeError):
    """The submitted source/output violated the task contract."""


class GraderError(RuntimeError):
    """The trusted verifier boundary failed and the run must be retried."""


@dataclass(frozen=True)
class Identity:
    dev: int
    ino: int
    size: int
    mtime_ns: int
    ctime_ns: int
    uid: int
    gid: int
    mode: int
    nlink: int


@dataclass(frozen=True)
class SourceEntry:
    relative: Path
    identity: Identity
    is_directory: bool
    payload: bytes | None


@dataclass(frozen=True)
class SourceBundle:
    root_identity: Identity
    entries: tuple[SourceEntry, ...]


@dataclass(frozen=True)
class RunResult:
    predictions: bytes
    stdout_tail: str
    stderr_tail: str
    run_uid: int
    run_gid: int
    scratch_bytes: int
    scratch_entries: int
    processes_killed: int
    sysv_ipc_removed: int


def identity(info: os.stat_result) -> Identity:
    return Identity(
        dev=info.st_dev,
        ino=info.st_ino,
        size=info.st_size,
        mtime_ns=info.st_mtime_ns,
        ctime_ns=info.st_ctime_ns,
        uid=info.st_uid,
        gid=info.st_gid,
        mode=info.st_mode,
        nlink=info.st_nlink,
    )


def _open_directory(path: Path) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SubmissionError("submission failed: source root is unavailable or unsafe") from exc
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode):
        os.close(descriptor)
        raise SubmissionError("submission failed: source root must be one real directory")
    return descriptor


def _read_regular_at(parent_fd: int, name: str, expected: Identity) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise SubmissionError("submission failed: source file changed while opening") from exc
    try:
        before = os.fstat(descriptor)
        if identity(before) != expected or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise SubmissionError("submission failed: source file changed while opening")
        payload = bytearray()
        while len(payload) <= MAX_SUBMISSION_FILE_BYTES:
            chunk = os.read(descriptor, min(1 << 20, MAX_SUBMISSION_FILE_BYTES + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if len(payload) != expected.size or len(payload) > MAX_SUBMISSION_FILE_BYTES or identity(after) != expected:
            raise SubmissionError("submission failed: source file changed while being staged")
        return bytes(payload)
    finally:
        os.close(descriptor)


def read_source_bundle(root: Path = SOURCE_ROOT) -> SourceBundle:
    root_fd = _open_directory(root)
    try:
        root_identity = identity(os.fstat(root_fd))
        entries: list[SourceEntry] = []
        file_count = 0
        total_bytes = 0

        def visit(directory_fd: int, relative: Path, depth: int) -> None:
            nonlocal file_count, total_bytes
            if depth > MAX_SUBMISSION_DEPTH:
                raise SubmissionError("submission failed: source depth limit exceeded")
            before = identity(os.fstat(directory_fd))
            try:
                children = sorted(os.scandir(directory_fd), key=lambda item: item.name)
            except OSError as exc:
                raise SubmissionError("submission failed: source directory is unreadable") from exc
            for child in children:
                name = child.name
                if not name or name in {".", ".."} or "/" in name or "\x00" in name:
                    raise SubmissionError("submission failed: unsafe source entry name")
                child_relative = relative / name
                if len(child_relative.parts) > MAX_SUBMISSION_DEPTH:
                    raise SubmissionError("submission failed: source depth limit exceeded")
                try:
                    snapshot = identity(child.stat(follow_symlinks=False))
                except OSError as exc:
                    raise SubmissionError("submission failed: source entry changed during enumeration") from exc
                entries.append(SourceEntry(child_relative, snapshot, stat.S_ISDIR(snapshot.mode), None))
                if len(entries) > MAX_SUBMISSION_ENTRIES:
                    raise SubmissionError("submission failed: source entry limit exceeded")
                if stat.S_ISLNK(snapshot.mode):
                    raise SubmissionError("submission failed: source symlinks are forbidden")
                if stat.S_ISDIR(snapshot.mode):
                    if name == "__pycache__":
                        raise SubmissionError("submission failed: generated cache directories are forbidden")
                    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
                    try:
                        child_fd = os.open(name, flags, dir_fd=directory_fd)
                    except OSError as exc:
                        raise SubmissionError("submission failed: source directory changed while opening") from exc
                    try:
                        if identity(os.fstat(child_fd)) != snapshot:
                            raise SubmissionError("submission failed: source directory changed while opening")
                        visit(child_fd, child_relative, depth + 1)
                        if identity(os.fstat(child_fd)) != snapshot:
                            raise SubmissionError("submission failed: source directory changed during staging")
                    finally:
                        os.close(child_fd)
                elif stat.S_ISREG(snapshot.mode):
                    if snapshot.nlink != 1:
                        raise SubmissionError("submission failed: source hardlinks are forbidden")
                    suffix = Path(name).suffix.lower()
                    if suffix in {".pyc", ".pyo"}:
                        raise SubmissionError("submission failed: generated bytecode is forbidden")
                    if suffix not in ALLOWED_SUFFIXES:
                        raise SubmissionError(f"submission failed: source type is not allowed: {child_relative}")
                    if snapshot.size > MAX_SUBMISSION_FILE_BYTES:
                        raise SubmissionError("submission failed: source file exceeds its byte limit")
                    payload = _read_regular_at(directory_fd, name, snapshot)
                    file_count += 1
                    total_bytes += len(payload)
                    if file_count > MAX_SUBMISSION_FILES:
                        raise SubmissionError("submission failed: source file limit exceeded")
                    if total_bytes > MAX_SUBMISSION_BYTES:
                        raise SubmissionError("submission failed: source package exceeds one MiB")
                    entries[-1] = SourceEntry(child_relative, snapshot, False, payload)
                else:
                    raise SubmissionError("submission failed: source special files are forbidden")
            if identity(os.fstat(directory_fd)) != before:
                raise SubmissionError("submission failed: source directory changed during staging")

        visit(root_fd, Path(), 0)
        if identity(os.fstat(root_fd)) != root_identity:
            raise SubmissionError("submission failed: source root changed during staging")
    finally:
        os.close(root_fd)
    files = {entry.relative for entry in entries if not entry.is_directory}
    if Path("solver.py") not in files:
        raise SubmissionError("submission failed: methods/main/solver.py is required")
    return SourceBundle(root_identity=root_identity, entries=tuple(entries))


def _children(bundle: SourceBundle) -> dict[Path, list[SourceEntry]]:
    result: dict[Path, list[SourceEntry]] = {}
    for entry in bundle.entries:
        result.setdefault(entry.relative.parent, []).append(entry)
    for values in result.values():
        values.sort(key=lambda item: item.relative.name)
    return result


def seal_original_tree(root: Path, bundle: SourceBundle) -> tuple[list[Path], list[Path]]:
    """Seal the exact enumerated source tree root-owned/read-only using directory fds."""
    root_fd = _open_directory(root)
    children = _children(bundle)
    directories: list[Path] = []
    files: list[Path] = []
    try:
        if identity(os.fstat(root_fd)) != bundle.root_identity:
            raise SubmissionError("submission failed: source root changed before sealing")

        def seal_directory(directory_fd: int, relative: Path, expected: Identity) -> None:
            if identity(os.fstat(directory_fd)) != expected:
                raise SubmissionError("submission failed: source directory changed before sealing")
            try:
                os.fchown(directory_fd, 0, 0)
                os.fchmod(directory_fd, 0o555)
            except OSError as exc:
                raise GraderError("grader failed: could not seal source directory") from exc
            directory_path = root if not relative.parts else root / relative
            directories.append(directory_path)
            try:
                actual_names = {entry.name for entry in os.scandir(directory_fd)}
            except OSError as exc:
                raise GraderError("grader failed: could not enumerate sealed source directory") from exc
            expected_entries = children.get(relative, [])
            if actual_names != {entry.relative.name for entry in expected_entries}:
                raise SubmissionError("submission failed: source tree changed before sealing")
            for entry in expected_entries:
                name = entry.relative.name
                if entry.is_directory:
                    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
                    try:
                        child_fd = os.open(name, flags, dir_fd=directory_fd)
                    except OSError as exc:
                        raise SubmissionError("submission failed: source directory changed before sealing") from exc
                    try:
                        seal_directory(child_fd, entry.relative, entry.identity)
                    finally:
                        os.close(child_fd)
                else:
                    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
                    try:
                        child_fd = os.open(name, flags, dir_fd=directory_fd)
                    except OSError as exc:
                        raise SubmissionError("submission failed: source file changed before sealing") from exc
                    try:
                        if identity(os.fstat(child_fd)) != entry.identity:
                            raise SubmissionError("submission failed: source file changed before sealing")
                        os.fchown(child_fd, 0, 0)
                        os.fchmod(child_fd, 0o444)
                    except OSError as exc:
                        raise GraderError("grader failed: could not seal source file") from exc
                    finally:
                        os.close(child_fd)
                    files.append(root / entry.relative)

        seal_directory(root_fd, Path(), bundle.root_identity)
    finally:
        os.close(root_fd)
    return directories, files


def write_staged_tree(bundle: SourceBundle, parent: Path) -> tuple[Path, list[Path], list[Path]]:
    root = Path(tempfile.mkdtemp(prefix="lung-submission-", dir=parent))
    directories = [root]
    files: list[Path] = []
    try:
        for entry in sorted((item for item in bundle.entries if item.is_directory), key=lambda item: len(item.relative.parts)):
            target = root / entry.relative
            target.mkdir(mode=0o700)
            directories.append(target)
        for entry in (item for item in bundle.entries if not item.is_directory):
            assert entry.payload is not None
            target = root / entry.relative
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o400)
            try:
                view = memoryview(entry.payload)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise GraderError("grader failed: short write while staging source")
                    view = view[written:]
                os.fsync(descriptor)
                os.fchown(descriptor, 0, 0)
                os.fchmod(descriptor, 0o444)
            finally:
                os.close(descriptor)
            files.append(target)
        for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
            os.chown(directory, 0, 0, follow_symlinks=False)
            os.chmod(directory, 0o555)
    except BaseException:
        _remove_tree(root)
        raise
    return root, directories, files


def _drop_identity(uid: int, gid: int) -> None:
    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)


def assert_child_readable(directories: Sequence[Path], files: Sequence[Path], uid: int, gid: int) -> None:
    pid = os.fork()
    if pid == 0:
        try:
            _drop_identity(uid, gid)
            for directory in directories:
                descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
                os.close(descriptor)
            for path in files:
                descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
                try:
                    opened = os.fstat(descriptor)
                    if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                        os._exit(3)
                    os.read(descriptor, 1)
                finally:
                    os.close(descriptor)
            os._exit(0)
        except BaseException:
            os._exit(1)
    waited, status_code = os.waitpid(pid, 0)
    if waited != pid or not os.WIFEXITED(status_code) or os.WEXITSTATUS(status_code) != 0:
        raise GraderError("grader failed: child identity cannot read every required source/input file")


def assert_child_write_denied(directories: Sequence[Path], files: Sequence[Path], uid: int, gid: int) -> None:
    probe_name = f".lung-write-probe-{os.getpid()}-{uid}"
    probes = [directory / probe_name for directory in directories]
    if any(path.exists() for path in probes):
        raise GraderError("grader failed: write-denial probe path already exists")
    pid = os.fork()
    if pid == 0:
        try:
            _drop_identity(uid, gid)
            for probe in probes:
                try:
                    descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EPERM, errno.EROFS):
                        os._exit(3)
                else:
                    os.close(descriptor)
                    os._exit(2)
            for path in files:
                try:
                    descriptor = os.open(path, os.O_WRONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EPERM, errno.EROFS):
                        os._exit(5)
                else:
                    os.close(descriptor)
                    os._exit(4)
            os._exit(0)
        except BaseException:
            os._exit(1)
    waited, status_code = os.waitpid(pid, 0)
    planted = False
    for probe in probes:
        try:
            probe.unlink()
            planted = True
        except FileNotFoundError:
            pass
        except OSError:
            planted = True
    if planted or waited != pid or not os.WIFEXITED(status_code) or os.WEXITSTATUS(status_code) != 0:
        raise GraderError("grader failed: source tree remains writable by child identity")


def validate_child_inputs(paths: Sequence[Path], uid: int, gid: int) -> None:
    for path in paths:
        try:
            info = path.lstat()
        except OSError as exc:
            raise GraderError("grader failed: child input is missing") from exc
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_uid != 0
            or info.st_gid != 0
            or stat.S_IMODE(info.st_mode) != 0o444
            or info.st_nlink != 1
        ):
            raise GraderError("grader failed: child input metadata is unsafe")
    assert_child_readable([INPUT_DIR], paths, uid, gid)
    assert_child_write_denied([INPUT_DIR], paths, uid, gid)


def assert_sealed_denied(paths: Sequence[Path], uid: int, gid: int) -> None:
    pid = os.fork()
    if pid == 0:
        try:
            _drop_identity(uid, gid)
            if glob.glob(str(SEALED_DIR / "*")):
                os._exit(2)
            try:
                with os.scandir(SEALED_DIR) as iterator:
                    next(iterator, None)
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EPERM):
                    os._exit(3)
            else:
                os._exit(4)
            for path in paths:
                try:
                    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EPERM):
                        os._exit(5)
                else:
                    os.close(descriptor)
                    os._exit(6)
            os._exit(0)
        except BaseException:
            os._exit(1)
    waited, status_code = os.waitpid(pid, 0)
    if waited != pid or not os.WIFEXITED(status_code) or os.WEXITSTATUS(status_code) != 0:
        raise GraderError("grader failed: sealed scorer assets are visible to child identity")


def _stable_root_bytes(path: Path, max_bytes: int, expected_sha256: str) -> bytes:
    try:
        info = path.lstat()
    except OSError as exc:
        raise GraderError("grader failed: sealed scorer asset is missing") from exc
    expected = identity(info)
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != 0
        or info.st_gid != 0
        or stat.S_IMODE(info.st_mode) != 0o400
        or info.st_nlink != 1
        or not 0 < info.st_size <= max_bytes
    ):
        raise GraderError("grader failed: sealed scorer asset metadata is unsafe")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise GraderError("grader failed: sealed scorer asset could not be opened") from exc
    try:
        before = os.fstat(descriptor)
        if identity(before) != expected:
            raise GraderError("grader failed: sealed scorer asset changed while opening")
        payload = bytearray()
        while len(payload) <= max_bytes:
            chunk = os.read(descriptor, min(1 << 20, max_bytes + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if len(payload) != expected.size or len(payload) > max_bytes or identity(after) != expected:
            raise GraderError("grader failed: sealed scorer asset changed while reading")
    finally:
        os.close(descriptor)
    value = bytes(payload)
    if hashlib.sha256(value).hexdigest() != expected_sha256:
        raise GraderError("grader failed: sealed scorer asset integrity mismatch")
    return value


def capture_and_unlink_sealed(specifications: Mapping[Path, tuple[int, str]]) -> dict[Path, bytes]:
    payloads = {path: _stable_root_bytes(path, max_bytes, digest) for path, (max_bytes, digest) in specifications.items()}
    for path in specifications:
        try:
            path.unlink()
        except OSError as exc:
            raise GraderError("grader failed: sealed scorer asset could not be removed") from exc
    if any(path.exists() for path in specifications):
        raise GraderError("grader failed: sealed scorer asset remained after removal")
    return payloads


def make_parent_nondumpable_subreaper() -> None:
    if os.geteuid() != 0:
        raise GraderError("grader failed: verifier parent must run as root")
    libc = ctypes.CDLL(None, use_errno=True)
    for option, value, name in (
        (PR_SET_DUMPABLE, 0, "PR_SET_DUMPABLE"),
        (PR_SET_CHILD_SUBREAPER, 1, "PR_SET_CHILD_SUBREAPER"),
    ):
        if libc.prctl(option, value, 0, 0, 0) != 0:
            error = ctypes.get_errno()
            raise GraderError(f"grader failed: {name} failed: errno {error}")


def remove_experiment_log() -> None:
    try:
        info = EXPERIMENT_LOG.lstat()
    except FileNotFoundError:
        return
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or info.st_size > MAX_EXPERIMENT_LOG_BYTES
    ):
        raise SubmissionError("submission failed: experiment log is unsafe or too large")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(EXPERIMENT_LOG, flags)
    try:
        if identity(os.fstat(descriptor)) != identity(info):
            raise SubmissionError("submission failed: experiment log changed while opening")
    finally:
        os.close(descriptor)
    try:
        EXPERIMENT_LOG.unlink()
    except OSError as exc:
        raise GraderError("grader failed: could not remove runtime-unneeded experiment log") from exc


def seal_external_write_surfaces() -> None:
    required = (
        (Path("/app"), 0o555),
        (Path("/logs"), 0o755),
        (Path("/tmp"), 0o755),
        (Path("/var/tmp"), 0o755),
        (Path("/dev/shm"), 0o555),
    )
    optional = ((Path("/dev/mqueue"), 0o555), (Path("/run/lock"), 0o755))
    for path, mode in required + optional:
        try:
            info = path.lstat()
        except FileNotFoundError:
            if (path, mode) in required:
                raise GraderError(f"grader failed: required write surface is missing: {path}")
            continue
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise GraderError(f"grader failed: write surface is unsafe: {path}")
        try:
            os.chown(path, 0, 0, follow_symlinks=False)
            os.chmod(path, mode)
        except OSError as exc:
            raise GraderError(f"grader failed: could not seal write surface: {path}") from exc


def _uid_pids(uid: int) -> list[int]:
    result: list[int] = []
    try:
        entries = list(os.scandir("/proc"))
    except OSError as exc:
        raise GraderError("grader failed: could not inspect process table") from exc
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            with open(f"/proc/{entry.name}/status", encoding="ascii") as handle:
                for line in handle:
                    if line.startswith("Uid:"):
                        values = [int(item) for item in line.split()[1:5]]
                        if uid in values:
                            result.append(int(entry.name))
                        break
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return result


def _uid_ipc_objects(uid: int) -> list[tuple[str, int]]:
    objects: list[tuple[str, int]] = []
    specifications = (
        ("shm", Path("/proc/sysvipc/shm"), "shmid"),
        ("msg", Path("/proc/sysvipc/msg"), "msqid"),
        ("sem", Path("/proc/sysvipc/sem"), "semid"),
    )
    for kind, path, identifier_name in specifications:
        try:
            lines = path.read_text(encoding="ascii").splitlines()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise GraderError("grader failed: could not inspect SysV IPC") from exc
        if not lines:
            continue
        header = lines[0].split()
        try:
            id_index = header.index(identifier_name)
            owner_indices = (header.index("uid"), header.index("cuid"))
        except ValueError as exc:
            raise GraderError("grader failed: unexpected SysV IPC table schema") from exc
        for line in lines[1:]:
            fields = line.split()
            try:
                owners = [int(fields[index]) for index in owner_indices]
                identifier = int(fields[id_index])
            except (IndexError, ValueError) as exc:
                raise GraderError("grader failed: malformed SysV IPC table") from exc
            if uid in owners:
                objects.append((kind, identifier))
    return objects


def allocate_run_identity() -> tuple[int, int]:
    for _ in range(256):
        offset = secrets.randbelow(RUN_IDENTITY_COUNT)
        uid = RUN_UID_START + offset
        gid = RUN_GID_START + offset
        if not _uid_pids(uid) and not _uid_ipc_objects(uid):
            return uid, gid
    raise GraderError("grader failed: could not allocate a fresh child identity")


def _cleanup_ipc(uid: int, gid: int) -> int:
    removed = 0
    for _ in range(3):
        objects = _uid_ipc_objects(uid)
        if not objects:
            return removed
        pid = os.fork()
        if pid == 0:
            try:
                _drop_identity(uid, gid)
                libc = ctypes.CDLL(None, use_errno=True)
                for kind, identifier in objects:
                    ctypes.set_errno(0)
                    if kind == "shm":
                        result = libc.shmctl(identifier, IPC_RMID, None)
                    elif kind == "msg":
                        result = libc.msgctl(identifier, IPC_RMID, None)
                    else:
                        result = libc.semctl(identifier, 0, IPC_RMID)
                    error = ctypes.get_errno()
                    if result != 0 and error not in (errno.EIDRM, errno.EINVAL):
                        os._exit(2)
                os._exit(0)
            except BaseException:
                os._exit(1)
        waited, status_code = os.waitpid(pid, 0)
        if waited != pid or not os.WIFEXITED(status_code) or os.WEXITSTATUS(status_code) != 0:
            raise GraderError("grader failed: SysV IPC cleanup helper failed")
        removed += len(objects)
        time.sleep(0.01)
    if _uid_ipc_objects(uid):
        raise GraderError("grader failed: SysV IPC object survived cleanup")
    return removed


def _reap_children() -> None:
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def _terminate_and_reap(process: subprocess.Popen[bytes], uid: int, gid: int) -> tuple[int, int]:
    signaled: set[int] = set()
    for sig, delay in ((signal.SIGTERM, 0.1), (signal.SIGKILL, 0.0)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        for pid in _uid_pids(uid):
            try:
                os.kill(pid, sig)
                signaled.add(pid)
            except ProcessLookupError:
                pass
        if delay:
            time.sleep(delay)
    if process.poll() is None:
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired as exc:
            raise GraderError("grader failed: direct child survived SIGKILL") from exc
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        _reap_children()
        remaining = _uid_pids(uid)
        if not remaining:
            break
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
                signaled.add(pid)
            except ProcessLookupError:
                pass
        time.sleep(0.01)
    _reap_children()
    if _uid_pids(uid):
        raise GraderError("grader failed: detached child process survived cleanup")
    return len(signaled), _cleanup_ipc(uid, gid)


def _child_preexec(uid: int, gid: int) -> None:
    resource.setrlimit(resource.RLIMIT_AS, (CHILD_ADDRESS_SPACE_BYTES, CHILD_ADDRESS_SPACE_BYTES))
    resource.setrlimit(resource.RLIMIT_NPROC, (CHILD_PROCESS_LIMIT, CHILD_PROCESS_LIMIT))
    resource.setrlimit(resource.RLIMIT_NOFILE, (CHILD_OPEN_FILE_LIMIT, CHILD_OPEN_FILE_LIMIT))
    resource.setrlimit(resource.RLIMIT_FSIZE, (CHILD_FILE_SIZE_BYTES, CHILD_FILE_SIZE_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (CHILD_CPU_SECONDS, CHILD_CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        os._exit(126)
    os.umask(0o077)
    _drop_identity(uid, gid)


def audit_scratch(root: Path, uid: int, gid: int) -> tuple[int, int]:
    entries = 0
    total = 0
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        if depth > MAX_SCRATCH_DEPTH:
            raise SubmissionError("submission failed: scratch depth limit exceeded")
        try:
            children = list(os.scandir(directory))
        except OSError as exc:
            raise SubmissionError("submission failed: scratch directory is unreadable") from exc
        for child in children:
            try:
                info = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise SubmissionError("submission failed: scratch entry changed during audit") from exc
            entries += 1
            if entries > MAX_SCRATCH_ENTRIES:
                raise SubmissionError("submission failed: scratch entry limit exceeded")
            if info.st_uid != uid or info.st_gid != gid:
                raise SubmissionError("submission failed: scratch entry has unexpected ownership")
            if stat.S_ISLNK(info.st_mode):
                raise SubmissionError("submission failed: scratch symlinks are forbidden")
            if stat.S_ISDIR(info.st_mode):
                stack.append((Path(child.path), depth + 1))
            elif stat.S_ISREG(info.st_mode):
                if info.st_nlink != 1:
                    raise SubmissionError("submission failed: scratch hardlinks are forbidden")
                total += info.st_size
                if total > MAX_SCRATCH_BYTES:
                    raise SubmissionError("submission failed: scratch byte limit exceeded")
            else:
                raise SubmissionError("submission failed: scratch special files are forbidden")
    return entries, total


def _stable_child_file(path: Path, uid: int, gid: int, max_bytes: int) -> bytes:
    try:
        info = path.lstat()
    except OSError as exc:
        raise SubmissionError("submission failed: solver produced no predictions") from exc
    expected = identity(info)
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != uid
        or info.st_gid != gid
        or info.st_nlink != 1
        or not 0 < info.st_size <= max_bytes
    ):
        raise SubmissionError("submission failed: predictions metadata is unsafe")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SubmissionError("submission failed: predictions could not be opened") from exc
    try:
        before = os.fstat(descriptor)
        if identity(before) != expected:
            raise SubmissionError("submission failed: predictions changed while opening")
        payload = bytearray()
        while len(payload) <= max_bytes:
            chunk = os.read(descriptor, min(1 << 20, max_bytes + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        after = os.fstat(descriptor)
        if len(payload) != expected.size or len(payload) > max_bytes or identity(after) != expected:
            raise SubmissionError("submission failed: predictions changed while reading")
    finally:
        os.close(descriptor)
    return bytes(payload)


def _extend_diagnostic(buffer: bytearray, chunk: bytes, stream: str) -> None:
    if len(buffer) + len(chunk) > MAX_DIAGNOSTIC_BYTES:
        raise SubmissionError(f"submission failed: {stream} exceeded its active byte limit")
    buffer.extend(chunk)


def _drain_ready_diagnostics(
    selector: selectors.BaseSelector,
    streams: dict[int, tuple[str, bytearray]],
    timeout: float,
) -> None:
    try:
        events = selector.select(timeout)
    except OSError as exc:
        raise GraderError("grader failed: could not poll submission diagnostics") from exc
    for key, _ in events:
        descriptor = int(key.fd)
        stream, buffer = streams[descriptor]
        while True:
            try:
                chunk = os.read(descriptor, DIAGNOSTIC_READ_CHUNK)
            except BlockingIOError:
                break
            except OSError as exc:
                raise GraderError("grader failed: could not read submission diagnostics") from exc
            if not chunk:
                try:
                    selector.unregister(descriptor)
                except KeyError:
                    pass
                break
            _extend_diagnostic(buffer, chunk, stream)


def _diagnostic_tail(payload: bytes) -> str:
    if len(payload) > MAX_DIAGNOSTIC_BYTES:
        raise GraderError("grader failed: captured diagnostics exceeded their trusted byte limit")
    return payload[-4_000:].decode("utf-8", errors="replace")


def run_submission(solver: Path, workspace: Path, uid: int, gid: int) -> RunResult:
    scratch = workspace / "scratch"
    scratch.mkdir(mode=0o700)
    os.chown(scratch, uid, gid)
    predictions = scratch / "predictions.csv"
    command = [
        sys.executable,
        str(solver),
        "--labeled", str(INPUT_DIR / "labeled.h5ad"),
        "--unlabeled", str(INPUT_DIR / "unlabeled.h5ad"),
        "--query", str(INPUT_DIR / "query.h5ad"),
        "--classes", str(INPUT_DIR / "classes.txt"),
        "--output", str(predictions),
    ]
    env = {
        "PATH": CHILD_PATH,
        "HOME": str(scratch),
        "TMPDIR": str(scratch),
        "XDG_CACHE_HOME": str(scratch / ".cache"),
        "MPLCONFIGDIR": str(scratch / ".matplotlib"),
        "NUMBA_CACHE_DIR": str(scratch / ".numba"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "NVIDIA_VISIBLE_DEVICES": "all",
        "LD_LIBRARY_PATH": CHILD_LD_LIBRARY_PATH,
        "OMP_NUM_THREADS": "1",
        "OMP_THREAD_LIMIT": "1",
        "NUMBA_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    if _uid_pids(uid) or _uid_ipc_objects(uid):
        raise GraderError("grader failed: fresh child identity was already in use")

    stdout_read = -1
    stdout_write = -1
    stderr_read = -1
    stderr_write = -1
    process: subprocess.Popen[bytes] | None = None
    selector: selectors.BaseSelector | None = None
    streams: dict[int, tuple[str, bytearray]] = {}
    cleanup_done = False
    try:
        try:
            stdout_read, stdout_write = os.pipe2(os.O_CLOEXEC)
            stderr_read, stderr_write = os.pipe2(os.O_CLOEXEC)
            selector = selectors.DefaultSelector()
        except Exception as exc:
            raise GraderError("grader failed: could not initialize submission diagnostics") from exc
        streams = {
            stdout_read: ("stdout", bytearray()),
            stderr_read: ("stderr", bytearray()),
        }
        for descriptor in streams:
            os.set_blocking(descriptor, False)
            selector.register(descriptor, selectors.EVENT_READ)
        try:
            process = subprocess.Popen(
                command,
                cwd=scratch,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout_write,
                stderr=stderr_write,
                close_fds=True,
                start_new_session=True,
                preexec_fn=lambda: _child_preexec(uid, gid),
            )
        except Exception as exc:
            raise GraderError("grader failed: could not launch isolated submission") from exc
        finally:
            os.close(stdout_write)
            stdout_write = -1
            os.close(stderr_write)
            stderr_write = -1

        deadline = time.monotonic() + METHOD_HARD_CAP_SEC
        timed_out = False
        while True:
            remaining = deadline - time.monotonic()
            _drain_ready_diagnostics(selector, streams, min(0.05, max(0.0, remaining)))
            if process.poll() is not None:
                break
            if remaining <= 0.0:
                timed_out = True
                break

        return_code = process.returncode
        processes_killed, ipc_removed = _terminate_and_reap(process, uid, gid)
        cleanup_done = True

        # Descendants are gone and the parent write ends were closed immediately
        # after launch, so the remaining pipe bytes are finite and reach EOF.
        drain_deadline = time.monotonic() + 1.0
        while selector.get_map():
            _drain_ready_diagnostics(selector, streams, 0.05)
            if time.monotonic() > drain_deadline and selector.get_map():
                raise GraderError("grader failed: diagnostic pipes did not close after child cleanup")

        if timed_out:
            raise SubmissionError("submission failed: submitted method timed out")
        if return_code != 0:
            stderr = _diagnostic_tail(bytes(streams[stderr_read][1]))
            raise SubmissionError(f"submission failed: submitted method exited {return_code}: {stderr[-1000:]}")
        scratch_entries, scratch_bytes = audit_scratch(scratch, uid, gid)
        payload = _stable_child_file(predictions, uid, gid, MAX_PREDICTION_BYTES)
        return RunResult(
            predictions=payload,
            stdout_tail=_diagnostic_tail(bytes(streams[stdout_read][1])),
            stderr_tail=_diagnostic_tail(bytes(streams[stderr_read][1])),
            run_uid=uid,
            run_gid=gid,
            scratch_bytes=scratch_bytes,
            scratch_entries=scratch_entries,
            processes_killed=processes_killed,
            sysv_ipc_removed=ipc_removed,
        )
    finally:
        if process is not None and not cleanup_done and (process.poll() is None or _uid_pids(uid) or _uid_ipc_objects(uid)):
            _terminate_and_reap(process, uid, gid)
        if selector is not None:
            selector.close()
        for descriptor in (stdout_read, stderr_read, stdout_write, stderr_write):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

def _remove_tree(root: Path) -> None:
    """Best-effort fd-relative cleanup that never follows candidate-created links."""
    directory_flags = (
        os.O_RDONLY
        | os.O_CLOEXEC
        | os.O_DIRECTORY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )

    def clear(directory_fd: int) -> None:
        try:
            children = list(os.scandir(directory_fd))
        except OSError:
            return
        for child in children:
            name = child.name
            try:
                info = child.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISDIR(info.st_mode):
                try:
                    child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                except OSError:
                    continue
                try:
                    opened = os.fstat(child_fd)
                    if opened.st_dev != info.st_dev or opened.st_ino != info.st_ino:
                        continue
                    try:
                        os.fchmod(child_fd, 0o700)
                    except OSError:
                        pass
                    clear(child_fd)
                finally:
                    os.close(child_fd)
                try:
                    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if stat.S_ISDIR(current.st_mode) and current.st_dev == info.st_dev and current.st_ino == info.st_ino:
                        os.rmdir(name, dir_fd=directory_fd)
                except OSError:
                    pass
            else:
                # Regular files, symlinks, FIFOs, sockets, and devices are unlinked by
                # parent-directory descriptor. In particular, never chmod a symlink path.
                try:
                    os.unlink(name, dir_fd=directory_fd)
                except OSError:
                    pass

    try:
        root_info = root.lstat()
    except OSError:
        return
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        try:
            root.unlink()
        except OSError:
            pass
        return
    try:
        root_fd = os.open(root, directory_flags)
    except OSError:
        return
    try:
        opened = os.fstat(root_fd)
        if opened.st_dev != root_info.st_dev or opened.st_ino != root_info.st_ino:
            return
        try:
            os.fchmod(root_fd, 0o700)
        except OSError:
            pass
        clear(root_fd)
    finally:
        os.close(root_fd)
    try:
        current = root.lstat()
        if stat.S_ISDIR(current.st_mode) and current.st_dev == root_info.st_dev and current.st_ino == root_info.st_ino:
            root.rmdir()
    except OSError:
        pass


@contextmanager
def staged_submission(workspace: Path, uid: int, gid: int) -> Iterator[Path]:
    bundle = read_source_bundle(SOURCE_ROOT)
    staged_root, staged_directories, staged_files = write_staged_tree(bundle, workspace)
    source_directories, source_files = seal_original_tree(SOURCE_ROOT, bundle)
    assert_child_readable(source_directories, source_files, uid, gid)
    assert_child_write_denied(source_directories, source_files, uid, gid)
    assert_child_readable(staged_directories, staged_files, uid, gid)
    assert_child_write_denied(staged_directories, staged_files, uid, gid)
    try:
        yield staged_root / "solver.py"
    finally:
        _remove_tree(staged_root)
