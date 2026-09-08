#!/usr/bin/env python3
"""Secure Harbor grader for time-held-out protein-ligand co-folding.

Security boundary
-----------------
The trusted parent owns the held-out manifest and crystal structures and never
imports submitted code.  It stages exactly one anonymized case and that case's
MSA into a fresh scratch, then launches ``child_predict.py`` as an unprivileged
runner under Landlock.  The worker process group is killed and reaped before
the trusted metric parses or scores its output.

Only aggregate results are written.  Hidden target identifiers, per-case
scores, child stderr, and hidden paths never cross the verifier boundary.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import glob
import hashlib
import importlib
import importlib.util
import json
import math
import os
import pwd
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


TESTS_DIR = Path(__file__).resolve().parent
HELDOUT_DIR = Path("/tests/heldout")
SUBMISSION_DIR = Path("/app/methods/main")
VERIFIER_LOG_DIR = Path("/logs/verifier")
ANCHOR_FILE = TESTS_DIR / "anchors.json"
CHILD_SCRIPT = TESTS_DIR / "child_predict.py"
SCORE_SCRIPT = TESTS_DIR / "score_pose_worker.py"
SOURCE_CONTRACT_SCRIPT = TESTS_DIR / "source_contract.py"
METRIC_SCRIPT = TESTS_DIR / "evaluate.py"
RUNNER_ROOT = Path("/tmp/protein_cofold_runner")
RUNNER_USER = "runner"

CASE_TIMEOUT_SEC = 3600.0
SCORE_TIMEOUT_SEC = 300.0
TOTAL_TIMEOUT_SEC = 43200.0
MAX_SUBMISSION_FILES = 32
MAX_SUBMISSION_FILE_BYTES = 128 * 1024
MAX_SUBMISSION_TOTAL_BYTES = 256 * 1024
MAX_SUBMISSION_DEPTH = 4
MAX_MSA_FILE_BYTES = 16 * 1024 * 1024
MAX_MSA_TOTAL_BYTES = 64 * 1024 * 1024
MAX_PREDICTION_JSON_BYTES = 24 * 1024 * 1024
MAX_CASE_CHAINS = 2
MAX_CASE_CHAIN_RESIDUES = 4096
MAX_PREDICTED_LIGAND_ATOMS = 192
MAX_STRUCTURE_COORDINATE_ABS = 100_000.0
SANDBOX_SETUP_EXIT = 125
SOURCE_SUFFIXES = frozenset({".py"})
CHAIN_ID_RE = re.compile(r"^[A-Za-z0-9]{1,8}$")
SEQUENCE_RE = re.compile(r"^[A-Za-z]+$")

# Linux Landlock ABI.  ABI 1 covers base filesystem access, ABI 2 adds REFER,
# and ABI 3 adds TRUNCATE.  Handling the latter is necessary to prevent an
# allowed-readable file outside scratch from being truncated.
LANDLOCK_CREATE_RULESET = 444
LANDLOCK_ADD_RULE = 445
LANDLOCK_RESTRICT_SELF = 446
LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1
LANDLOCK_ACCESS_FS_EXECUTE = 1 << 0
LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
LANDLOCK_ACCESS_FS_READ_FILE = 1 << 2
LANDLOCK_ACCESS_FS_READ_DIR = 1 << 3
LANDLOCK_ACCESS_FS_REMOVE_DIR = 1 << 4
LANDLOCK_ACCESS_FS_REMOVE_FILE = 1 << 5
LANDLOCK_ACCESS_FS_MAKE_CHAR = 1 << 6
LANDLOCK_ACCESS_FS_MAKE_DIR = 1 << 7
LANDLOCK_ACCESS_FS_MAKE_REG = 1 << 8
LANDLOCK_ACCESS_FS_MAKE_SOCK = 1 << 9
LANDLOCK_ACCESS_FS_MAKE_FIFO = 1 << 10
LANDLOCK_ACCESS_FS_MAKE_BLOCK = 1 << 11
LANDLOCK_ACCESS_FS_MAKE_SYM = 1 << 12
LANDLOCK_ACCESS_FS_REFER = 1 << 13
LANDLOCK_ACCESS_FS_TRUNCATE = 1 << 14
LANDLOCK_READ_ACCESS = LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR
LANDLOCK_BASE_WRITE_ACCESS = (
    LANDLOCK_ACCESS_FS_WRITE_FILE
    | LANDLOCK_ACCESS_FS_REMOVE_DIR
    | LANDLOCK_ACCESS_FS_REMOVE_FILE
    | LANDLOCK_ACCESS_FS_MAKE_CHAR
    | LANDLOCK_ACCESS_FS_MAKE_DIR
    | LANDLOCK_ACCESS_FS_MAKE_REG
    | LANDLOCK_ACCESS_FS_MAKE_SOCK
    | LANDLOCK_ACCESS_FS_MAKE_FIFO
    | LANDLOCK_ACCESS_FS_MAKE_BLOCK
    | LANDLOCK_ACCESS_FS_MAKE_SYM
)
PR_SET_NO_NEW_PRIVS = 38
PR_SET_CHILD_SUBREAPER = 36
PR_SET_DUMPABLE = 4
O_PATH = getattr(os, "O_PATH", 0)
CLONE_NEWNS = 0x00020000
MS_NOSUID = 1 << 1
MS_NODEV = 1 << 2
MS_NOEXEC = 1 << 3
MS_REC = 1 << 14
MS_PRIVATE = 1 << 18
PRIVATE_SHM_SIZE = "8g"

# Runtime trees are readable but never writable by the submission.  In
# particular, /tests (except CHILD_SCRIPT), /app, /logs, /root and /home are
# deliberately absent.  Model weights and caches in the verifier image must be
# baked under /opt (for example /opt/boltz-cache).
RUNTIME_READ_ROOTS = (
    Path("/opt"),
    Path("/usr"),
    Path("/lib"),
    Path("/lib64"),
    Path("/etc"),
    Path("/dev"),
    Path("/proc"),
    Path("/sys"),
    Path("/run"),
)
GPU_DEVICE_GLOBS = (
    "/dev/nvidia*",
    "/dev/nvidia-caps/*",
    "/dev/dri/renderD*",
    "/dev/dri/card*",
)
SAFE_WRITABLE_DEVICE_PATHS = (Path("/dev/null"),)
PERSISTENT_IPC_SYSCALLS = (
    # System V shared memory, message queues, and semaphore sets are not
    # namespaced by Landlock and would otherwise persist across case scratches.
    "shmget",
    "shmat",
    "shmdt",
    "shmctl",
    "msgget",
    "msgsnd",
    "msgrcv",
    "msgctl",
    "semget",
    "semop",
    "semtimedop",
    "semctl",
    "ipc",
    # Linux keyrings are another same-UID cross-process persistence channel.
    "add_key",
    "request_key",
    "keyctl",
    # The worker already starts in a fresh session.  Prevent descendants from
    # escaping its process group so killpg is a race-free primary teardown.
    "setsid",
    "setpgid",
)
# The trusted launcher needs a mount namespace only while it is still root.
# Once the private /dev/shm exists and privileges have been permanently
# dropped, submitted code must not be able to create/join namespaces or alter
# the mount view.  In particular, an unprivileged user namespace can confer
# namespaced CAP_SYS_ADMIN even though the launcher verified an empty
# capability set before importing the submission.
NAMESPACE_MOUNT_SYSCALLS = (
    "unshare",
    "setns",
    "mount",
    "umount2",
    "pivot_root",
    "chroot",
    "move_mount",
    "open_tree",
    "fsopen",
    "fsconfig",
    "fsmount",
    "fspick",
    "mount_setattr",
    # Added to the upstream Linux syscall table after the pinned Ubuntu
    # userland/libseccomp release.  It combines open_tree with mount-attribute
    # changes, so a default-allow filter must deny it even on a newer host
    # kernel whose userspace does not know the symbolic name yet.
    "open_tree_attr",
)
# This task's pinned CUDA image is linux/amd64.  libseccomp 2.5.x predates
# open_tree_attr, but numeric seccomp rules are intentionally forward-safe:
# they return EPERM on a host that implements the syscall and also prevent a
# future host-kernel upgrade from silently widening the sandbox.
AMD64_FUTURE_SYSCALL_NUMBERS = {"open_tree_attr": 467}
# clone3 stores flags behind a userspace pointer, so classic seccomp cannot
# safely inspect them.  Deny it and let libc fall back to clone(2).  For
# clone(2), deny only namespace-creating flags so normal threads and worker
# processes remain available to the inference libraries.
CLONE_NAMESPACE_FLAGS = (
    0x00000080,  # CLONE_NEWTIME
    0x00020000,  # CLONE_NEWNS
    0x02000000,  # CLONE_NEWCGROUP
    0x04000000,  # CLONE_NEWUTS
    0x08000000,  # CLONE_NEWIPC
    0x10000000,  # CLONE_NEWUSER
    0x20000000,  # CLONE_NEWPID
    0x40000000,  # CLONE_NEWNET
)
SCMP_ACT_ALLOW = 0x7FFF0000
SCMP_ACT_ERRNO_BASE = 0x00050000
SCMP_CMP_MASKED_EQ = 7


def _resolve_seccomp_syscall_number(seccomp, syscall_name: str) -> int:
    """Resolve a syscall, with an audited fallback for pinned linux/amd64."""
    number = seccomp.seccomp_syscall_resolve_name(syscall_name.encode("ascii"))
    if number >= 0:
        return int(number)
    if os.uname().machine == "x86_64":
        return AMD64_FUTURE_SYSCALL_NUMBERS.get(syscall_name, -1)
    return -1


class ConfigurationError(RuntimeError):
    """The verifier image or injected configuration is invalid."""


class DatasetError(RuntimeError):
    """The trusted held-out package is missing or malformed."""


class SubmissionError(RuntimeError):
    """The submitted source tree violates the source-only contract."""


class IsolationError(RuntimeError):
    """The untrusted runner could not be securely isolated."""


class PredictionError(RuntimeError):
    """The child failed or emitted an invalid result."""


class WorkerTimeout(PredictionError):
    """A child exceeded its declared runtime budget."""


@dataclass(frozen=True)
class Anchors:
    baseline: float
    upper_bound: float


@dataclass(frozen=True)
class CaseAssets:
    child_item: Mapping[str, Any]
    expected_chains: Sequence[Mapping[str, str]]
    msa_source_dir: Path
    crystal_ligand: Path
    crystal_protein: Path


@dataclass(frozen=True)
class CaseScratch:
    root: Path
    solver: Path
    item_json: Path
    prediction_json: Path
    work_dir: Path


@dataclass(frozen=True)
class CaseOutcome:
    passed: bool
    pb_valid: bool
    rmsd_within_2a: bool


class _LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _LandlockPathBeneathAttr(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class _ScmpArgCmp(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_uint),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]


def load_anchors() -> Anchors:
    info = ANCHOR_FILE.lstat()
    if ANCHOR_FILE.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ConfigurationError("unsafe root-only anchor file")
    if ANCHOR_FILE == Path("/tests/anchors.json") and (
        (info.st_uid, info.st_gid) != (0, 0) or stat.S_IMODE(info.st_mode) != 0o400
    ):
        raise ConfigurationError("anchor file must be root-owned mode 0400")
    payload = json.loads(ANCHOR_FILE.read_text(encoding="utf-8"))
    if set(payload) != {"BASELINE", "UPPER_BOUND"}:
        raise ConfigurationError("unexpected anchor payload")
    anchors = Anchors(
        baseline=float(payload["BASELINE"]),
        upper_bound=float(payload["UPPER_BOUND"]),
    )
    if not (0.0 <= anchors.baseline < anchors.upper_bound <= 1.0):
        raise ConfigurationError(
            "anchors must satisfy 0 <= BASELINE < UPPER_BOUND <= 1"
        )
    return anchors


def normalized_reward(metric: float, anchors: Anchors) -> float:
    """Linearly map baseline/upper to 0/1 and clamp to [0, 1]."""
    value = float(metric)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("success-rate metric must be finite and in [0, 1]")
    reward = (value - anchors.baseline) / (
        anchors.upper_bound - anchors.baseline
    )
    return min(1.0, max(0.0, float(reward)))


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {token}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise DatasetError("trusted JSON asset must be a regular non-symlink file")
        if info.st_size <= 0 or info.st_size > 64 * 1024 * 1024:
            raise DatasetError("trusted JSON asset has an invalid size")
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except DatasetError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DatasetError("trusted JSON asset cannot be loaded") from exc


def _safe_relative_path(root: Path, raw: Any, *, kind: str, directory: bool) -> Path:
    if not isinstance(raw, str) or not raw:
        raise DatasetError(f"{kind} path is missing")
    relative = Path(raw)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise DatasetError(f"{kind} path is not a safe relative path")
    candidate = root.joinpath(relative)
    try:
        root_real = root.resolve(strict=True)
        candidate_real = candidate.resolve(strict=True)
        candidate_real.relative_to(root_real)
        info = candidate.lstat()
    except (OSError, ValueError) as exc:
        raise DatasetError(f"{kind} asset is missing or leaves heldout root") from exc
    if stat.S_ISLNK(info.st_mode):
        raise DatasetError(f"{kind} asset must not be a symlink")
    if directory and not stat.S_ISDIR(info.st_mode):
        raise DatasetError(f"{kind} asset must be a directory")
    if not directory and not stat.S_ISREG(info.st_mode):
        raise DatasetError(f"{kind} asset must be a regular file")
    return candidate_real


def _validate_chain(chain: Any) -> dict[str, str]:
    if not isinstance(chain, Mapping):
        raise DatasetError("protein chain must be an object")
    if set(chain) != {"chain_id", "sequence"}:
        raise DatasetError("protein chain must contain only chain_id and sequence")
    chain_id = chain.get("chain_id")
    sequence = chain.get("sequence")
    if not isinstance(chain_id, str) or CHAIN_ID_RE.fullmatch(chain_id) is None:
        raise DatasetError("protein chain ID is invalid")
    if not isinstance(sequence, str):
        raise DatasetError("protein sequence is invalid")
    sequence = "".join(sequence.split()).upper()
    if not sequence or SEQUENCE_RE.fullmatch(sequence) is None:
        raise DatasetError("protein sequence is invalid")
    if len(sequence) > MAX_CASE_CHAIN_RESIDUES:
        raise DatasetError("protein sequence exceeds the declared task envelope")
    return {"chain_id": chain_id, "sequence": sequence}


def _parse_case(item: Any) -> CaseAssets:
    if not isinstance(item, Mapping):
        raise DatasetError("held-out item must be an object")
    required = {
        "target_id",
        "protein_chains",
        "ligand_smiles",
        "msa_dir",
        "crystal_ligand_sdf",
        "crystal_protein_pdb",
    }
    if not required.issubset(item):
        raise DatasetError("held-out item is missing required fields")
    target_id = item.get("target_id")
    if not isinstance(target_id, str) or not target_id:
        raise DatasetError("held-out target identity is invalid")
    chains_raw = item.get("protein_chains")
    if (
        not isinstance(chains_raw, list)
        or not chains_raw
        or len(chains_raw) > MAX_CASE_CHAINS
    ):
        raise DatasetError("held-out protein chain list is invalid")
    chains = [_validate_chain(chain) for chain in chains_raw]
    if len({chain["chain_id"] for chain in chains}) != len(chains):
        raise DatasetError("held-out protein chain IDs must be unique")
    smiles = item.get("ligand_smiles")
    if not isinstance(smiles, str) or not smiles.strip() or len(smiles) > 100_000:
        raise DatasetError("held-out ligand SMILES is invalid")

    msa_source = _safe_relative_path(
        HELDOUT_DIR, item.get("msa_dir"), kind="MSA", directory=True
    )
    crystal_ligand = _safe_relative_path(
        HELDOUT_DIR,
        item.get("crystal_ligand_sdf"),
        kind="crystal ligand",
        directory=False,
    )
    crystal_protein = _safe_relative_path(
        HELDOUT_DIR,
        item.get("crystal_protein_pdb"),
        kind="crystal protein",
        directory=False,
    )

    # Deliberately omit target_id and both crystal paths.  The concrete MSA
    # path is filled after this single case is staged in its random scratch.
    child_item = {
        "protein_chains": chains,
        "ligand_smiles": smiles,
        "msa_dir": "__STAGED_PER_CASE__",
    }
    return CaseAssets(
        child_item=child_item,
        expected_chains=tuple(chains),
        msa_source_dir=msa_source,
        crystal_ligand=crystal_ligand,
        crystal_protein=crystal_protein,
    )


def load_heldout_cases() -> list[CaseAssets]:
    raw = _load_json(HELDOUT_DIR / "items.json")
    if not isinstance(raw, list) or not raw:
        raise DatasetError("held-out items.json must contain a non-empty list")
    cases = [_parse_case(item) for item in raw]
    # Duplicate hidden identifiers would make aggregate weighting ambiguous.
    target_ids = [item.get("target_id") for item in raw]
    if len(set(target_ids)) != len(target_ids):
        raise DatasetError("held-out target identities must be unique")
    return cases


def _open_stable_regular_source(path: Path, *, max_bytes: int, error_type: type[RuntimeError]):
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise error_type("source asset must be a regular non-symlink file")
        if before.st_nlink != 1:
            raise error_type("source asset must not be hard-linked")
        if before.st_size <= 0 or before.st_size > max_bytes:
            raise error_type("source asset has an invalid size")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags)
        after = os.fstat(fd)
        if (
            not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or after.st_size != before.st_size
            or after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
        ):
            os.close(fd)
            raise error_type("source asset changed while being opened")
        return os.fdopen(fd, "rb", closefd=True), int(after.st_size)
    except RuntimeError:
        raise
    except OSError as exc:
        raise error_type("source asset cannot be opened safely") from exc


def _source_files(source_root: Path) -> list[tuple[Path, Path, int]]:
    try:
        root_info = source_root.lstat()
    except OSError as exc:
        raise SubmissionError("submission directory is missing") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise SubmissionError("submission must be a regular directory")

    selected: list[tuple[Path, Path, int]] = []
    total = 0
    for directory, dirnames, filenames in os.walk(source_root, followlinks=False):
        current = Path(directory)
        relative_dir = current.relative_to(source_root)
        if len(relative_dir.parts) > MAX_SUBMISSION_DEPTH:
            raise SubmissionError("submission source tree is too deep")
        clean_dirs: list[str] = []
        for name in sorted(dirnames):
            path = current / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise SubmissionError("submission contains a linked or special directory")
            if name == "__pycache__":
                continue
            if name.startswith("."):
                raise SubmissionError("submission contains a hidden directory")
            clean_dirs.append(name)
        dirnames[:] = clean_dirs

        for name in sorted(filenames):
            source = current / name
            info = source.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise SubmissionError("submission contains a linked or special file")
            if name.startswith("."):
                raise SubmissionError("submission contains a hidden file")
            if source.suffix.lower() not in SOURCE_SUFFIXES:
                raise SubmissionError("submission artifact may contain only Python files")
            if name.lower() in {
                "child_predict.py",
                "evaluate.py",
                "grade.py",
                "metric.py",
                "score_pose_worker.py",
                "selfcheck.py",
                "source_contract.py",
            }:
                raise SubmissionError("submission uses a reserved verifier filename")
            if info.st_nlink != 1:
                raise SubmissionError("submission source must not be hard-linked")
            relative = source.relative_to(source_root)
            if len(relative.parts) > MAX_SUBMISSION_DEPTH + 1:
                raise SubmissionError("submission source tree is too deep")
            if info.st_size <= 0 or info.st_size > MAX_SUBMISSION_FILE_BYTES:
                raise SubmissionError("submission source file has an invalid size")
            total += int(info.st_size)
            if total > MAX_SUBMISSION_TOTAL_BYTES:
                raise SubmissionError("submission source tree is too large")
            selected.append((source, relative, int(info.st_size)))
            if len(selected) > MAX_SUBMISSION_FILES:
                raise SubmissionError("submission contains too many source files")

    if not any(relative == Path("solver.py") for _, relative, _ in selected):
        raise SubmissionError("submission must contain top-level solver.py")
    return selected


def _load_trusted_source_contract() -> Any:
    try:
        info = SOURCE_CONTRACT_SCRIPT.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ConfigurationError("trusted source contract must be a regular file")
        spec = importlib.util.spec_from_file_location(
            "trusted_cofold_source_contract", SOURCE_CONTRACT_SCRIPT
        )
        if spec is None or spec.loader is None:
            raise ConfigurationError("cannot load trusted source contract")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for name in ("validate_source_payload", "validate_source_tree"):
            if not callable(getattr(module, name, None)):
                raise ConfigurationError("trusted source contract API is incomplete")
        return module
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError("trusted source contract cannot be loaded") from exc


def _validate_source_payload(payload: bytes) -> None:
    try:
        _load_trusted_source_contract().validate_source_payload(payload)
    except ConfigurationError:
        raise
    except Exception as exc:
        raise SubmissionError(str(exc)) from exc


def _copy_submission(source_root: Path, destination: Path) -> Path:
    try:
        _load_trusted_source_contract().validate_source_tree(source_root)
    except ConfigurationError:
        raise
    except Exception as exc:
        raise SubmissionError(str(exc)) from exc
    files = _source_files(source_root)
    destination.mkdir(mode=0o700)
    for source, relative, declared_size in files:
        parent = destination / relative.parent
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle, actual_size = _open_stable_regular_source(
            source,
            max_bytes=MAX_SUBMISSION_FILE_BYTES,
            error_type=SubmissionError,
        )
        if actual_size != declared_size:
            handle.close()
            raise SubmissionError("submission source changed during copy")
        with handle:
            payload = handle.read()
        if len(payload) != declared_size:
            raise SubmissionError("submission source changed during copy")
        _validate_source_payload(payload)
        target = destination / relative
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(target, flags, 0o400)
        try:
            with os.fdopen(fd, "wb", closefd=True) as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        os.chmod(target, 0o444)

    # Python can read the copied tree, but cannot modify it or create pyc files.
    directories = [path for path in destination.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        os.chmod(path, 0o555)
    os.chmod(destination, 0o555)
    return destination / "solver.py"


def _hash_effective_submission(snapshot: Path) -> str:
    digest = hashlib.sha256(b"protein-cofold-source-snapshot-v1\0")
    files = _source_files(snapshot)
    for source, relative, declared_size in sorted(files, key=lambda value: str(value[1])):
        handle, actual_size = _open_stable_regular_source(
            source,
            max_bytes=MAX_SUBMISSION_FILE_BYTES,
            error_type=SubmissionError,
        )
        with handle:
            payload = handle.read()
        if actual_size != declared_size or len(payload) != declared_size:
            raise SubmissionError("submission snapshot changed while hashing")
        _validate_source_payload(payload)
        relative_bytes = relative.as_posix().encode("utf-8")
        digest.update(len(relative_bytes).to_bytes(4, "big"))
        digest.update(relative_bytes)
        digest.update(declared_size.to_bytes(8, "big"))
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def _trusted_file_sha256(path: Path, *, error_type: type[RuntimeError]) -> str:
    handle, _ = _open_stable_regular_source(
        path,
        max_bytes=64 * 1024 * 1024,
        error_type=error_type,
    )
    digest = hashlib.sha256()
    with handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_submission_snapshot() -> tuple[Path, str]:
    """Freeze the effective source artifact once, before any hidden case is read."""
    parent = Path(tempfile.mkdtemp(prefix="protein_submission_"))
    os.chmod(parent, 0o700)
    try:
        snapshot = parent / "source"
        _copy_submission(SUBMISSION_DIR, snapshot)
        artifact_sha256 = _hash_effective_submission(snapshot)
        return snapshot, artifact_sha256
    except Exception:
        shutil.rmtree(parent, ignore_errors=True)
        raise


def _remove_submission_snapshot(snapshot: Path | None) -> None:
    if snapshot is None:
        return
    shutil.rmtree(snapshot.parent, ignore_errors=True)


def _copy_case_msas(case: CaseAssets, destination: Path) -> None:
    destination.mkdir(mode=0o700)
    total = 0
    for chain in case.expected_chains:
        chain_id = chain["chain_id"]
        source = case.msa_source_dir / f"{chain_id}.a3m"
        handle, size = _open_stable_regular_source(
            source,
            max_bytes=MAX_MSA_FILE_BYTES,
            error_type=DatasetError,
        )
        total += size
        if total > MAX_MSA_TOTAL_BYTES:
            handle.close()
            raise DatasetError("current case MSA payload is too large")
        target = destination / f"{chain_id}.a3m"
        raw = handle.read()
        handle.close()
        # Historical assets can contain embedded NUL bytes rejected by Boltz.
        # Removing NUL is a transport repair; sequence/alignment bytes remain.
        cleaned = raw.replace(b"\x00", b"")
        if not cleaned:
            raise DatasetError("current case MSA is empty")
        target.write_bytes(cleaned)
        os.chmod(target, 0o444)
    os.chmod(destination, 0o555)


def _runner_identity() -> tuple[int, int]:
    try:
        record = pwd.getpwnam(RUNNER_USER)
    except KeyError as exc:
        raise ConfigurationError("dedicated runner user is missing") from exc
    if record.pw_uid == 0 or record.pw_gid == 0 or record.pw_uid == os.geteuid():
        raise ConfigurationError("runner must use a distinct, non-root UID/GID")
    return record.pw_uid, record.pw_gid


def _validate_trusted_launcher_permissions() -> None:
    # The inference child imports grade.py before applying Landlock; the score
    # worker is a separate trusted process.  All launchers must be immutable
    # packaged files.  World-readability is needed by the distinct runner UID.
    for path in (
        CHILD_SCRIPT,
        SCORE_SCRIPT,
        SOURCE_CONTRACT_SCRIPT,
        TESTS_DIR / "grade.py",
    ):
        try:
            info = path.lstat()
        except OSError as exc:
            raise ConfigurationError("trusted child launcher file is missing") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ConfigurationError("trusted child launcher must be a regular file")
        if not info.st_mode & stat.S_IROTH:
            raise ConfigurationError("trusted child launcher is not runner-readable")


def _make_case_scratch(
    case: CaseAssets,
    runner_uid: int,
    runner_gid: int,
    submission_snapshot: Path,
) -> CaseScratch:
    try:
        RUNNER_ROOT.mkdir(parents=True, mode=0o711, exist_ok=True)
        root_info = RUNNER_ROOT.lstat()
    except OSError as exc:
        raise IsolationError("runner scratch root cannot be prepared") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise IsolationError("runner scratch root must be a real directory")
    if root_info.st_uid != os.geteuid():
        raise IsolationError("runner scratch root must be owned by the trusted parent")
    os.chmod(RUNNER_ROOT, 0o711)
    root = Path(tempfile.mkdtemp(prefix="case_", dir=RUNNER_ROOT))
    try:
        submission = root / "submission"
        solver = _copy_submission(submission_snapshot, submission)

        input_dir = root / "input"
        input_dir.mkdir(mode=0o700)
        msa_dir = input_dir / "msa"
        _copy_case_msas(case, msa_dir)
        child_item = dict(case.child_item)
        child_item["msa_dir"] = str(msa_dir)
        item_json = input_dir / "item.json"
        item_json.write_text(
            json.dumps(child_item, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        os.chmod(item_json, 0o444)
        os.chmod(input_dir, 0o555)

        output_dir = root / "output"
        work_dir = root / "work"
        home_dir = root / "home"
        tmp_dir = root / "tmp"
        for directory in (output_dir, work_dir, home_dir, tmp_dir):
            directory.mkdir(mode=0o700)
            os.chown(directory, runner_uid, runner_gid)

        os.chown(root, runner_uid, runner_gid)
        os.chmod(root, 0o700)
        return CaseScratch(
            root=root,
            solver=solver,
            item_json=item_json,
            prediction_json=output_dir / "prediction.json",
            work_dir=work_dir,
        )
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def _landlock_abi(libc: Any) -> int:
    abi = libc.syscall(
        LANDLOCK_CREATE_RULESET,
        ctypes.c_void_p(),
        ctypes.c_size_t(0),
        ctypes.c_uint(LANDLOCK_CREATE_RULESET_VERSION),
    )
    if abi < 1:
        err = ctypes.get_errno()
        raise IsolationError(f"Landlock unavailable (errno={err})")
    return int(abi)


def _add_landlock_path_rule(
    libc: Any, ruleset_fd: int, path: Path, allowed_access: int
) -> None:
    path_fd = os.open(path, O_PATH | os.O_CLOEXEC)
    try:
        attr = _LandlockPathBeneathAttr(
            allowed_access=allowed_access,
            parent_fd=path_fd,
        )
        result = libc.syscall(
            LANDLOCK_ADD_RULE,
            ruleset_fd,
            LANDLOCK_RULE_PATH_BENEATH,
            ctypes.byref(attr),
            0,
        )
        if result < 0:
            err = ctypes.get_errno()
            raise IsolationError(f"Landlock path rule failed (errno={err})")
    finally:
        os.close(path_fd)


def prepare_private_shm_namespace(runner_uid: int, runner_gid: int) -> int:
    """Create an ephemeral per-case /dev/shm before dropping privileges.

    Python's multiprocessing.SemLock (used by the pinned GPU runtime) creates
    and immediately unlinks POSIX semaphore files under /dev/shm.  A fresh
    mount namespace keeps those required writes from becoming a cross-case
    persistence channel.
    """
    if os.geteuid() != 0 or runner_uid <= 0 or runner_gid <= 0:
        raise IsolationError("private shared-memory setup requires trusted root")
    target = Path("/dev/shm")
    try:
        before = target.lstat()
    except OSError as exc:
        raise IsolationError("shared-memory mount point is missing") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise IsolationError("shared-memory mount point must be a real directory")

    libc = ctypes.CDLL(None, use_errno=True)
    libc.unshare.argtypes = [ctypes.c_int]
    libc.unshare.restype = ctypes.c_int
    libc.mount.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_ulong,
        ctypes.c_char_p,
    ]
    libc.mount.restype = ctypes.c_int
    if libc.unshare(CLONE_NEWNS) < 0:
        err = ctypes.get_errno()
        raise IsolationError(f"private mount namespace unavailable (errno={err})")
    if libc.mount(None, b"/", None, MS_REC | MS_PRIVATE, None) < 0:
        err = ctypes.get_errno()
        raise IsolationError(f"mount propagation cannot be isolated (errno={err})")
    options = (
        f"size={PRIVATE_SHM_SIZE},mode=0700,uid={runner_uid},gid={runner_gid}"
    ).encode("ascii")
    if libc.mount(
        b"tmpfs",
        b"/dev/shm",
        b"tmpfs",
        MS_NOSUID | MS_NODEV | MS_NOEXEC,
        options,
    ) < 0:
        err = ctypes.get_errno()
        raise IsolationError(f"private shared-memory mount failed (errno={err})")
    try:
        after = target.lstat()
    except OSError as exc:
        raise IsolationError("private shared-memory mount disappeared") from exc
    if (
        stat.S_ISLNK(after.st_mode)
        or not stat.S_ISDIR(after.st_mode)
        or after.st_dev == before.st_dev
        or after.st_uid != runner_uid
        or after.st_gid != runner_gid
        or stat.S_IMODE(after.st_mode) != 0o700
    ):
        raise IsolationError("private shared-memory mount identity is invalid")
    return int(after.st_dev)


def drop_child_privileges(runner_uid: int, runner_gid: int) -> None:
    """Permanently become the dedicated unprivileged inference identity."""
    if os.geteuid() != 0 or runner_uid <= 0 or runner_gid <= 0:
        raise IsolationError("child privilege drop has invalid identities")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0:
        err = ctypes.get_errno()
        raise IsolationError(f"PR_SET_NO_NEW_PRIVS failed before UID drop (errno={err})")
    try:
        os.setgroups([])
        os.setgid(runner_gid)
        os.setuid(runner_uid)
    except OSError as exc:
        raise IsolationError("child privileges could not be dropped") from exc
    if (
        os.getresuid() != (runner_uid, runner_uid, runner_uid)
        or os.getresgid() != (runner_gid, runner_gid, runner_gid)
        or os.getgroups()
    ):
        raise IsolationError("child privilege drop did not take effect")
    if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) < 0:
        err = ctypes.get_errno()
        raise IsolationError(f"child PR_SET_DUMPABLE failed (errno={err})")
    try:
        status = {}
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if ":" in line:
                name, value = line.split(":", 1)
                status[name] = value.strip()
        capability_fields = ("CapPrm", "CapEff", "CapAmb")
        if any(int(status[name], 16) != 0 for name in capability_fields):
            raise ValueError("capability set is not empty")
        if status.get("NoNewPrivs") != "1":
            raise ValueError("no-new-privileges flag is not set")
    except (OSError, UnicodeError, KeyError, ValueError) as exc:
        raise IsolationError("child privilege state could not be verified") from exc


def restrict_child_filesystem(
    scratch: Path, *, private_shm_device: int | None = None
) -> None:
    """Allow runtime reads while confining persistent writes to this case scratch."""
    libc = ctypes.CDLL(None, use_errno=True)
    abi = _landlock_abi(libc)
    handled = LANDLOCK_READ_ACCESS | LANDLOCK_BASE_WRITE_ACCESS
    if abi >= 2:
        handled |= LANDLOCK_ACCESS_FS_REFER
    if abi >= 3:
        handled |= LANDLOCK_ACCESS_FS_TRUNCATE
    scratch_access = handled

    ruleset_attr = _LandlockRulesetAttr(handled_access_fs=handled)
    ruleset_fd = libc.syscall(
        LANDLOCK_CREATE_RULESET,
        ctypes.byref(ruleset_attr),
        ctypes.sizeof(ruleset_attr),
        0,
    )
    if ruleset_fd < 0:
        err = ctypes.get_errno()
        raise IsolationError(f"Landlock ruleset creation failed (errno={err})")
    try:
        for path in RUNTIME_READ_ROOTS:
            if path.exists():
                _add_landlock_path_rule(libc, ruleset_fd, path, LANDLOCK_READ_ACCESS)
        # CUDA/NVML names worker threads by opening
        # /proc/self/task/<tid>/comm with O_WRONLY|O_TRUNC.  Boltz also starts
        # a fresh Python process, so a rule resolved only at the launcher's
        # /proc/self path would not cover the descendant PID.  Permit only
        # WRITE_FILE (and TRUNCATE on ABI >= 3) across procfs.  Normal procfs
        # ownership/DAC rules still limit the unprivileged runner to its own
        # process tree; the trusted parent uses another UID and is non-dumpable.
        # No procfs make/remove rights are granted, and procfs state disappears
        # when the fully reaped worker tree exits.
        proc = Path("/proc")
        if proc.exists():
            proc_access = LANDLOCK_READ_ACCESS | LANDLOCK_ACCESS_FS_WRITE_FILE
            if abi >= 3:
                proc_access |= LANDLOCK_ACCESS_FS_TRUNCATE
            _add_landlock_path_rule(libc, ruleset_fd, proc, proc_access)
        if private_shm_device is not None:
            private_shm = Path("/dev/shm")
            try:
                shm_info = private_shm.lstat()
            except OSError as exc:
                raise IsolationError("private shared-memory mount is missing") from exc
            if (
                stat.S_ISLNK(shm_info.st_mode)
                or not stat.S_ISDIR(shm_info.st_mode)
                or shm_info.st_dev != private_shm_device
                or shm_info.st_uid != os.geteuid()
                or stat.S_IMODE(shm_info.st_mode) != 0o700
            ):
                raise IsolationError("private shared-memory mount was not preserved")
            # SemLock uses create -> same-directory hard-link -> unlink.  A
            # same-directory link needs no Landlock REFER permission.  No
            # directory, socket, FIFO, symlink, or device creation is allowed.
            shm_access = (
                LANDLOCK_READ_ACCESS
                | LANDLOCK_ACCESS_FS_WRITE_FILE
                | LANDLOCK_ACCESS_FS_REMOVE_FILE
                | LANDLOCK_ACCESS_FS_MAKE_REG
            )
            _add_landlock_path_rule(libc, ruleset_fd, private_shm, shm_access)
        # CUDA opens its existing device nodes read/write.  Grant WRITE_FILE
        # only on discovered GPU character devices, never on /dev as a whole.
        seen_devices: set[Path] = set()
        for device in SAFE_WRITABLE_DEVICE_PATHS:
            try:
                device_info = device.lstat()
            except OSError:
                continue
            if stat.S_ISCHR(device_info.st_mode):
                seen_devices.add(device)
                _add_landlock_path_rule(
                    libc,
                    ruleset_fd,
                    device,
                    LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_WRITE_FILE,
                )
        for pattern in GPU_DEVICE_GLOBS:
            for raw in glob.glob(pattern):
                device = Path(raw)
                try:
                    device_info = device.lstat()
                except OSError:
                    continue
                if device in seen_devices or not stat.S_ISCHR(device_info.st_mode):
                    continue
                seen_devices.add(device)
                _add_landlock_path_rule(
                    libc,
                    ruleset_fd,
                    device,
                    LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_WRITE_FILE,
                )
        if not CHILD_SCRIPT.is_file():
            raise IsolationError("untrusted child entry point is missing")
        _add_landlock_path_rule(
            libc, ruleset_fd, CHILD_SCRIPT, LANDLOCK_ACCESS_FS_READ_FILE
        )
        _add_landlock_path_rule(libc, ruleset_fd, scratch, scratch_access)

        if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0:
            err = ctypes.get_errno()
            raise IsolationError(f"PR_SET_NO_NEW_PRIVS failed (errno={err})")
        if libc.syscall(LANDLOCK_RESTRICT_SELF, ruleset_fd, 0) < 0:
            err = ctypes.get_errno()
            raise IsolationError(f"Landlock restriction failed (errno={err})")
    finally:
        os.close(ruleset_fd)


def restrict_child_persistent_ipc() -> None:
    """Deny persistent state plus namespace/mount privilege recovery paths.

    A separate IPC namespace would require CAP_SYS_ADMIN in typical Docker
    deployments.  A no-new-privileges seccomp filter is the fail-closed,
    capability-free equivalent for this single-GPU inference task.  POSIX
    named shm/mqueues are already blocked by the Landlock write policy on
    /dev/shm and /dev/mqueue; anonymous memfd state dies with the reaped tree.

    The trusted launcher has already created its private mount namespace and
    dropped every real/effective/saved UID, GID, supplementary group, and
    capability before reaching this function.  Denying namespace creation,
    namespace entry, and every mount/chroot API prevents submitted code from
    recovering namespaced capabilities or changing that trusted mount view.
    """
    # clone(2)'s flags argument is arg0 on the pinned linux/amd64 ABI.  Do not
    # silently install the wrong masked rule if this image is ever rebuilt for
    # another architecture.
    if os.uname().machine != "x86_64":
        raise IsolationError("the verifier seccomp policy requires linux/amd64")

    library_name = ctypes.util.find_library("seccomp")
    if not library_name:
        raise IsolationError("libseccomp is required for per-case IPC isolation")
    try:
        seccomp = ctypes.CDLL(library_name, use_errno=True)
    except OSError as exc:
        raise IsolationError("libseccomp cannot be loaded") from exc

    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    seccomp.seccomp_rule_add.restype = ctypes.c_int
    seccomp.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(_ScmpArgCmp),
    ]
    seccomp.seccomp_rule_add_array.restype = ctypes.c_int
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_load.restype = ctypes.c_int
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.restype = None

    context = seccomp.seccomp_init(SCMP_ACT_ALLOW)
    if not context:
        raise IsolationError("seccomp filter initialization failed")
    try:
        deny_action = SCMP_ACT_ERRNO_BASE | errno.EPERM
        resolved_names: set[str] = set()
        denied_syscalls = PERSISTENT_IPC_SYSCALLS + NAMESPACE_MOUNT_SYSCALLS
        for syscall_name in denied_syscalls:
            number = _resolve_seccomp_syscall_number(seccomp, syscall_name)
            if number < 0:  # syscall is absent on this architecture
                continue
            resolved_names.add(syscall_name)
            if seccomp.seccomp_rule_add(context, deny_action, number, 0) != 0:
                raise IsolationError(
                    f"seccomp deny rule could not be added: {syscall_name}"
                )

        clone_number = seccomp.seccomp_syscall_resolve_name(b"clone")
        clone3_number = seccomp.seccomp_syscall_resolve_name(b"clone3")
        if clone_number < 0 or clone3_number < 0:
            raise IsolationError("seccomp cannot resolve clone namespace entry points")
        resolved_names.update({"clone", "clone3"})
        for namespace_flag in CLONE_NAMESPACE_FLAGS:
            comparison = _ScmpArgCmp(
                arg=0,
                op=SCMP_CMP_MASKED_EQ,
                datum_a=namespace_flag,
                datum_b=namespace_flag,
            )
            if (
                seccomp.seccomp_rule_add_array(
                    context,
                    deny_action,
                    clone_number,
                    1,
                    ctypes.byref(comparison),
                )
                != 0
            ):
                raise IsolationError("seccomp clone namespace rule could not be added")
        # Returning ENOSYS makes libc use its established clone(2) fallback for
        # ordinary thread/process creation.  Namespace-bearing clone(2) calls
        # remain subject to the masked rules above.
        clone3_action = SCMP_ACT_ERRNO_BASE | errno.ENOSYS
        if seccomp.seccomp_rule_add(context, clone3_action, clone3_number, 0) != 0:
            raise IsolationError("seccomp clone3 deny rule could not be added")
        direct_sysv = {
            "shmget",
            "shmat",
            "shmdt",
            "shmctl",
            "msgget",
            "msgsnd",
            "msgrcv",
            "msgctl",
            "semget",
            "semop",
            "semctl",
        }
        if "ipc" not in resolved_names and not direct_sysv.issubset(resolved_names):
            raise IsolationError("seccomp cannot cover every System V IPC family")
        if not {"add_key", "request_key", "keyctl"}.issubset(resolved_names):
            raise IsolationError("seccomp cannot cover every keyring syscall")
        if not {"setsid", "setpgid"}.issubset(resolved_names):
            raise IsolationError("seccomp cannot lock the worker process group")
        if not set(NAMESPACE_MOUNT_SYSCALLS).issubset(resolved_names):
            missing = sorted(set(NAMESPACE_MOUNT_SYSCALLS) - resolved_names)
            raise IsolationError(
                "seccomp cannot cover namespace/mount boundary: " + ",".join(missing)
            )
        if seccomp.seccomp_load(context) != 0:
            raise IsolationError("seccomp isolation filter could not be loaded")
    finally:
        seccomp.seccomp_release(context)


def _become_child_subreaper() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    # /proc remains readable because CUDA/PyTorch inspect CPU, memory, driver,
    # and self-process metadata.  The worker has a distinct UID and the trusted
    # parent is explicitly non-dumpable, so /proc/<parent>/{mem,environ,fd,root}
    # cannot become a side channel.  Parent argv/env contain no case identity.
    if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) < 0:
        err = ctypes.get_errno()
        raise IsolationError(f"PR_SET_DUMPABLE failed (errno={err})")
    if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) < 0:
        err = ctypes.get_errno()
        raise IsolationError(f"PR_SET_CHILD_SUBREAPER failed (errno={err})")


def _safe_cache_path_env(name: str) -> str | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        candidate = Path(raw).expanduser().resolve(strict=False)
    except OSError:
        return None
    for root in (Path("/opt"), Path("/usr")):
        try:
            candidate.relative_to(root)
            return str(candidate)
        except ValueError:
            continue
    return None


def _child_environment(scratch: CaseScratch) -> dict[str, str]:
    python_bin = str(Path(sys.executable).resolve().parent)
    path_parts = [
        python_bin,
        "/opt/conda/bin",
        "/usr/local/cuda/bin",
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
    ]
    env = {
        "PATH": ":".join(dict.fromkeys(path_parts)),
        "HOME": str(scratch.root / "home"),
        "TMPDIR": str(scratch.root / "tmp"),
        "TMP": str(scratch.root / "tmp"),
        "TEMP": str(scratch.root / "tmp"),
        "XDG_CACHE_HOME": str(scratch.root / "home" / ".cache"),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    # Preserve only non-secret runtime selectors.  Never forward API keys,
    # cloud/HF tokens, HOME, PYTHONPATH, or verifier anchor values.
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "CUDA_MODULE_LOADING",
        "CUBLAS_WORKSPACE_CONFIG",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "LD_LIBRARY_PATH",
    ):
        value = os.environ.get(name)
        if value:
            env[name] = value
    for name in (
        "BOLTZ_CACHE",
        "TORCH_HOME",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "CHAI_DOWNLOADS_DIR",
    ):
        value = _safe_cache_path_env(name)
        if value is not None:
            env[name] = value
    return env


def _signal_group(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _direct_child_pids() -> list[int]:
    """Find adopted direct children after the worker group has been stopped."""
    parent_pid = os.getpid()
    result: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # /proc/<pid>/stat field 4 is PPID; comm may contain spaces and ')'.
            text = (entry / "stat").read_text(encoding="utf-8")
            after_name = text.rsplit(")", 1)[1].split()
            ppid = int(after_name[1])
        except (OSError, ValueError, IndexError):
            continue
        if ppid == parent_pid:
            result.append(int(entry.name))
    return result


def _kill_and_reap_worker(proc: subprocess.Popen[Any], *, grace_sec: float = 0.3) -> None:
    """Stop the worker group, then kill/reap any descendants that escaped it."""
    pgid = proc.pid
    _signal_group(pgid, signal.SIGTERM)
    deadline = time.monotonic() + grace_sec
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(0.01)
    _signal_group(pgid, signal.SIGKILL)
    try:
        proc.wait(timeout=max(0.1, grace_sec))
    except subprocess.TimeoutExpired:
        _signal_group(pgid, signal.SIGKILL)
        proc.wait(timeout=1.0)

    # A malicious solver can call setsid().  As a subreaper, the grader adopts
    # orphaned descendants; terminate those too before any result is trusted.
    reap_deadline = time.monotonic() + 2.0
    while True:
        adopted = _direct_child_pids()
        if not adopted:
            break
        for pid in adopted:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for pid in adopted:
            try:
                os.waitpid(pid, 0)
            except (ChildProcessError, ProcessLookupError):
                pass
        if time.monotonic() >= reap_deadline and _direct_child_pids():
            raise IsolationError("worker descendants could not be fully reaped")


def _launch_case(
    scratch: CaseScratch,
    *,
    runner_uid: int,
    runner_gid: int,
    timeout_sec: float,
) -> None:
    if timeout_sec <= 0 or not math.isfinite(timeout_sec):
        raise WorkerTimeout("case runtime budget exhausted")
    argv = [
        sys.executable,
        str(CHILD_SCRIPT),
        "--scratch",
        str(scratch.root),
        "--solver",
        str(scratch.solver),
        "--item",
        str(scratch.item_json),
        "--output",
        str(scratch.prediction_json),
        "--runner-uid",
        str(runner_uid),
        "--runner-gid",
        str(runner_gid),
    ]
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(scratch.work_dir),
            env=_child_environment(scratch),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise IsolationError("isolated child could not be launched") from exc

    timed_out = False
    try:
        try:
            return_code = proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            return_code = None
    finally:
        _kill_and_reap_worker(proc)
    if timed_out:
        raise WorkerTimeout("isolated child timed out")
    if return_code == SANDBOX_SETUP_EXIT:
        raise IsolationError("isolated child could not establish its sandbox")
    if return_code != 0:
        raise PredictionError("isolated child failed")


def _validate_prediction_inode(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise PredictionError("prediction JSON is missing") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise PredictionError("prediction JSON must be a regular non-symlink file")
    if info.st_nlink != 1:
        raise PredictionError("prediction JSON must not be hard-linked")
    if info.st_size <= 0 or info.st_size > MAX_PREDICTION_JSON_BYTES:
        raise PredictionError("prediction JSON has an invalid size")


def _finite_bounded_coordinate(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise PredictionError("structure contains an invalid coordinate") from exc
    if not math.isfinite(value) or abs(value) > MAX_STRUCTURE_COORDINATE_ABS:
        raise PredictionError("structure contains an invalid coordinate")
    return value


def _sdf_v2000_counts(text: str, *, trusted: bool = False) -> tuple[int, int, list[str]]:
    error_type: type[RuntimeError] = DatasetError if trusted else PredictionError
    lines = text.splitlines()
    if len(lines) < 4 or any(len(line) > 4096 for line in lines):
        raise error_type("SDF has an invalid bounded text layout")
    counts = lines[3]
    if len(counts) < 6 or "V2000" not in counts:
        raise error_type("SDF must use the bounded V2000 layout")
    try:
        atoms = int(counts[0:3])
        bonds = int(counts[3:6])
    except ValueError as exc:
        raise error_type("SDF counts line is malformed") from exc
    if atoms < 1 or bonds < 0 or len(lines) < 4 + atoms + bonds:
        raise error_type("SDF counts are invalid")
    return atoms, bonds, lines


def _trusted_crystal_ligand_atom_count(path: Path) -> int:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise DatasetError("crystal ligand must be a regular file")
        if info.st_size <= 0 or info.st_size > 4 * 1024 * 1024:
            raise DatasetError("crystal ligand size is invalid")
        text = path.read_text(encoding="utf-8")
    except DatasetError:
        raise
    except (OSError, UnicodeError) as exc:
        raise DatasetError("crystal ligand cannot be read") from exc
    atoms, _, _ = _sdf_v2000_counts(text, trusted=True)
    return atoms


def _validate_prediction_work_bounds(
    prediction: Mapping[str, str], case: CaseAssets, metric_module: Any
) -> None:
    """Use the byte-identical visible/hidden metric's parser-work contract."""
    try:
        metric_module.validate_prediction_work_bounds(
            prediction,
            expected_chains=case.expected_chains,
            crystal_ligand_path=case.crystal_ligand,
        )
    except Exception as exc:
        raise PredictionError("prediction exceeds trusted work bounds") from exc


def _run_trusted_score_worker(
    scratch: CaseScratch, case: CaseAssets
) -> CaseOutcome:
    """Contain native parser/RMSD crashes and pathological graph runtimes."""
    if SCORE_TIMEOUT_SEC <= 0 or not math.isfinite(SCORE_TIMEOUT_SEC):
        raise ConfigurationError("SCORE_TIMEOUT_SEC must be positive and finite")
    with tempfile.TemporaryDirectory(prefix="protein_score_") as temporary:
        root = Path(temporary)
        work = root / "work"
        work.mkdir(mode=0o700)
        request = root / "request.json"
        output = root / "result.json"
        request.write_text(
            json.dumps(
                {
                    "prediction": str(scratch.prediction_json),
                    "crystal_ligand": str(case.crystal_ligand),
                    "crystal_protein": str(case.crystal_protein),
                    "expected_chains": list(case.expected_chains),
                    "expected_ligand_smiles": case.child_item["ligand_smiles"],
                    "work_dir": str(work),
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        os.chmod(request, 0o400)
        try:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(SCORE_SCRIPT),
                    "--metric",
                    str(METRIC_SCRIPT),
                    "--request",
                    str(request),
                    "--output",
                    str(output),
                ],
                cwd=str(work),
                env=_child_environment(scratch),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ConfigurationError("trusted score worker could not launch") from exc

        timed_out = False
        try:
            try:
                return_code = proc.wait(timeout=SCORE_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                timed_out = True
                return_code = None
        finally:
            _kill_and_reap_worker(proc)
        if timed_out or return_code != 0:
            raise PredictionError("prediction failed bounded trusted scoring")

        try:
            info = output.lstat()
            if (
                stat.S_ISLNK(info.st_mode)
                or not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_size <= 0
                or info.st_size > 4096
            ):
                raise ValueError("invalid score result inode")
            result = json.loads(output.read_text(encoding="utf-8"))
            if set(result) != {"passed", "pb_valid", "rmsd_within_2a"}:
                raise ValueError("invalid score result schema")
            if any(type(value) is not bool for value in result.values()):
                raise ValueError("invalid score result type")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise PredictionError("trusted score worker returned no valid result") from exc
        return CaseOutcome(
            passed=result["passed"],
            pb_valid=result["pb_valid"],
            rmsd_within_2a=result["rmsd_within_2a"],
        )


def _load_trusted_metric() -> Any:
    try:
        info = METRIC_SCRIPT.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ConfigurationError("trusted metric must be a regular non-symlink file")
        spec = importlib.util.spec_from_file_location("trusted_cofold_metric", METRIC_SCRIPT)
        if spec is None or spec.loader is None:
            raise ConfigurationError("cannot create trusted metric module spec")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        for name in (
            "load_prediction_json",
            "validate_prediction_work_bounds",
            "score_pose",
            "success_rate",
        ):
            if not callable(getattr(module, name, None)):
                raise ConfigurationError("trusted metric API is incomplete")
        return module
    except ConfigurationError:
        raise
    except Exception as exc:  # noqa: BLE001 - converted to a non-sensitive code
        raise ConfigurationError("trusted metric cannot be loaded") from exc


def _preflight_trusted_metric_and_cases(
    metric_module: Any, cases: Sequence[CaseAssets]
) -> None:
    """Separate verifier/gold failures from submission-caused case failures."""
    required_private_api = (
        "_new_posebusters",
        "_read_protein_chains",
        "_validate_expected_chains",
        "_candidate_assignments",
        "_superpose_proteins",
        "_parse_ligand_sdf",
    )
    if any(not callable(getattr(metric_module, name, None)) for name in required_private_api):
        raise ConfigurationError("trusted metric preflight API is incomplete")
    try:
        importlib.import_module("gemmi")
        importlib.import_module("rdkit")
        metric_module._new_posebusters()
    except Exception as exc:  # noqa: BLE001 - trusted dependency/config failure
        raise ConfigurationError("trusted metric runtime preflight failed") from exc

    for case in cases:
        try:
            crystal_chains = metric_module._read_protein_chains(case.crystal_protein)
            expected = metric_module._validate_expected_chains(
                case.expected_chains
            )
            crystal_assignments = metric_module._candidate_assignments(
                crystal_chains,
                expected,
                allow_extra_observed=True,
            )
            crystal_target_chains = tuple(
                crystal_chains[index] for index in crystal_assignments[0]
            )
            # Validate both crystal parsing and expected-chain correspondence
            # before any adversarial output is run.  A deposited biological
            # assembly may contain additional chains; use the same trusted
            # sequence-only selection as the scorer instead of pretending the
            # complete crystal assembly is a submitted prediction.
            metric_module._superpose_proteins(
                crystal_target_chains,
                crystal_chains,
                case.expected_chains,
            )
            crystal_ligand_text = case.crystal_ligand.read_text(encoding="utf-8")
            crystal_ligand = metric_module._parse_ligand_sdf(crystal_ligand_text)
            metric_module._validate_ligand_identity(
                crystal_ligand, case.child_item["ligand_smiles"]
            )
            _trusted_crystal_ligand_atom_count(case.crystal_ligand)
        except Exception as exc:  # noqa: BLE001 - trusted gold failure
            raise DatasetError("trusted crystal asset failed metric preflight") from exc


def _score_one_case(
    case: CaseAssets,
    metric_module: Any,
    *,
    runner_uid: int,
    runner_gid: int,
    timeout: float,
    submission_snapshot: Path,
) -> CaseOutcome:
    scratch = _make_case_scratch(
        case, runner_uid, runner_gid, submission_snapshot
    )
    try:
        _launch_case(
            scratch,
            runner_uid=runner_uid,
            runner_gid=runner_gid,
            timeout_sec=timeout,
        )
        # The entire untrusted process tree is gone before this point.
        _validate_prediction_inode(scratch.prediction_json)
        try:
            prediction = metric_module.load_prediction_json(scratch.prediction_json)
        except Exception as exc:  # malformed output is one failed complex
            raise PredictionError("prediction JSON failed trusted validation") from exc
        _validate_prediction_work_bounds(prediction, case, metric_module)
        return _run_trusted_score_worker(scratch, case)
    finally:
        shutil.rmtree(scratch.root, ignore_errors=True)


def _score_all_cases(
    cases: Sequence[CaseAssets], metric_module: Any, submission_snapshot: Path
) -> dict[str, Any]:
    runner_uid, runner_gid = _runner_identity()
    _validate_trusted_launcher_permissions()
    _become_child_subreaper()
    deadline = time.monotonic() + TOTAL_TIMEOUT_SEC
    scores: list[CaseOutcome] = []
    invalid_cases = 0
    try:
        for case in cases:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                scores.append(CaseOutcome(False, False, False))
                invalid_cases += 1
                continue
            try:
                score = _score_one_case(
                    case,
                    metric_module,
                    runner_uid=runner_uid,
                    runner_gid=runner_gid,
                    timeout=min(CASE_TIMEOUT_SEC, remaining),
                    submission_snapshot=submission_snapshot,
                )
                scores.append(score)
            except PredictionError:
                # Match public self-check semantics: an invalid pose, child
                # error, or per-case timeout is a failed complex, not a way to
                # delete that complex from the denominator.  No prefix or
                # failure index is emitted while the suite is running.
                scores.append(CaseOutcome(False, False, False))
                invalid_cases += 1
    finally:
        # RUNNER_ROOT contains no persistent agent-visible state or cache.
        try:
            RUNNER_ROOT.rmdir()
        except OSError:
            pass

    metric = float(metric_module.success_rate(scores))
    if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
        raise PredictionError("trusted aggregate metric is invalid")
    return {
        "metric": metric,
        "n_cases": len(scores),
        "passed_cases": sum(int(score.passed) for score in scores),
        "pb_valid_cases": sum(int(score.pb_valid) for score in scores),
        "rmsd_within_2a_cases": sum(int(score.rmsd_within_2a) for score in scores),
        "invalid_cases": invalid_cases,
    }


def _safe_error_code(exc: BaseException) -> str:
    if isinstance(exc, ConfigurationError):
        return "configuration_error"
    if isinstance(exc, DatasetError):
        return "dataset_error"
    if isinstance(exc, SubmissionError):
        return "submission_error"
    if isinstance(exc, WorkerTimeout):
        return "timeout"
    if isinstance(exc, IsolationError):
        return "isolation_error"
    if isinstance(exc, PredictionError):
        return "prediction_error"
    return "internal_error"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_outputs(result: Mapping[str, Any]) -> None:
    # reward.json stays numeric for Harbor consumers.  grade_debug.json is
    # still aggregate-only and intentionally has no errors, paths, IDs, or
    # per-case records.
    numeric = {
        "reward": float(result.get("reward", 0.0)),
        "metric": float(result.get("metric", 0.0)),
        "n_cases": int(result.get("n_cases", 0)),
        "passed_cases": int(result.get("passed_cases", 0)),
    }
    _atomic_write(VERIFIER_LOG_DIR / "reward.txt", f"{numeric['reward']}\n")
    _atomic_write(
        VERIFIER_LOG_DIR / "reward.json",
        json.dumps(numeric, sort_keys=True, separators=(",", ":")),
    )
    _atomic_write(
        VERIFIER_LOG_DIR / "grade_debug.json",
        json.dumps(dict(result), sort_keys=True, separators=(",", ":")),
    )
    _atomic_write(
        VERIFIER_LOG_DIR / "score_details.json",
        json.dumps(dict(result), sort_keys=True, separators=(",", ":")),
    )


def main() -> int:
    result: dict[str, Any] = {
        "reward": 0.0,
        "metric": 0.0,
        "n_cases": 0,
        "passed_cases": 0,
        "pb_valid_cases": 0,
        "rmsd_within_2a_cases": 0,
        "invalid_cases": 0,
        "correctness": False,
        "error_code": "internal_error",
        "error": "grader failed: verifier did not complete",
        "protocol": "protein-cofold-sealed-runner-v2",
        "split_protocol": "posebusters-visible20-hidden42-isolated-final",
        "artifact_sha256": "unavailable",
        "split_manifest_sha256": "unavailable",
        "metric_sha256": "unavailable",
        "grader_sha256": "unavailable",
        "score_worker_sha256": "unavailable",
        "source_contract_sha256": "unavailable",
    }
    submission_snapshot: Path | None = None
    try:
        if not math.isfinite(CASE_TIMEOUT_SEC) or CASE_TIMEOUT_SEC <= 0:
            raise ConfigurationError("CASE_TIMEOUT_SEC must be positive and finite")
        if not math.isfinite(SCORE_TIMEOUT_SEC) or SCORE_TIMEOUT_SEC <= 0:
            raise ConfigurationError("SCORE_TIMEOUT_SEC must be positive and finite")
        if not math.isfinite(TOTAL_TIMEOUT_SEC) or TOTAL_TIMEOUT_SEC <= 0:
            raise ConfigurationError("TOTAL_TIMEOUT_SEC must be positive and finite")
        anchors = load_anchors()
        # Freeze the only executable artifact before reading hidden items.  All
        # cases clone this root-owned snapshot; a live workspace can no longer
        # serve different code to different hidden complexes.
        submission_snapshot, artifact_sha256 = _make_submission_snapshot()
        result["artifact_sha256"] = artifact_sha256
        cases = load_heldout_cases()
        result["split_manifest_sha256"] = _trusted_file_sha256(
            HELDOUT_DIR / "items.json", error_type=DatasetError
        )
        result["metric_sha256"] = _trusted_file_sha256(
            METRIC_SCRIPT, error_type=ConfigurationError
        )
        result["grader_sha256"] = _trusted_file_sha256(
            Path(__file__).resolve(), error_type=ConfigurationError
        )
        result["score_worker_sha256"] = _trusted_file_sha256(
            SCORE_SCRIPT, error_type=ConfigurationError
        )
        result["source_contract_sha256"] = _trusted_file_sha256(
            SOURCE_CONTRACT_SCRIPT, error_type=ConfigurationError
        )
        metric_module = _load_trusted_metric()
        _preflight_trusted_metric_and_cases(metric_module, cases)
        aggregate = _score_all_cases(cases, metric_module, submission_snapshot)
        result.update(aggregate)
        result["reward"] = normalized_reward(result["metric"], anchors)
        result["correctness"] = True
        result["error_code"] = "none"
        result["error"] = ""
    except Exception as exc:  # noqa: BLE001 - output must not expose hidden details
        # Do not publish partial counts: even aggregate prefixes can reveal
        # which hidden case caused a targeted failure.
        error_code = _safe_error_code(exc)
        infrastructure = error_code in {
            "configuration_error", "dataset_error", "isolation_error", "internal_error",
        }
        result.update(
            reward=0.0,
            metric=0.0,
            n_cases=0,
            passed_cases=0,
            pb_valid_cases=0,
            rmsd_within_2a_cases=0,
            invalid_cases=0,
            correctness=False,
            error_code=error_code,
            error=("grader failed: " if infrastructure else "submission failed: ") + error_code,
        )
    finally:
        _remove_submission_snapshot(submission_snapshot)
        try:
            _write_outputs(result)
        except Exception:  # noqa: BLE001 - last-resort Harbor scalar
            try:
                VERIFIER_LOG_DIR.mkdir(parents=True, exist_ok=True)
                (VERIFIER_LOG_DIR / "reward.txt").write_text("0\n", encoding="utf-8")
            except Exception:
                pass
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
