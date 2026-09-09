"""Local Harbor task inspection and private staging helpers.

The initial compatibility profile intentionally stays small: one local,
single-step, Linux container task using Harbor schema 1.4.  Harbor itself is
still the runtime authority; these host-side checks only fail early and keep
portable verifier assets out of CORAL agent worktrees.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HARBOR_RUNTIME_VERSION = "0.22.0"
HARBOR_SCHEMA_VERSION = "1.4"
HARBOR_GRADER_ENTRYPOINT = "coral.grader.harbor:HarborTaskGrader"
HARBOR_PRIVATE_TASK_DIR = "harbor_task"
HARBOR_ADAPTER_MARKER = "local-single-step-v1"

_ENVIRONMENT_DEFINITIONS = ("Dockerfile", "docker-compose.yaml", "docker-compose.yml")
_CANARY_LINE_RE = re.compile(r"^(<!--.*canary.*-->|#.*canary.*)$", re.IGNORECASE)


@dataclass(frozen=True)
class HarborTaskDescriptor:
    """Validated metadata needed to configure the local Harbor adapter."""

    source: Path
    name: str
    instruction: str
    schema_version: str
    package_version: str | None
    digest: str


def _strip_canary(text: str) -> str:
    """Match Harbor v0.22.0's leading instruction canary removal."""
    lines = text.split("\n")
    index = 0
    while index < len(lines) and _CANARY_LINE_RE.match(lines[index].strip()):
        index += 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    return "\n".join(lines[index:])


def _resolve_local_source(source: str, base_dir: Path | None) -> Path:
    if not isinstance(source, str) or not source.strip():
        raise ValueError("task.source must be a non-empty local Harbor task directory")

    path = Path(source).expanduser()
    if not path.is_absolute():
        path = (base_dir or Path.cwd()) / path
    path = path.resolve()

    if not path.is_dir():
        if "@" in source and not source.startswith(("./", "../", "/")):
            raise ValueError(
                "Registry Harbor task sources are not supported by the initial adapter; "
                "use a local task directory"
            )
        raise ValueError(f"Local Harbor task directory not found: {path}")
    return path


def _reject_symlinks(task_dir: Path) -> None:
    for path in task_dir.rglob("*"):
        if path.is_symlink():
            rel = path.relative_to(task_dir)
            raise ValueError(
                f"Initial Harbor compatibility does not support symlinks: {rel.as_posix()}"
            )
        if not path.is_dir() and not path.is_file():
            rel = path.relative_to(task_dir)
            raise ValueError(
                "Initial Harbor compatibility supports regular files and directories only: "
                f"{rel.as_posix()}"
            )


def _task_digest(task_dir: Path) -> str:
    """Hash relative paths and bytes without host-specific filesystem modes."""
    digest = hashlib.sha256()
    for path in sorted(task_dir.rglob("*"), key=lambda item: item.relative_to(task_dir).as_posix()):
        rel = path.relative_to(task_dir).as_posix()
        kind = "dir" if path.is_dir() else "file"
        digest.update(f"{kind}\0{rel}\0".encode())
        if path.is_file():
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def inspect_local_harbor_task(
    source: str | Path,
    *,
    base_dir: Path | None = None,
) -> HarborTaskDescriptor:
    """Validate and describe an initial-profile local Harbor task."""
    task_dir = _resolve_local_source(str(source), base_dir)
    _reject_symlinks(task_dir)

    config_path = task_dir / "task.toml"
    instruction_path = task_dir / "instruction.md"
    environment_dir = task_dir / "environment"
    test_path = task_dir / "tests" / "test.sh"

    if not config_path.is_file():
        raise ValueError(f"Harbor task is missing task.toml: {task_dir}")
    if not instruction_path.is_file():
        raise ValueError(f"Harbor task is missing instruction.md: {task_dir}")
    if not environment_dir.is_dir():
        raise ValueError(f"Harbor task is missing environment/: {task_dir}")
    if not any((environment_dir / name).is_file() for name in _ENVIRONMENT_DEFINITIONS):
        raise ValueError(
            "Initial Harbor compatibility requires environment/Dockerfile or "
            "environment/docker-compose.yaml"
        )
    if not test_path.is_file():
        raise ValueError("Initial Harbor compatibility requires a Linux tests/test.sh verifier")

    try:
        raw: dict[str, Any] = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Invalid Harbor task.toml: {exc}") from exc

    schema_version = str(raw.get("schema_version", raw.get("version", "")))
    if schema_version != HARBOR_SCHEMA_VERSION:
        raise ValueError(
            f"Initial Harbor compatibility requires task schema {HARBOR_SCHEMA_VERSION}, "
            f"got {schema_version or 'missing'}"
        )
    if raw.get("steps"):
        raise ValueError("Initial Harbor compatibility supports single-step tasks only")
    if raw.get("artifacts"):
        raise ValueError(
            "Initial Harbor compatibility does not yet publish declared task artifacts"
        )

    environment = raw.get("environment") or {}
    if not isinstance(environment, dict):
        raise ValueError("Harbor task.toml [environment] must be a table")
    task_os = str(environment.get("os", "linux")).lower()
    if task_os != "linux":
        raise ValueError(
            f"Initial Harbor compatibility supports Linux container tasks only, got {task_os!r}"
        )

    task_table = raw.get("task") or {}
    if not isinstance(task_table, dict):
        raise ValueError("Harbor task.toml [task] must be a table")
    name = task_table.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Initial Harbor compatibility requires non-empty [task].name")

    instruction = _strip_canary(instruction_path.read_text(encoding="utf-8")).strip()
    if not instruction:
        raise ValueError("Harbor instruction.md must not be empty")

    package_version = task_table.get("version")
    if package_version is not None and not isinstance(package_version, str):
        raise ValueError("Harbor [task].version must be a string when present")

    return HarborTaskDescriptor(
        source=task_dir,
        name=name.strip(),
        instruction=instruction,
        schema_version=schema_version,
        package_version=package_version,
        digest=_task_digest(task_dir),
    )


def stage_local_harbor_task(
    source: str | Path,
    *,
    base_dir: Path,
    private_dir: Path,
    expected_digest: str,
) -> Path:
    """Copy a verified Harbor task into CORAL's manager-only private area."""
    descriptor = inspect_local_harbor_task(source, base_dir=base_dir)
    if descriptor.digest != expected_digest:
        raise ValueError(
            "Harbor task changed after configuration was loaded: "
            f"expected {expected_digest}, got {descriptor.digest}"
        )

    destination = private_dir / HARBOR_PRIVATE_TASK_DIR
    if destination.exists():
        raise FileExistsError(f"Harbor private task destination already exists: {destination}")
    shutil.copytree(descriptor.source, destination)

    staged = inspect_local_harbor_task(destination)
    if staged.digest != expected_digest:
        raise RuntimeError(
            f"Staged Harbor task digest mismatch: expected {expected_digest}, got {staged.digest}"
        )
    return destination
