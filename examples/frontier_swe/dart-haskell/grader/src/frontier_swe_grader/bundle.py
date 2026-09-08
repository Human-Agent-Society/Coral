"""Strict parser for the cumulative source-bundle candidate format."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

_FILE_HEADER = re.compile(r"^=== FILE: (.+) ===$")
_FILE_FOOTER = "=== END FILE ==="


class BundleError(ValueError):
    """Raised when a candidate bundle is malformed or unsafe to materialize."""


def parse_bundle(text: str) -> list[tuple[PurePosixPath, str]]:
    """Parse a bundle while rejecting ambiguous or escaping paths."""

    if "\x00" in text:
        raise BundleError("bundle contains a NUL byte")

    lines = text.splitlines(keepends=True)
    files: list[tuple[PurePosixPath, str]] = []
    seen: set[PurePosixPath] = set()
    index = 0

    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue

        header = lines[index].rstrip("\r\n")
        match = _FILE_HEADER.fullmatch(header)
        if match is None:
            raise BundleError(f"expected file header at line {index + 1}")

        relative = _safe_relative_path(match.group(1))
        if relative in seen:
            raise BundleError(f"duplicate file section: {relative}")
        seen.add(relative)
        index += 1

        content: list[str] = []
        while index < len(lines) and lines[index].rstrip("\r\n") != _FILE_FOOTER:
            content.append(lines[index])
            index += 1
        if index >= len(lines):
            raise BundleError(f"missing end marker for {relative}")

        files.append((relative, "".join(content)))
        index += 1

    if not files:
        raise BundleError("bundle contains no files")
    return files


def materialize_bundle(text: str, destination: Path) -> list[Path]:
    """Write parsed bundle files below an already isolated destination."""

    written: list[Path] = []
    for relative, content in parse_bundle(text):
        output = destination.joinpath(*relative.parts)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        written.append(output)
    return written


def _safe_relative_path(raw: str) -> PurePosixPath:
    if not raw or "\\" in raw:
        raise BundleError(f"invalid bundle path: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or path.as_posix() != raw:
        raise BundleError(f"bundle path must be normalized and relative: {raw!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise BundleError(f"bundle path escapes or aliases the workspace: {raw!r}")
    return path
