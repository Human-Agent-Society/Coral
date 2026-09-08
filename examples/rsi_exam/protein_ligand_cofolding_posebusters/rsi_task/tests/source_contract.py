#!/usr/bin/env python3
"""Public source-only submission contract shared by self-check and verifier."""

from __future__ import annotations

import argparse
import ast
import json
import os
import stat
from pathlib import Path


MAX_FILES = 32
MAX_FILE_BYTES = 128 * 1024
MAX_TOTAL_BYTES = 256 * 1024
MAX_DEPTH = 4
MAX_AST_NODES = 20_000
MAX_LITERAL_BYTES = 16 * 1024
MAX_LITERAL_TOTAL_BYTES = 64 * 1024
FORBIDDEN_DATA_MODULES = frozenset(
    {"base64", "binascii", "bz2", "gzip", "lzma", "marshal", "pickle", "zlib"}
)
FORBIDDEN_DYNAMIC_CODE_NAMES = frozenset({"__import__", "compile", "eval", "exec"})
RESERVED_FILENAMES = frozenset(
    {
        "child_predict.py",
        "evaluate.py",
        "grade.py",
        "metric.py",
        "score_pose_worker.py",
        "selfcheck.py",
        "source_contract.py",
    }
)


class SourceContractError(ValueError):
    """The editable source tree is not a valid source-only artifact."""


def validate_source_payload(payload: bytes) -> None:
    try:
        text = payload.decode("utf-8")
        tree = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise SourceContractError("Python source is invalid") from exc

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise SourceContractError("source contains an oversized static program")

    literal_total = 0
    for node in nodes:
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            literal = (
                node.value.encode("utf-8")
                if isinstance(node.value, str)
                else node.value
            )
            if len(literal) > MAX_LITERAL_BYTES:
                raise SourceContractError("source contains an oversized static literal")
            literal_total += len(literal)
        elif isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            if roots & FORBIDDEN_DATA_MODULES:
                raise SourceContractError("source imports an embedded-data codec")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root in FORBIDDEN_DATA_MODULES:
                raise SourceContractError("source imports an embedded-data codec")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FORBIDDEN_DYNAMIC_CODE_NAMES
        ):
            raise SourceContractError("source uses dynamic code loading")

    if literal_total > MAX_LITERAL_TOTAL_BYTES:
        raise SourceContractError("source contains too much static literal data")


def validate_source_tree(root: str | os.PathLike[str]) -> tuple[str, ...]:
    source_root = Path(root)
    try:
        root_info = source_root.lstat()
    except OSError as exc:
        raise SourceContractError("source directory is missing") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise SourceContractError("source must be a regular directory")

    selected: list[tuple[Path, Path, int]] = []
    total = 0
    for directory, dirnames, filenames in os.walk(source_root, followlinks=False):
        current = Path(directory)
        relative_dir = current.relative_to(source_root)
        if len(relative_dir.parts) > MAX_DEPTH:
            raise SourceContractError("source tree is too deep")
        clean_dirs: list[str] = []
        for name in sorted(dirnames):
            path = current / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise SourceContractError("source contains a linked or special directory")
            if name == "__pycache__":
                continue
            if name.startswith("."):
                raise SourceContractError("source contains a hidden directory")
            clean_dirs.append(name)
        dirnames[:] = clean_dirs

        for name in sorted(filenames):
            path = current / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise SourceContractError("source contains a linked or special file")
            if name.startswith("."):
                raise SourceContractError("source contains a hidden file")
            if path.suffix.lower() != ".py":
                raise SourceContractError("source artifact may contain only Python files")
            if name.lower() in RESERVED_FILENAMES:
                raise SourceContractError("source uses a reserved verifier filename")
            if info.st_nlink != 1:
                raise SourceContractError("source file must not be hard-linked")
            relative = path.relative_to(source_root)
            if len(relative.parts) > MAX_DEPTH + 1:
                raise SourceContractError("source tree is too deep")
            if info.st_size <= 0 or info.st_size > MAX_FILE_BYTES:
                raise SourceContractError("source file has an invalid size")
            total += int(info.st_size)
            if total > MAX_TOTAL_BYTES:
                raise SourceContractError("source tree is too large")
            selected.append((path, relative, int(info.st_size)))
            if len(selected) > MAX_FILES:
                raise SourceContractError("source contains too many files")

    if not any(relative == Path("solver.py") for _, relative, _ in selected):
        raise SourceContractError("source must contain top-level solver.py")
    for path, _, declared_size in selected:
        payload = path.read_bytes()
        if len(payload) != declared_size:
            raise SourceContractError("source changed while being validated")
        validate_source_payload(payload)
    return tuple(relative.as_posix() for _, relative, _ in selected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    files = validate_source_tree(args.source)
    print(json.dumps({"source_contract": "PASS", "python_files": len(files)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
