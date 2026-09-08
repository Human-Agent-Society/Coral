from __future__ import annotations

import ast
from pathlib import Path


ALLOWED_IMPORT_ROOTS = {"__future__", "functools", "math", "torch", "triton", "typing"}
RESERVED_IMPORT_ROOTS = {
    "builtins", "child_eval", "gc", "grade", "importlib", "io", "json", "numpy",
    "os", "pathlib", "protocol", "restricted_runner", "submission_guard", "sys", "time",
}
FORBIDDEN_NAMES = {
    "__import__", "breakpoint", "compile", "delattr", "dir", "eval", "exec",
    "getattr", "globals", "hasattr", "help", "input", "locals", "open", "setattr",
    "type", "vars",
}
FORBIDDEN_ATTRS = {
    "Event", "ctypes", "environ", "getenv", "inspect", "pathlib", "pickle",
    "subprocess", "system", "popen", "walk", "listdir", "scandir", "read_text",
    "read_bytes", "write_text", "write_bytes", "time", "perf_counter",
    "current_stream", "default_stream", "synchronize", "load_inline", "cpp_extension",
    "Stream", "ExternalStream", "stream", "set_stream", "wait_stream", "record_stream",
    "CUDAGraph", "graph", "make_graphed_callables", "open",
}
FORBIDDEN_TEXT = {
    "/logs", "/proc", "/runner", "/tests", "anchor_status", "anchors.json",
    "grade.py", "reward.json", "score_details.json", "test_spec.json",
}
PROTECTED_ASSIGNMENT_ROOTS = {"torch", "triton", "time", "os", "sys"}


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return node.attr if not prefix else prefix + "." + node.attr
    return ""


def _string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        pieces = []
        for value in node.values:
            piece = _string(value)
            if piece is None:
                return None
            pieces.append(piece)
        return "".join(pieces)
    return None


def violations(root: Path) -> list[str]:
    failures: list[str] = []
    local_modules = {path.stem for path in root.rglob("*.py")}
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            failures.append(f"{path.name}: syntax error: {exc}")
            continue
        for node in ast.walk(tree):
            literal = _string(node)
            if literal is not None:
                lowered = literal.lower()
                for token in FORBIDDEN_TEXT:
                    if token in lowered:
                        failures.append(f"{path.name}: forbidden literal {token!r}")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imported = [item.name for item in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                for item in imported:
                    root_name = item.split(".", 1)[0]
                    if root_name in RESERVED_IMPORT_ROOTS or (
                        root_name and root_name not in ALLOWED_IMPORT_ROOTS and root_name not in local_modules
                    ):
                        failures.append(f"{path.name}: import {item!r} is not allowed")
                if isinstance(node, ast.ImportFrom):
                    for item in node.names:
                        if item.name in FORBIDDEN_ATTRS:
                            failures.append(f"{path.name}: imported accelerator API {item.name!r} is not allowed")
            elif isinstance(node, ast.Call):
                called = _name(node.func)
                if "." not in called and called in FORBIDDEN_NAMES:
                    failures.append(f"{path.name}: call {called!r} is not allowed")
            elif isinstance(node, ast.Attribute):
                if node.attr in FORBIDDEN_ATTRS or node.attr.startswith("__"):
                    failures.append(f"{path.name}: attribute {node.attr!r} is not allowed")
            elif isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
                failures.append(f"{path.name}: builtin/dunder reference {node.id!r} is not allowed")
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
                targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
                for target in targets:
                    qualified = _name(target)
                    if isinstance(target, ast.Attribute):
                        failures.append(f"{path.name}: attribute mutation is not allowed: {qualified!r}")
    return sorted(set(failures))
