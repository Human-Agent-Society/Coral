from __future__ import annotations

import ast
import json
from pathlib import Path


ALLOWED_IMPORT_ROOTS = {"__future__", "functools", "math", "torch", "triton", "typing"}
FORBIDDEN_NAMES = {
    "__import__", "breakpoint", "compile", "delattr", "dir", "eval", "exec",
    "getattr", "globals", "help", "input", "locals", "open", "setattr", "vars",
}
FORBIDDEN_ATTRS = {
    "Event", "ctypes", "inspect", "pathlib", "pickle", "subprocess", "system",
    "popen", "walk", "listdir", "scandir", "read_text", "read_bytes", "write_text",
    "write_bytes", "time", "perf_counter", "current_stream", "default_stream",
    "synchronize", "load_inline", "cpp_extension",
}
FORBIDDEN_CALLS = {
    "torch.load", "torch.save", "torch.from_file", "torch.cuda.Event",
    "torch.cuda.synchronize", "torch.cuda.current_stream", "torch.cuda.default_stream",
    "torch.utils.cpp_extension.load", "torch.utils.cpp_extension.load_inline",
    "torch.nn.functional.scaled_dot_product_attention",
}
# Sealed token list; absent from the agent image, so the set is empty there.
_TOKEN_FILE = Path(__file__).with_name("guard_tokens.json")
FORBIDDEN_TEXT = (
    set(json.loads(_TOKEN_FILE.read_text()))
    if _TOKEN_FILE.is_file()
    else set()
)


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name(node.value)
        return node.attr if not prefix else prefix + "." + node.attr
    return ""


def _constant_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _constant_string(node.left), _constant_string(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        parts = [_constant_string(value) for value in node.values]
        return None if any(part is None for part in parts) else "".join(parts)
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
            literal = _constant_string(node)
            if literal is not None:
                lowered = literal.lower()
                for token in FORBIDDEN_TEXT:
                    if token in lowered:
                        failures.append(f"{path.name}: forbidden literal {token!r}")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports = [item.name for item in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                for imported in imports:
                    root_name = imported.split(".", 1)[0]
                    if root_name and root_name not in ALLOWED_IMPORT_ROOTS and root_name not in local_modules:
                        failures.append(f"{path.name}: import {imported!r} is not allowed")
            elif isinstance(node, ast.Call):
                called = _name(node.func)
                if (
                    called in FORBIDDEN_CALLS
                    or "scaled_dot_product" in called
                    or ("." not in called and called in FORBIDDEN_NAMES)
                ):
                    failures.append(f"{path.name}: call {called!r} is not allowed")
            elif isinstance(node, ast.Attribute):
                if node.attr in FORBIDDEN_ATTRS or node.attr.startswith("__"):
                    failures.append(f"{path.name}: attribute {node.attr!r} is not allowed")
    return sorted(set(failures))
