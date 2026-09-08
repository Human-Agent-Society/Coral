"""Static guard over the submission before it is imported. Advisory, not a sandbox."""
import ast
from pathlib import Path

BANNED_IMPORTS = {"socket", "subprocess", "ctypes", "multiprocessing", "shutil"}
BANNED_CALLS = {"eval", "exec", "compile", "__import__", "open"}


def violations(source_dir: Path) -> list[str]:
    out = []
    for f in sorted(Path(source_dir).rglob("*.py")):
        try:
            tree = ast.parse(f.read_text(), filename=str(f))
        except SyntaxError as e:
            out.append(f"{f.name}: syntax error: {e}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in BANNED_IMPORTS:
                        out.append(f"{f.name}:{node.lineno}: import {a.name}")
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] in BANNED_IMPORTS:
                    out.append(f"{f.name}:{node.lineno}: from {node.module}")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in BANNED_CALLS:
                    out.append(f"{f.name}:{node.lineno}: call {node.func.id}()")
    return out
