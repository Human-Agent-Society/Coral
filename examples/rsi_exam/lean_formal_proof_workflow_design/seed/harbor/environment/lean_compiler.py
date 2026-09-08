"""Trusted construction and bounded compilation of one fixed Lean theorem."""
from __future__ import annotations

import os
import re
import resource
import signal
import subprocess
import tempfile
import unicodedata
from pathlib import Path


MAX_PROOF_CHARS = 120_000
MAX_DIAGNOSTIC_CHARS = 12_000
PINNED_MATHLIB_COMMIT = "f0957a7575317490107578ebaee9efaf8e62a4ab"
HEADER = """import Mathlib
import Aesop

set_option maxHeartbeats 400000
set_option maxRecDepth 10000

open BigOperators

"""
FORBIDDEN = re.compile(
    r"(?is)(?:\bsorry\b|\bsorryAx\b|\badmit\b|\baxiom\b|\bunsafe\b|\brun_tac\b|"
    r"\belab\b|\bmacro\b|\bsyntax\b|\bimport\b|\btheorem\b|\blemma\b|"
    r"\bopaque\b|\bexample\b|#\s*(?:eval|check|print|reduce)|"
    r"\b(?:IO|System|Process|FilePath|Socket)\b)"
)


class LeanInfrastructureError(RuntimeError):
    """The trusted Lean toolchain is absent, unreadable, or cannot be launched."""


def _runtime_paths():
    mathlib = Path("/opt/mathlib")
    lean = Path("/home/lean/.elan/toolchains/leanprover--lean4---v4.9.0/bin/lean")
    lean_path_file = Path("/opt/lean.path")
    return mathlib, lean, lean_path_file


def compiler_preflight():
    """Verify the exact trusted compiler before any sealed dataset is read."""
    mathlib, lean, lean_path_file = _runtime_paths()
    commit_file = Path("/opt/mathlib.commit")
    try:
        if not mathlib.is_dir() or not lean.is_file() or not lean_path_file.is_file() \
                or not commit_file.is_file():
            raise LeanInfrastructureError("pinned Lean runtime files are unavailable")
        commit = commit_file.read_text(encoding="utf-8").strip()
        lean_path = lean_path_file.read_text(encoding="utf-8").strip()
        if commit != PINNED_MATHLIB_COMMIT or not lean_path:
            raise LeanInfrastructureError("pinned mathlib identity does not match")
        completed = subprocess.run(
            [str(lean), "--version"], check=True, capture_output=True,
            text=True, timeout=30,
        )
        version = completed.stdout.strip()
        if "version 4.9.0" not in version:
            raise LeanInfrastructureError("pinned Lean version does not match")
        smoke = compile_proof("theorem runtime_preflight_smoke : True :=", "by trivial",
                              timeout_sec=45)
        if not smoke["ok"]:
            raise LeanInfrastructureError("pinned Lean/mathlib smoke compilation failed")
    except LeanInfrastructureError:
        raise
    except (OSError, UnicodeError, subprocess.SubprocessError) as exc:
        raise LeanInfrastructureError(
            f"Lean compiler preflight failed ({type(exc).__name__})"
        ) from exc
    return {"lean_version": version, "mathlib_commit": commit,
            "compiler_verified": True, "mathlib_import_verified": True}


def validate_proof(proof):
    if not isinstance(proof, str):
        return "proof must be a string"
    try:
        proof.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return "proof contains an invalid Unicode scalar"
    proof = unicodedata.normalize("NFKC", proof).strip()
    if not proof.startswith("by") or not re.match(r"^by(?:\s|$)", proof):
        return "proof must begin with the Lean token 'by'"
    if len(proof) > MAX_PROOF_CHARS:
        return "proof exceeds size limit"
    hit = FORBIDDEN.search(proof)
    if hit:
        return f"forbidden proof construct: {hit.group(0)}"
    return ""


def _drop_privileges():
    """Enter the fixed nobody identity, or fail closed while still privileged."""
    if os.geteuid() == 0:
        try:
            os.setgroups([])
            os.setgid(65534)
            os.setuid(65534)
        except OSError as exc:
            raise RuntimeError("could not drop compiler privileges") from exc
    if os.geteuid() == 0 or os.getegid() == 0:
        raise RuntimeError("Lean compiler retained a privileged identity")


def _compiler_setup():
    resource.setrlimit(resource.RLIMIT_CPU, (70, 72))
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**2, 8 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    _drop_privileges()


def compile_proof(formal_statement, proof, timeout_sec=75):
    invalid = validate_proof(proof)
    if invalid:
        return {"ok": False, "diagnostics": invalid}
    if not isinstance(formal_statement, str) or not formal_statement.rstrip().endswith(":="):
        return {"ok": False, "diagnostics": "trusted formal statement is malformed"}
    mathlib, lean, lean_path_file = _runtime_paths()
    if not mathlib.is_dir() or not lean.is_file() or not lean_path_file.is_file():
        raise LeanInfrastructureError("pinned Lean runtime is unavailable")
    code = HEADER + formal_statement.rstrip() + "\n" + proof.strip() + "\n"
    try:
        kernel_home = Path("/tmp/lean-kernel-home")
        kernel_home.mkdir(mode=0o777, parents=True, exist_ok=True)
        kernel_home.chmod(0o777)
        with tempfile.TemporaryDirectory(prefix="lean-kernel-") as tmp:
            root = Path(tmp)
            root.chmod(0o755)
            source = root / "Main.lean"
            source.write_text(code, encoding="utf-8")
            source.chmod(0o644)
            env = {
                "PATH": "/home/lean/.elan/bin:/usr/local/bin:/usr/bin:/bin",
                "HOME": "/tmp/lean-kernel-home",
                "ELAN_HOME": "/home/lean/.elan",
                "LEAN_PATH": lean_path_file.read_text(encoding="utf-8").strip(),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
            }
            proc = subprocess.Popen(
                [str(lean), str(source)],
                cwd=str(mathlib), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                start_new_session=True, preexec_fn=_compiler_setup,
            )
            try:
                output, _ = proc.communicate(timeout=float(timeout_sec))
            except (subprocess.TimeoutExpired, TimeoutError):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                output, _ = proc.communicate()
                return {"ok": False, "diagnostics": "Lean compiler timeout"}
    except LeanInfrastructureError:
        raise
    except (OSError, UnicodeError, subprocess.SubprocessError) as exc:
        raise LeanInfrastructureError(
            f"Lean compiler invocation failed ({type(exc).__name__})"
        ) from exc
    text = output.decode("utf-8", errors="replace")[-MAX_DIAGNOSTIC_CHARS:]
    return {"ok": proc.returncode == 0, "diagnostics": text}
