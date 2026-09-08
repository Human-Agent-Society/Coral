"""Shared trusted evaluator for visible self-check and sealed hidden grading."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import resource
import select
import signal
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from lean_compiler import (MAX_PROOF_CHARS, LeanInfrastructureError,
                           compile_proof, compiler_preflight)
from model_proxy import BudgetExceeded, ModelInfrastructureError, ModelProxy


PROTOCOL_VERSION = "lean-proof-workflow-v2.5-gpt54-fixed-statement"
SUBMISSION_DIR = Path("/app/methods/main")
CHILD_RUNNER = Path(__file__).resolve().with_name("child_runner.py")
BUDGET = {
    "max_llm_calls_per_question": 3,
    "max_total_tokens_per_question": 12000,
    "max_lean_checks_per_question": 4,
    "max_wall_time_per_question_sec": 180,
}
EXPECTED_BUDGET = {
    "max_llm_calls_per_question": 3,
    "max_total_tokens_per_question": 12000,
    "max_lean_checks_per_question": 4,
    "max_wall_time_per_question_sec": 180,
}
REQUIRE_KERNEL_ISOLATION = Path("/tests/heldout").is_dir()
MAX_LINE = 1_000_000
MAX_SOURCE_BYTES = 500_000
MAX_PROMPT_CHARS = 200_000
BLOCKED_IMPORTS = {"socket", "requests", "urllib", "http", "httpx", "aiohttp", "subprocess",
                   "multiprocessing", "ctypes", "_ctypes", "pyseccomp", "os",
                   "pathlib", "importlib"}
BLOCKED_CALLS = {"breakpoint", "eval", "exec", "compile", "open",
                 "builtins.breakpoint", "builtins.eval", "builtins.exec", "builtins.compile",
                 "builtins.open", "__import__"}
SEALED_LITERALS = ("/tests", "heldout", "calibration.jsonl", "source_id", "gold.jsonl")


def load_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("dataset must contain JSON objects")
    ids = [row.get("id") for row in rows]
    if len(ids) != len(set(ids)) or not all(isinstance(value, str) and value for value in ids):
        raise ValueError("dataset IDs must be non-empty and unique")
    required = {"id", "problem", "formal_statement", "difficulty", "subtopic", "stratum"}
    for row in rows:
        if not required.issubset(row) or not row["formal_statement"].rstrip().endswith(":="):
            raise ValueError(f"malformed dataset row: {row.get('id')}")
    return rows


def dataset_digest(rows):
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")) for row in rows).encode()
    return hashlib.sha256(payload).hexdigest()


def tree_digest(root):
    digest = hashlib.sha256()
    root = Path(root)
    for path in sorted(x for x in root.rglob("*") if x.is_file()
                       and "__pycache__" not in x.parts and path_suffix_ok(x)):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def path_suffix_ok(path):
    return path.suffix not in {".pyc", ".pyo"}


def _dotted_name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def static_guard(submission_dir=SUBMISSION_DIR):
    try:
        return _static_guard(submission_dir)
    except (OSError, UnicodeError) as exc:
        return f"submission tree is unreadable ({type(exc).__name__})"


def _static_guard(submission_dir=SUBMISSION_DIR):
    submission_dir = Path(submission_dir)
    if not (submission_dir / "solver.py").is_file():
        return "submission must contain solver.py"
    total = 0
    for path in submission_dir.rglob("*"):
        if path.is_symlink():
            return f"symlink is not allowed: {path.name}"
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            return f"only regular files are allowed: {path.name}"
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        total += path.stat().st_size
        if path.stat().st_size > MAX_SOURCE_BYTES or total > MAX_SOURCE_BYTES:
            return "submission exceeds 500 KB"
        if path.suffix != ".py":
            return f"only Python source is allowed: {path.name}"
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            return f"syntax error in {path.name}: {exc.msg}"
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                lowered = node.value.casefold()
                for literal in SEALED_LITERALS:
                    if literal.casefold() in lowered:
                        return f"sealed-data reference in {path.name}: {literal}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".", 1)[0] in BLOCKED_IMPORTS:
                        return f"forbidden import in {path.name}: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".", 1)[0]
                if root in BLOCKED_IMPORTS:
                    return f"forbidden import in {path.name}: {node.module}"
            elif isinstance(node, ast.Call) and _dotted_name(node.func) in BLOCKED_CALLS:
                return f"forbidden call in {path.name}: {_dotted_name(node.func)}"
            elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                    and node.slice.value == "id":
                return f"problem-ID lookup is forbidden in {path.name}"
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "get" and node.args \
                    and isinstance(node.args[0], ast.Constant) and node.args[0].value == "id":
                return f"problem-ID lookup is forbidden in {path.name}"
    return ""


def _drop_privileges():
    """Enter the fixed nobody identity, or fail closed while still privileged."""
    if os.geteuid() == 0:
        try:
            os.setgroups([])
            os.setgid(65534)
            os.setuid(65534)
        except OSError as exc:
            raise RuntimeError("could not drop child privileges") from exc
    if os.geteuid() == 0 or os.getegid() == 0:
        raise RuntimeError("child process retained a privileged identity")


def _child_setup():
    wall = max(1, BUDGET["max_wall_time_per_question_sec"])
    resource.setrlimit(resource.RLIMIT_CPU, (wall, wall + 2))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024**2, 4 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    _drop_privileges()


def _child_env(home, sandbox):
    return {"PATH": "/home/lean/.elan/bin:/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "HOME": str(home),
            "TMPDIR": str(sandbox), "TMP": str(sandbox), "TEMP": str(sandbox),
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
            "LEAN_REQUIRE_SECCOMP": "1" if REQUIRE_KERNEL_ISOLATION else "0",
            "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8", "NUMEXPR_NUM_THREADS": "8"}


def _kill(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass


class IsolationInfrastructureError(RuntimeError):
    """The required seccomp/Landlock child boundary is unavailable."""


def isolation_preflight():
    """Exercise both kernel boundaries before any sealed dataset is read."""
    if not REQUIRE_KERNEL_ISOLATION:
        return
    with tempfile.TemporaryDirectory(prefix="lean-isolation-preflight-") as tmp:
        root = Path(tmp)
        root.chmod(0o711)
        submission = root / "submission"
        sandbox = root / "sandbox"
        sealed_probe = root / "sealed-probe"
        submission.mkdir(mode=0o755)
        sandbox.mkdir(mode=0o700)
        sealed_probe.write_text("sealed", encoding="utf-8")
        sealed_probe.chmod(0o644)
        if os.geteuid() == 0:
            os.chown(sandbox, 65534, 65534)
        try:
            proc = subprocess.Popen(
                [sys.executable, str(CHILD_RUNNER), "--isolation-preflight",
                 str(submission), str(sandbox), str(sealed_probe)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=sandbox,
                env=_child_env(sandbox, sandbox), start_new_session=True,
                preexec_fn=_child_setup,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise IsolationInfrastructureError(
                f"could not start kernel isolation preflight ({type(exc).__name__})"
            ) from exc
        try:
            stdout, stderr = proc.communicate(timeout=20)
        except subprocess.TimeoutExpired as exc:
            _kill(proc)
            raise IsolationInfrastructureError("kernel isolation preflight timed out") from exc
        if proc.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace")[-500:]
            raise IsolationInfrastructureError(
                f"kernel isolation preflight failed ({proc.returncode}): {detail}"
            )
        try:
            frame = json.loads(stdout)
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise IsolationInfrastructureError("invalid kernel isolation preflight result") from exc
        expected_keys = {"type", "ok", "seccomp", "landlock", "uid", "gid"}
        if set(frame) != expected_keys or frame.get("type") != "isolation_preflight" \
                or frame.get("ok") is not True or frame.get("seccomp") is not True \
                or frame.get("landlock") is not True \
                or not isinstance(frame.get("uid"), int) \
                or not isinstance(frame.get("gid"), int) \
                or frame["uid"] == 0 or frame["gid"] == 0:
            raise IsolationInfrastructureError(
                "kernel isolation preflight did not attest both layers and low privilege"
            )
    return {"isolation_verified": True, "isolation_backend": "seccomp+landlock",
            "isolated_uid": frame["uid"], "isolated_gid": frame["gid"]}


def _write(proc, payload):
    raw = (json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False) + "\n").encode()
    if len(raw) > MAX_LINE:
        raise RuntimeError("protocol request too large")
    proc.stdin.write(raw)
    proc.stdin.flush()


def _read(proc, buffer, deadline):
    fd = proc.stdout.fileno()
    while b"\n" not in buffer:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("per-problem wall limit exceeded")
        readable, _, _ = select.select([fd], [], [], min(remaining, 0.25))
        if not readable:
            if proc.poll() is not None:
                raise RuntimeError(f"child exited early ({proc.returncode})")
            continue
        chunk = os.read(fd, 65536)
        if not chunk:
            raise RuntimeError("child closed protocol stream")
        buffer.extend(chunk)
        if len(buffer) > MAX_LINE:
            raise RuntimeError("protocol response too large")
    raw, rest = bytes(buffer).split(b"\n", 1)
    buffer[:] = rest
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key is forbidden: {key}")
            value[key] = item
        return value

    value = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)
    _validate_unicode_scalars(value)
    if not isinstance(value, dict):
        raise RuntimeError("protocol response must be an object")
    return value


def _validate_unicode_scalars(value):
    if isinstance(value, str):
        value.encode("utf-8", errors="strict")
    elif isinstance(value, list):
        for item in value:
            _validate_unicode_scalars(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_unicode_scalars(key)
            _validate_unicode_scalars(item)


def _validate_model_call(message):
    messages = message.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 64:
        raise ValueError("messages must be a non-empty list")
    clean, chars = [], 0
    for item in messages:
        if not isinstance(item, dict) or item.get("role") not in {"system", "user", "assistant"} \
                or not isinstance(item.get("content"), str):
            raise ValueError("invalid chat message")
        chars += len(item["content"])
        clean.append({"role": item["role"], "content": item["content"]})
    if chars > MAX_PROMPT_CHARS:
        raise ValueError("prompt too large")
    max_tokens = message.get("max_tokens", 1024)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) \
            or not 1 <= max_tokens <= BUDGET["max_total_tokens_per_question"]:
        raise ValueError("invalid max_tokens")
    stop = message.get("stop")
    if stop is not None and (not isinstance(stop, list) or len(stop) > 8
                             or not all(isinstance(x, str) and len(x) <= 200 for x in stop)):
        raise ValueError("invalid stop sequences")
    return clean, max_tokens, stop


@contextmanager
def _alarm(seconds):
    old_handler = signal.getsignal(signal.SIGALRM)

    def handler(signum, frame):
        raise TimeoutError("capability call exceeded remaining wall time")

    signal.signal(signal.SIGALRM, handler)
    old_timer = signal.setitimer(signal.ITIMER_REAL, max(0.001, seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
        signal.signal(signal.SIGALRM, old_handler)


def run_question(example, submission_dir=SUBMISSION_DIR):
    clean = {key: example[key] for key in
             ("problem", "formal_statement", "difficulty", "subtopic", "stratum")}
    clean["id"] = "opaque-problem"
    model = ModelProxy(BUDGET["max_llm_calls_per_question"],
                       BUDGET["max_total_tokens_per_question"])
    deadline = time.monotonic() + BUDGET["max_wall_time_per_question_sec"]
    started = time.monotonic()
    checks, over, status, error, proof = 0, False, "error", "", ""
    buffer = bytearray()
    diagnostics = ""
    with tempfile.TemporaryDirectory(prefix="lean-submission-") as tmp, \
            tempfile.TemporaryFile(mode="w+b") as stderr_file:
        root = Path(tmp)
        root.chmod(0o711)
        child_submission = root / "main"
        shutil.copytree(submission_dir, child_submission,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        for path in child_submission.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        child_submission.chmod(0o755)
        sandbox = root / "sandbox"
        home = sandbox / "home"
        sandbox.mkdir(mode=0o700)
        home.mkdir(mode=0o700)
        if os.geteuid() == 0:
            os.chown(sandbox, 65534, 65534)
            os.chown(home, 65534, 65534)
        proc = subprocess.Popen(
            [sys.executable, str(CHILD_RUNNER), str(child_submission), str(sandbox)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_file, cwd=sandbox,
            env=_child_env(home, sandbox), bufsize=0,
            start_new_session=True, preexec_fn=_child_setup,
        )
        try:
            _write(proc, {"type": "start", "example": clean, "budget": BUDGET})
            while True:
                message = _read(proc, buffer, deadline)
                kind = message.get("type")
                if kind == "llm_call":
                    if set(message) != {"type", "messages", "max_tokens", "stop"}:
                        raise ValueError("llm_call has unexpected or missing fields")
                    try:
                        args = _validate_model_call(message)
                        with _alarm(deadline - time.monotonic()):
                            text = model(*args)
                        _write(proc, {"type": "llm_response", "ok": True, "text": text})
                    except BudgetExceeded as exc:
                        over = True
                        _write(proc, {"type": "llm_response", "ok": False,
                                      "budget_exceeded": True, "error": str(exc)})
                    except ModelInfrastructureError:
                        raise
                    except Exception as exc:
                        _write(proc, {"type": "llm_response", "ok": False,
                                      "budget_exceeded": False,
                                      "error": f"{type(exc).__name__}: {exc}"})
                        if isinstance(exc, (TimeoutError, ValueError)):
                            raise
                elif kind == "lean_check":
                    if set(message) != {"type", "proof"}:
                        raise ValueError("lean_check has unexpected or missing fields")
                    checks += 1
                    if checks > BUDGET["max_lean_checks_per_question"]:
                        over = True
                        _write(proc, {"type": "lean_response", "rpc_ok": False,
                                      "budget_exceeded": True, "error": "Lean-check budget exceeded"})
                        continue
                    candidate = message.get("proof")
                    if not isinstance(candidate, str) or len(candidate) > MAX_PROOF_CHARS:
                        raise ValueError("invalid Lean-check proof")
                    with _alarm(deadline - time.monotonic()):
                        result = compile_proof(example["formal_statement"], candidate,
                                               timeout_sec=min(75, deadline - time.monotonic()))
                    _write(proc, {"type": "lean_response", "rpc_ok": True, **result})
                elif kind == "result":
                    if set(message) != {"type", "proof"}:
                        raise ValueError("result has unexpected or missing fields")
                    proof = message.get("proof")
                    if not isinstance(proof, str) or len(proof) > MAX_PROOF_CHARS:
                        raise ValueError("invalid final proof")
                    break
                elif kind == "error":
                    if set(message) != {"type", "error"}:
                        raise ValueError("error frame has unexpected or missing fields")
                    raise RuntimeError(str(message.get("error", "child error")))
                else:
                    raise RuntimeError("unexpected protocol message")
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("child did not exit after its result") from exc
            trailing = bytes(buffer)
            while True:
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    break
                trailing += chunk
                if len(trailing) > MAX_LINE:
                    raise RuntimeError("trailing protocol output is too large")
            if trailing:
                raise RuntimeError("trailing protocol output after result")
            if proc.returncode != 0:
                raise RuntimeError(f"child exited nonzero after result ({proc.returncode})")
            status = "ok"
        except TimeoutError as exc:
            status, error, over = "timeout", str(exc), True
        except (ModelInfrastructureError, LeanInfrastructureError):
            raise
        except Exception as exc:
            status, error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            _kill(proc)
            for stream in (proc.stdin, proc.stdout):
                if stream:
                    stream.close()
            if model.calls_exceeded or model.tokens_exceeded:
                over = True
            stderr_file.seek(0, os.SEEK_END)
            size = stderr_file.tell()
            stderr_file.seek(max(0, size - 4000))
            diagnostics = stderr_file.read().decode("utf-8", errors="replace")[-1000:]
    final = {"ok": False, "diagnostics": "not compiled"}
    if status == "ok" and not over:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            status, error, over = "timeout", "per-problem wall limit exceeded", True
        else:
            try:
                with _alarm(remaining):
                    final = compile_proof(
                        example["formal_statement"], proof, timeout_sec=min(75, remaining)
                    )
            except TimeoutError:
                status, error, over = "timeout", "per-problem wall limit exceeded", True
                final = {"ok": False, "diagnostics": "Lean compiler timeout"}
            if time.monotonic() > deadline:
                status, error, over = "timeout", "per-problem wall limit exceeded", True
                final = {"ok": False, "diagnostics": "Lean compiler exceeded wall limit"}
    return {"correct": bool(final["ok"] and status == "ok" and not over), "status": status,
            "error": error, "budget_exceeded": bool(over), "calls": model.num_calls,
            "tokens": model.total_tokens, "lean_checks": checks,
            "wall_sec": round(time.monotonic() - started, 3),
            "diagnostics": diagnostics, "final_diagnostics": final.get("diagnostics", "")[-2000:],
            "proof_sha256": hashlib.sha256(proof.encode()).hexdigest() if proof else "",
            "proof_excerpt": proof[-4000:]}


def model_preflight():
    if BUDGET != EXPECTED_BUDGET:
        raise ModelInfrastructureError("runtime budgets do not match the frozen protocol")
    return ModelProxy(BUDGET["max_llm_calls_per_question"],
                      BUDGET["max_total_tokens_per_question"]).healthcheck()


def runtime_preflight():
    identity = model_preflight()
    identity.update(compiler_preflight())
    identity.update(isolation_preflight())
    return identity


def evaluate(rows, submission_dir=SUBMISSION_DIR, model_identity=None):
    invalid = static_guard(submission_dir)
    if invalid:
        return {"accuracy_pct": 0.0, "n_correct": 0, "n_total": len(rows),
                "invalid": True, "error": invalid, "per_problem": []}
    if model_identity is None:
        model_identity = runtime_preflight()
    per = []
    for index, row in enumerate(rows, 1):
        result = run_question(row, submission_dir)
        result.update({"id": row["id"], "stratum": row["stratum"],
                       "difficulty": row["difficulty"]})
        per.append(result)
        print(f"[{index:02d}/{len(rows)}] {row['id']}: correct={result['correct']} "
              f"calls={result['calls']} checks={result['lean_checks']} wall={result['wall_sec']:.1f}s",
              flush=True)
    n_correct = sum(int(item["correct"]) for item in per)
    accuracy = 100.0 * n_correct / len(rows)
    strata = {}
    for name in sorted({row["stratum"] for row in rows}):
        items = [item for item in per if item["stratum"] == name]
        strata[name] = {"n_correct": sum(int(x["correct"]) for x in items), "n_total": len(items)}
    return {"accuracy_pct": round(accuracy, 6), "n_correct": n_correct,
            "n_total": len(rows), "invalid": False, "error": "", "strata": strata,
            "per_problem": per, "artifact_sha256": tree_digest(submission_dir),
            "dataset_sha256": dataset_digest(rows), "protocol_version": PROTOCOL_VERSION,
            **model_identity,
            "mathlib_commit": Path("/opt/mathlib.commit").read_text().strip()
            if Path("/opt/mathlib.commit").exists() else "unavailable"}
