"""Untrusted per-theorem workflow runner; all capabilities are parent-owned RPC."""
from __future__ import annotations

import contextlib
import errno
import importlib.util
import json
import os
import sys
from pathlib import Path


MAX_LINE = 1_000_000
MAX_PROOF = 120_000


def _send(stream, payload):
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(raw) > MAX_LINE:
        raise RuntimeError("protocol message too large")
    stream.write(raw + "\n")
    stream.flush()


def _recv(stream):
    raw = stream.readline(MAX_LINE + 1)
    if not raw or len(raw) > MAX_LINE:
        raise RuntimeError("invalid parent protocol frame")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError("protocol frame must be an object")
    return value


class RemoteBudgetExceeded(RuntimeError):
    pass


class RPCModel:
    __slots__ = ("reader", "writer")

    def __init__(self, reader, writer):
        self.reader, self.writer = reader, writer

    def __call__(self, messages, max_tokens=1024, stop=None):
        _send(self.writer, {"type": "llm_call", "messages": messages,
                            "max_tokens": max_tokens, "stop": list(stop) if stop else None})
        result = _recv(self.reader)
        if result.get("type") != "llm_response":
            raise RuntimeError("invalid model response")
        if not result.get("ok"):
            if result.get("budget_exceeded"):
                raise RemoteBudgetExceeded(str(result.get("error", "budget exceeded")))
            raise RuntimeError(str(result.get("error", "model call failed")))
        if not isinstance(result.get("text"), str):
            raise RuntimeError("model response must be text")
        return result["text"]


class RPCLean:
    __slots__ = ("reader", "writer")

    def __init__(self, reader, writer):
        self.reader, self.writer = reader, writer

    def __call__(self, proof_body):
        _send(self.writer, {"type": "lean_check", "proof": proof_body})
        result = _recv(self.reader)
        if result.get("type") != "lean_response":
            raise RuntimeError("invalid Lean response")
        if not result.get("rpc_ok"):
            if result.get("budget_exceeded"):
                raise RemoteBudgetExceeded(str(result.get("error", "budget exceeded")))
            raise RuntimeError(str(result.get("error", "Lean check failed")))
        return {"ok": bool(result.get("ok")), "diagnostics": str(result.get("diagnostics", ""))}


def _inside(path, root):
    return path == root or root in path.parents


def _write_open(args):
    mode = args[1] if len(args) > 1 else "r"
    flags = args[2] if len(args) > 2 else 0
    mode_write = isinstance(mode, str) and any(token in mode for token in "wax+")
    flag_write = isinstance(flags, int) and bool(
        flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
    )
    return mode_write or flag_write


def _install_audit_hook(submission_dir, sandbox_dir):
    sealed = tuple(Path(x).resolve() for x in
                   ("/tests", "/logs", "/proc", "/sys", "/root", "/opt/mathlib"))
    shared_tmp = Path("/tmp").resolve()
    blocked_imports = {"socket", "requests", "urllib", "http", "httpx", "aiohttp",
                       "subprocess", "multiprocessing", "ctypes", "_ctypes", "pyseccomp",
                       }
    blocked_events = {"subprocess.Popen", "os.system", "os.posix_spawn", "os.posix_spawnp",
                      "os.fork", "os.forkpty", "os.exec", "os.kill", "os.killpg", "pty.spawn"}
    mutating_path_events = {
        "os.remove", "os.rename", "os.renames", "os.replace", "os.rmdir", "os.mkdir",
        "os.makedirs", "os.chmod", "os.chown", "os.link", "os.symlink", "os.truncate",
        "os.utime",
    }

    def direct_from_submission():
        try:
            frame = sys._getframe(2)
        except (ValueError, AttributeError):
            return True
        while frame is not None:
            name = frame.f_code.co_filename
            if name and not name.startswith("<frozen importlib"):
                try:
                    path = Path(name).resolve()
                except Exception:
                    return True
                if path == submission_dir or submission_dir in path.parents:
                    return True
                if path != Path(__file__).resolve():
                    return False
            frame = frame.f_back
        return True

    def hook(event, args):
        if event in blocked_events or event.startswith("socket."):
            raise PermissionError(f"operation blocked: {event}")
        if event == "import" and args:
            root = str(args[0]).split(".", 1)[0]
            if root in blocked_imports and direct_from_submission():
                raise PermissionError(f"import blocked: {root}")
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if any(path == root or root in path.parents for root in sealed):
                raise PermissionError(f"sealed path blocked: {path}")
            if _inside(path, shared_tmp) and not (
                    _inside(path, sandbox_dir) or _inside(path, submission_dir)):
                raise PermissionError("shared temporary paths are blocked")
            if _write_open(args) and not _inside(path, sandbox_dir):
                raise PermissionError("writes outside the per-question sandbox are blocked")
        if event in mutating_path_events:
            for value in args[:2]:
                if isinstance(value, (str, bytes, os.PathLike)):
                    path = Path(os.fsdecode(value)).resolve()
                    if not _inside(path, sandbox_dir):
                        raise PermissionError("filesystem mutation outside the sandbox is blocked")
        if event in {"os.listdir", "os.scandir", "os.chdir"} and args \
                and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if _inside(path, shared_tmp) and not (
                    _inside(path, sandbox_dir) or _inside(path, submission_dir)):
                raise PermissionError("shared temporary paths are blocked")

    sys.addaudithook(hook)


def _install_seccomp_if_required():
    """Deny networking, process creation, namespace changes, and cross-process access."""
    if os.environ.get("LEAN_REQUIRE_SECCOMP") != "1":
        return
    try:
        import pyseccomp as seccomp
        syscall_filter = seccomp.SyscallFilter(defaction=seccomp.ALLOW)
        denied = (
            "socket", "socketpair", "connect", "bind", "listen", "accept", "accept4",
            "clone", "clone3", "fork", "vfork", "execve", "execveat", "ptrace",
            "process_vm_readv", "process_vm_writev", "pidfd_open", "pidfd_getfd",
            "mount", "umount2", "pivot_root", "setns", "unshare", "bpf",
            "perf_event_open", "keyctl", "open_by_handle_at", "name_to_handle_at",
            "io_uring_setup", "io_uring_enter", "io_uring_register",
            "shmget", "shmat", "shmdt", "shmctl", "msgget", "msgsnd", "msgrcv",
            "msgctl", "semget", "semop", "semtimedop", "semctl", "ipc",
            "mq_open", "mq_unlink", "mq_timedsend", "mq_timedreceive", "mq_notify",
            "mq_getsetattr", "kill", "tkill", "tgkill",
        )
        for name in denied:
            try:
                syscall_filter.add_rule(seccomp.ERRNO(errno.EPERM), name)
            except (KeyError, RuntimeError, ValueError):
                pass
        syscall_filter.load()
    except Exception as exc:
        raise RuntimeError(
            f"required seccomp filter unavailable: {type(exc).__name__}: {exc}"
        ) from exc


def _install_landlock_if_required(submission_dir, sandbox_dir):
    """Restrict the child to read-only runtime files and one writable question sandbox."""
    if os.environ.get("LEAN_REQUIRE_SECCOMP") != "1":
        return
    ctypes = None
    ruleset_fd = -1
    try:
        ctypes = __import__("ctypes")

        class RulesetAttr(ctypes.Structure):
            _fields_ = [("handled_access_fs", ctypes.c_uint64)]

        class PathBeneathAttr(ctypes.Structure):
            _fields_ = [("allowed_access", ctypes.c_uint64),
                        ("parent_fd", ctypes.c_int32),
                        ("reserved", ctypes.c_uint32)]

        create_ruleset, add_rule, restrict_self = 444, 445, 446
        create_version, path_beneath = 1, 1
        execute = 1 << 0
        write_file = 1 << 1
        read_file = 1 << 2
        read_dir = 1 << 3
        remove_dir = 1 << 4
        remove_file = 1 << 5
        make_char = 1 << 6
        make_dir = 1 << 7
        make_reg = 1 << 8
        make_sock = 1 << 9
        make_fifo = 1 << 10
        make_block = 1 << 11
        make_sym = 1 << 12
        refer = 1 << 13
        truncate = 1 << 14
        ioctl_dev = 1 << 15

        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        abi = libc.syscall(create_ruleset, ctypes.c_void_p(), 0, create_version)
        if abi < 3:
            raise RuntimeError(f"Landlock ABI >= 3 is required, found {abi}")
        handled = (execute | write_file | read_file | read_dir | remove_dir | remove_file |
                   make_char | make_dir | make_reg | make_sock | make_fifo | make_block |
                   make_sym | refer | truncate)
        if abi >= 5:
            handled |= ioctl_dev
        ruleset = RulesetAttr(handled)
        ruleset_fd = libc.syscall(create_ruleset, ctypes.byref(ruleset),
                                  ctypes.sizeof(ruleset), 0)
        if ruleset_fd < 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))

        def allow(path, access):
            candidate = Path(path)
            if not candidate.exists():
                return
            flags = getattr(os, "O_PATH", os.O_RDONLY) | os.O_CLOEXEC
            parent_fd = os.open(candidate, flags)
            try:
                permitted = access
                if not candidate.is_dir():
                    permitted &= ~(read_dir | remove_dir | make_char | make_dir | make_reg |
                                   make_sock | make_fifo | make_block | make_sym | refer)
                rule = PathBeneathAttr(permitted, parent_fd, 0)
                if libc.syscall(add_rule, ruleset_fd, path_beneath,
                                ctypes.byref(rule), 0) != 0:
                    code = ctypes.get_errno()
                    raise OSError(code, os.strerror(code))
            finally:
                os.close(parent_fd)

        readonly = execute | read_file | read_dir
        for path in ("/usr", "/lib", "/lib64", "/opt/venv", "/etc",
                     "/dev/null", "/dev/urandom", "/dev/random", submission_dir):
            allow(path, readonly)
        allow(sandbox_dir, handled)

        if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        if libc.syscall(restrict_self, ruleset_fd, 0) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
    except Exception as exc:
        raise RuntimeError(
            f"required Landlock isolation unavailable: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if ruleset_fd >= 0:
            os.close(ruleset_fd)
        if ctypes is not None:
            for name in tuple(sys.modules):
                if (name == "ctypes" or name.startswith("ctypes.") or name == "_ctypes"
                        or name == "pyseccomp" or name.startswith("pyseccomp.")):
                    sys.modules.pop(name, None)


def _load(submission_dir):
    source = submission_dir / "solver.py"
    if not source.is_file():
        raise RuntimeError("submission must contain solver.py")
    sys.path.insert(0, str(submission_dir))
    spec = importlib.util.spec_from_file_location("submitted_solver", source)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    entry = getattr(module, "answer_batch", None)
    if not callable(entry):
        raise RuntimeError("solver.py must expose answer_batch(examples, llm, lean, budget)")
    return entry


def _run_isolation_preflight(submission_dir, sandbox_dir, sealed_probe):
    """Kernel-boundary probe used by the trusted parent before it reads sealed data."""
    _install_seccomp_if_required()
    _install_landlock_if_required(submission_dir, sandbox_dir)
    network_blocked = False
    sealed_blocked = False
    try:
        socket_module = __import__("socket")
        handle = socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM)
    except OSError as exc:
        network_blocked = exc.errno in {errno.EPERM, errno.EACCES}
    else:
        handle.close()
    try:
        descriptor = os.open(sealed_probe, os.O_RDONLY)
    except OSError as exc:
        sealed_blocked = exc.errno in {errno.EPERM, errno.EACCES}
    else:
        os.close(descriptor)
    if not network_blocked or not sealed_blocked:
        raise RuntimeError("required kernel isolation probe did not deny access")
    _send(sys.stdout, {"type": "isolation_preflight", "ok": True,
                       "seccomp": True, "landlock": True,
                       "uid": os.geteuid(), "gid": os.getegid()})


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "--isolation-preflight":
        _run_isolation_preflight(
            Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve(),
            Path(sys.argv[4]).resolve(),
        )
        return
    if len(sys.argv) != 3:
        raise RuntimeError("usage: child_runner.py SUBMISSION_DIR SANDBOX_DIR")
    submission_dir = Path(sys.argv[1]).resolve()
    sandbox_dir = Path(sys.argv[2]).resolve()
    if not sandbox_dir.is_dir():
        raise RuntimeError("per-question sandbox is missing")
    reader, writer = sys.stdin, sys.stdout
    request = _recv(reader)
    if request.get("type") != "start" or not isinstance(request.get("example"), dict):
        raise RuntimeError("invalid start frame")
    _install_seccomp_if_required()
    _install_landlock_if_required(submission_dir, sandbox_dir)
    _install_audit_hook(submission_dir, sandbox_dir)
    with contextlib.redirect_stdout(sys.stderr):
        entry = _load(submission_dir)
        values = entry([request["example"]], RPCModel(reader, writer), RPCLean(reader, writer),
                       dict(request.get("budget", {})))
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], str):
        raise RuntimeError("answer_batch must return one proof string")
    if len(values[0]) > MAX_PROOF:
        raise RuntimeError("proof exceeds size limit")
    _send(writer, {"type": "result", "proof": values[0]})


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        try:
            _send(sys.stdout, {"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
        raise
