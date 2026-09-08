"""Run the untrusted child as a deprivileged, isolated subprocess."""
import ctypes
import json
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

from submission_guard import violations

BASE_UID = 10001


def _readonly_copy(source: Path, dst: Path):
    shutil.copytree(source, dst)
    for p in dst.rglob("*"):
        p.chmod(0o555 if p.is_dir() else 0o444)
    dst.chmod(0o555)


def run_case(case: dict, submission_dir: Path, case_index: int, timeout: int) -> dict:
    """Guard, then run child_eval on one case as uid BASE_UID+idx. Fail closed."""
    failures = violations(submission_dir)
    if failures:
        return {"name": case["name"], "ok": False,
                "err": "submission rejected: " + "; ".join(failures[:4])}

    root = Path(tempfile.mkdtemp(prefix=f"tpucase-{case_index:02d}-", dir="/tmp"))
    cand, runtime = root / "submission", root / "runtime"
    uid = BASE_UID + case_index
    try:
        _readonly_copy(submission_dir, cand)
        runtime.mkdir()
        os.chown(runtime, uid, uid)
        runtime.chmod(0o700)
        root.chmod(0o755)  # child (other) needs read+traverse on cwd parent for libtpu runfiles
        env = {
            "HOME": str(runtime), "TMPDIR": str(runtime),
            # libtpu writes driver logs; keep them inside the uid-owned runtime
            "TPU_LOG_DIR": str(runtime), "TEST_TMPDIR": str(runtime),
            "GRPC_VERBOSITY": "NONE",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONPATH": "/tests", "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
            "TPU_SKIP_MDS_QUERY": os.environ.get("TPU_SKIP_MDS_QUERY", ""),
            "TPU_ACCELERATOR_TYPE": os.environ.get("TPU_ACCELERATOR_TYPE", ""),
            "TPU_CHIPS_PER_HOST_BOUNDS": os.environ.get("TPU_CHIPS_PER_HOST_BOUNDS", ""),
            "TPU_HOST_BOUNDS": os.environ.get("TPU_HOST_BOUNDS", ""),
            "TPU_WORKER_ID": os.environ.get("TPU_WORKER_ID", ""),
            "TPU_WORKER_HOSTNAMES": os.environ.get("TPU_WORKER_HOSTNAMES", ""),
            "TPU_VISIBLE_DEVICES": os.environ.get("TPU_VISIBLE_DEVICES", ""),
            "LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS", ""),
        }

        def prepare():
            os.setsid()
            os.umask(0o077)
            try:  # PR_SET_NO_NEW_PRIVS
                ctypes.CDLL(None).prctl(38, 1, 0, 0, 0)
            except Exception:
                pass
            os.setgroups([])
            os.setgid(uid)
            os.setuid(uid)

        proc = subprocess.Popen(
            ["python", "/tests/child_eval.py", json.dumps(case), str(cand / "attention.py")],
            cwd=runtime, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, preexec_fn=prepare)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            return {"name": case["name"], "ok": False, "err": "timed out"}

        line = (out or "").strip().splitlines()
        if not line:
            return {"name": case["name"], "ok": False,
                    "err": f"no result (rc={proc.returncode}): {(err or '')[:200]}"}
        try:
            r = json.loads(line[-1])
        except json.JSONDecodeError:
            return {"name": case["name"], "ok": False, "err": "unparseable child output"}
        r.setdefault("name", case["name"])
        return r
    finally:
        shutil.rmtree(root, ignore_errors=True)
