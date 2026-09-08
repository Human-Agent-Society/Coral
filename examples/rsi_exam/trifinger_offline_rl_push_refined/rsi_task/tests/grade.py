"""TWO-PHASE Harbor verifier (PARENT / trusted side) for TriFinger offline-RL push.

ARTIFACT-EVAL: the agent already trained a policy and saved a checkpoint under /app/submission/model;
the graded directory /app/methods/main/ holds the agent's submitted solver.py. This verifier RE-RUNS
the submitted Policy.get_action() on SEALED held-out episodes the agent never saw (it does NOT
re-train), so untrusted code executes inside the verifier container. Per docs/anti-cheat.md this uses
the two-phase pattern:

    PARENT (this file, trusted — NEVER imports the submission):
      0. as root, read the sealed material this run needs (anchors.json, heldout_seeds.py) INTO
         MEMORY — everything sealed is 0400 root, so the child can never open it itself,
      1. spawn a CHILD (`python3 child_solve.py`, a fresh process in its own session/group, DROPPED
         to the unprivileged uid/gid 4242 `solver`) that loads the submitted Policy + checkpoint and
         runs it on each sealed held-out episode to record ONLY the per-episode ACTION TRACE (plain
         data); the episode seeds are handed to it on stdin,
      2. SIGKILL the whole child group, then score with the sealed replay_score.py, which replays each
         trace on a FRESH env and recomputes the return — the policy code never runs in this trusted
         step, so it cannot fake the metric or monkeypatch the scorer,
      3. reward = mean-return mapped through the 3 anchors (from tests/anchors.json) by THIS file's
         reward_of() — landing 0 / 0.6 / 1.0; any failure -> 0.

Why the child is de-privileged (four separate holes close with this one change):
  A  it can no longer write /logs/verifier/reward.* nor signal the root parent, so "seed a 1.0 and
     then SIGKILL the grader" stops working at the source;
  B  it can no longer overwrite /tests/replay_score.py — a file the trusted parent executes LATER;
  D  /proc/1/environ belongs to root and is unreadable at uid 4242;
  E  the sealed 0400 files (anchors.json, heldout_seeds.py, replay_score.py) are unreadable to it,
     while root reads them regardless of mode.
test.sh's pre-clean + exit-code gate and the `except BaseException` at the bottom of this file are
defence layers two and three.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# reward.json must hold finite numbers only; text diagnostics live elsewhere.
_RJ_TEXT_KEYS = ("error", "errors", "status", "reason", "detail", "details",
                 "message", "messages", "_details", "note", "notes")


def _reward_json_numeric(payload, grader_failed=None):
    """reward.json payload with finite numbers only (bool -> int, non-numeric keys dropped)."""
    import math as _math

    out = {}
    flags = {}

    def _flatten(v):
        if isinstance(v, str):
            return v
        if isinstance(v, (list, tuple)):
            return " | ".join(_flatten(x) for x in v)
        if isinstance(v, dict):
            return " | ".join(_flatten(x) for x in v.values())
        return "" if v is None else str(v)

    def _strings(v):
        """Flatten to individual strings — prefix matching must not join them."""
        if isinstance(v, str):
            yield v
        elif isinstance(v, (list, tuple)):
            for _x in v:
                yield from _strings(_x)
        elif isinstance(v, dict):
            for _x in v.values():
                yield from _strings(_x)
        elif v is not None:
            yield str(v)

    _marker_texts = []

    for k, v in (payload or {}).items():
        if isinstance(v, bool):
            out[k] = int(v)
            continue
        if isinstance(v, (int, float)) and _math.isfinite(v):
            out[k] = v
            continue
        # Non-numeric: drop the key, but distil it into numeric flags first.
        if k in _RJ_TEXT_KEYS:
            low = _flatten(v).lower()
            _marker_texts.extend(_strings(v))
            if "timed out" in low or "timeout" in low:
                flags["submission_timed_out"] = 1
            if "returncode" in low or "killed by signal" in low:
                flags["submission_child_failed"] = 1

    # Only a prefix the grader itself minted at string start counts.
    if grader_failed is None:
        _pfx = (globals().get("GRADER_FAILED_PREFIX")
                or globals().get("GRADER_FAIL_PREFIX")
                or "grader failed: ")
        # Two gates: prefix at string start, and the grader phase.
        grader_failed = (globals().get("_PHASE", "grader") == "grader"
                         and any(s.startswith(_pfx) for s in _marker_texts))
    if grader_failed:
        flags["grader_failed"] = 1

    out.update(flags)
    if "reward" not in out:
        out["reward"] = 0.0
    return out


sys.dont_write_bytecode = True  # never leave a readable __pycache__ copy of a sealed module in /tests


# Phase decides attribution: grader-side failures get the `grader failed:` prefix.
_PHASE = "grader"


def _set_phase(name):
    """Statement-form phase switch."""
    global _PHASE
    _PHASE = name


class _phase:
    """`with _phase("submission"):` — exceptions inside are submission-side."""

    def __init__(self, name):
        self.name = name
        self.prev = None

    def __enter__(self):
        global _PHASE
        self.prev = _PHASE
        _PHASE = self.name
        return self

    def __exit__(self, exc_type, exc, tb):
        global _PHASE
        # Do not restore while an exception propagates: the raise site owns the phase.
        if exc_type is None:
            _PHASE = self.prev
        return False

# hard-pin the BLAS/OpenMP pools for THIS process and everything it spawns —
# in particular the sealed scorer subprocess, which inherits os.environ. Constants on purpose: read
# from os.environ (even with a default) and the grading host's core count silently changes the score.
# Set before numpy/torch are imported anywhere in the chain.
# The constant is 4, not 1 -- it must equal this task's [verifier] cpus. What the pinning
# requires is that the thread count not be a function of the HOST; it never needed to be 1,
# and 1 measurably broke this task (agent-side training 7x slower than its own 4.5 h budget allows).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "4"

TESTS = Path(__file__).resolve().parent
SUBMISSION_DIR = Path("/app/methods/main")
MODEL_DIR = "/app/submission/model"   # constant, not host-overridable
REWARD_DIR = Path("/logs/verifier")
CHILD = TESTS / "child_solve.py"
SCORER = TESTS / "replay_score.py"
ANCHOR_FILE = TESTS / "anchors.json"
ANCHOR_KEYS = ("BASELINE", "FRONTIER", "UPPER_BOUND")

CHILD_TIMEOUT = 3600.0   # was 1800.0
SCORE_TIMEOUT = 1800.0   # was 1200.0

# The trace gates below run on the SUBMISSION side: a rejected trace is a real 0.
ACT_DIM = 9                        # == trifinger_score.ACT_DIM
MAX_TRACES_BYTES = 64 * 1024 * 1024
MAX_TRACE_STEPS = 1500             # 2x the 750-step episode length
INVALID_TRACE_RC = 3               # replay_score.py: invalid trace data -> submission side

# Dedicated unprivileged account the untrusted child is dropped to (created in tests/Dockerfile).
SOLVER_UID = 4242
SOLVER_GID = 4242


# ---------------------------------------------------------------------------
# artifact readability gate (run as ROOT, BEFORE the privilege drop)
#
# The graded artifacts (task.toml `artifacts`) are handed to us by the agent
# side with whatever ownership/mode that side happened to leave. The child that
# actually runs them is dropped to SOLVER_UID, so if the child cannot read them
# the failure is SILENT: no output file -> "solver produced no predictions" ->
# reward 0, with nothing anywhere saying "permission denied". That is
# indistinguishable from a genuinely bad submission, which is exactly the
# failure mode we must never have.
#
# The probe FORKS and really becomes (SOLVER_UID, SOLVER_GID) before touching
# anything:
#   * os.access() consults the REAL uid, so seteuid()+os.access() would still
#     answer for root -- it has to be a full setuid in a throwaway child;
#   * stat().st_mode's "other" bits describe the leaf only -- they miss a
#     missing +x on a parent directory, a POSIX ACL, and a nosuid/noexec-style
#     mount option. Only a real open() sees all three.
# The ancestor chain is probed explicitly with stat(dir + "/.") -- that is the
# operation that needs search (+x) permission on `dir` itself, and it is what
# a 0700 /app would break even with a perfectly 0755 /app/methods underneath.
# ---------------------------------------------------------------------------

_READ_PROBE_MAX_FILES = 4096      # submissions are small; bound it anyway
_READ_PROBE_MAX_PROBLEMS = 12     # enough to diagnose, short enough to log


class ArtifactUnreadableError(RuntimeError):
    """Graded artifacts are not readable by the unprivileged solver account."""


def _probe_one_as_solver(path, uid, want_exec):
    """Inside the setuid'd probe child: report why `path` is not usable."""
    problems = []
    parts = [p for p in path.split(os.sep) if p]
    for i in range(len(parts)):                     # every ancestor, "/" first
        d = os.sep + os.sep.join(parts[:i])
        try:
            os.stat(os.path.join(d, "."))           # needs +x on d itself
        except OSError as exc:
            return ["uid %d cannot traverse %s (%s) on the way to %s"
                    % (uid, d, exc.strerror, path)]
    if os.path.isdir(path):
        stack, seen = [path], 0
        while stack and seen < _READ_PROBE_MAX_FILES:
            d = stack.pop()
            try:
                names = sorted(os.listdir(d))       # needs +r and +x on d
            except OSError as exc:
                problems.append("uid %d cannot list directory %s (%s)"
                                % (uid, d, exc.strerror))
                if len(problems) >= _READ_PROBE_MAX_PROBLEMS:
                    return problems
                continue
            for name in names:
                p = os.path.join(d, name)
                # Never follow symlinks: these targets are agent-writable.
                if os.path.islink(p):
                    continue
                if os.path.isdir(p):
                    stack.append(p)
                    continue
                if not os.path.isfile(p):           # sockets/fifos/dangling
                    continue
                seen += 1
                if seen > _READ_PROBE_MAX_FILES:
                    break
                problems.extend(_probe_file_as_solver(p, uid, False))
                if len(problems) >= _READ_PROBE_MAX_PROBLEMS:
                    return problems
    else:
        problems.extend(_probe_file_as_solver(path, uid, want_exec))
    return problems


def _probe_file_as_solver(path, uid, want_exec):
    problems = []
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as exc:
        return ["uid %d cannot open %s for reading (%s)"
                % (uid, path, exc.strerror)]
    try:
        os.read(fd, 1)                              # a mode bit is not a read
    except OSError as exc:
        problems.append("uid %d cannot read %s (%s)" % (uid, path, exc.strerror))
    finally:
        os.close(fd)
    if want_exec and not os.access(path, os.X_OK):  # real uid == uid here
        problems.append("uid %d cannot execute %s" % (uid, path))
    return problems


def assert_artifacts_readable_by_solver(*targets, **kw):
    """Fail LOUDLY if the solver uid could not read the graded artifacts.

    Called as root before any privilege drop. Targets that do not exist at all
    are left alone -- a missing submission is the caller's own error to report,
    not a permission problem.
    """
    want_exec = bool(kw.get("want_exec", False))
    uid, gid = SOLVER_UID, SOLVER_GID
    paths = [os.path.abspath(str(t)) for t in targets if t is not None]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths or uid == 0 or os.geteuid() != 0:
        return                                      # authoring-host run: no drop
    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:                                    # ---- probe child ----
        found = []
        try:
            os.close(r_fd)
            os.setgroups([])
            os.setgid(gid)
            os.setuid(uid)
            if os.getuid() != uid or os.geteuid() != uid:
                found.append("privilege drop to uid %d did not take" % uid)
            else:
                for p in paths:
                    found.extend(_probe_one_as_solver(p, uid, want_exec))
        except BaseException as exc:                # noqa: BLE001
            found.append("probe failed: %s: %s" % (type(exc).__name__, exc))
        try:
            os.write(w_fd, json.dumps(found[:_READ_PROBE_MAX_PROBLEMS]).encode())
        except BaseException:                       # noqa: BLE001
            pass
        os._exit(0)
    os.close(w_fd)
    with os.fdopen(r_fd, "rb") as pipe:
        raw = pipe.read()
    _, status = os.waitpid(pid, 0)
    try:
        problems = json.loads(raw.decode()) if raw else None
    except ValueError:
        problems = None
    if problems is None:
        problems = ["the readability probe died without a verdict "
                    "(wait status %d)" % status]
    if problems:
        # Also shout into the verifier log: some callers truncate or summarise the
        # exception text, and this failure must never be reconstructible only from
        # a bare "reward 0".
        print("FATAL: " + "; ".join(problems), file=sys.stderr, flush=True)
        raise ArtifactUnreadableError(
            # Details FIRST: several graders truncate the exception text when they
            # record it, and the path+errno is the part that is actually actionable.
            "graded artifacts unreadable by solver uid %d/gid %d: %s -- the "
            "submission is executed as that account, so this would otherwise have "
            "been a silent no-output run scored 0 as a bad submission. Fix the "
            "artifact permissions (test.sh does `chmod -R a+rX`) and re-grade."
            % (SOLVER_UID, SOLVER_GID, "; ".join(problems)))


def anchors():
    """Read the three anchors from the sealed root-0400 file; a missing anchor fails loudly."""
    with open(ANCHOR_FILE, encoding="utf-8") as fh:
        raw = json.load(fh)
    missing = [k for k in ANCHOR_KEYS if k not in raw]
    if missing:
        raise RuntimeError(f"anchors missing from {ANCHOR_FILE}: {missing}")
    return tuple(float(raw[k]) for k in ANCHOR_KEYS)


def reward_of(metric: float, baseline: float, frontier: float, upper: float) -> float:
    """baseline->0, frontier->0.6, upper->1."""
    # Linear in the raw metric on both segments, clamped at both ends.
    if not math.isfinite(metric):
        return 0.0
    pts = [(baseline, 0.0), (frontier, 0.6), (upper, 1.0)]
    if metric <= pts[0][0]:
        return 0.0
    for (v0, r0), (v1, r1) in zip(pts, pts[1:]):
        if metric <= v1:
            if metric == v1:            # exact anchor landing, no float drift
                return float(r1)
            return float(r0 + (r1 - r0) * (metric - v0) / max(1e-12, v1 - v0))
    return 1.0                          # hard clamp at the theoretical ceiling


def heldout_seeds() -> list:
    """Read the sealed episode seeds as root, before the untrusted child exists."""
    sys.path.insert(0, str(TESTS))
    from heldout_seeds import HELDOUT_SEEDS  # sealed 0400: importable by root only

    return [int(s) for s in HELDOUT_SEEDS]


def _child_env(scratch: Path) -> dict:
    """Inherit the run's environment (thread pinning etc. must survive so the rollout stays
    bit-reproducible) but point every cache/home path at a scratch dir the de-privileged child can
    actually write, and strip the anchor names defensively in case a harness still injects them."""
    env = {k: v for k, v in os.environ.items() if k not in ANCHOR_KEYS and k != "UPPER"}
    env.update({
        "HOME": str(scratch), "TMPDIR": str(scratch), "XDG_CACHE_HOME": str(scratch),
        "MPLCONFIGDIR": str(scratch), "TORCH_HOME": str(scratch), "NUMBA_CACHE_DIR": str(scratch),
        "MODEL_DIR": MODEL_DIR, "PYTHONDONTWRITEBYTECODE": "1",
    })
    # pin the BLAS/OpenMP pools to 1 with CONSTANTS, never os.environ lookups
    # and never a host-overridable default. This is not a performance knob here, it CHANGES THE
    # SCORE: same checkpoint, same sealed batch, threads in {1,4,8} -> mean return 460.9641695477112,
    # threads = host cpu_count (96) -> 447.4884114018423 (13.48 return = 0.0246 reward). Mechanism:
    # BLAS reduction order shifts the low bits of the action, and 750 chaotic sim steps amplify it.
    # Both Dockerfiles carry the same four ENVs so the
    # agent's selfcheck and the verifier are isomorphic.
    env.update({
        "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4",
    })
    return env


_CHILD_LOG_TAIL_BYTES = 64 * 1024     # tail only: parent memory stays constant


def _tail_text(path: Path, limit: int = _CHILD_LOG_TAIL_BYTES) -> str:
    """Read the last `limit` bytes of a REGULAR file; any error degrades to an empty string."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        return ""
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return ""
        size = st.st_size
        if size > limit:
            os.lseek(fd, size - limit, os.SEEK_SET)
        data = b""
        while len(data) < limit:
            chunk = os.read(fd, limit - len(data))
            if not chunk:
                break
            data += chunk
    except OSError:
        return ""
    finally:
        os.close(fd)
    text = data.decode("utf-8", "replace")
    if size > limit:
        text = ("[... %d earlier bytes of child output dropped ...]\n"
                % (size - limit)) + text
    return text


def _run_child(out_json: Path, scratch: Path, seeds: list) -> dict:
    """Run the untrusted phase; returns a status dict. Only a SPAWN failure propagates."""
    status = {"timed_out": False, "returncode": None, "wall_sec": None}
    t0 = time.monotonic()
    # Spawning is GRADER side; the child logs to a file the parent only tails.
    log_path = scratch / "child_output.log"
    with _phase("grader"):
        _log_fh = open(log_path, "wb")
        try:
            proc = subprocess.Popen(
                [sys.executable, str(CHILD), str(SUBMISSION_DIR), str(MODEL_DIR), str(out_json), str(TESTS)],
                stdin=subprocess.PIPE, stdout=_log_fh, stderr=subprocess.STDOUT, text=True,
                start_new_session=True,                            # own session/group -> killpg reaps grandchildren
                user=SOLVER_UID, group=SOLVER_GID, extra_groups=[],  # <- the G1 A/B/D/E master switch
                env=_child_env(scratch), cwd=str(scratch),
            )
        finally:
            _log_fh.close()        # the child already holds its own fd
    try:
        proc.communicate(input=json.dumps({"seeds": seeds}), timeout=CHILD_TIMEOUT)
        # Read the real exit status HERE, before the killpg below turns it into -9 unconditionally.
        status["returncode"] = proc.returncode
        stdout = _tail_text(log_path)
        if stdout:
            print(stdout, file=sys.stderr)
    except subprocess.TimeoutExpired:
        status["timed_out"] = True
        print(f"child timed out after {CHILD_TIMEOUT}s and was killed", file=sys.stderr)
    except BaseException as exc:  # noqa: BLE001 — the untrusted phase must never abort the grader
        status["spawn_error"] = f"{type(exc).__name__}: {exc}"
        print(f"child failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            pass
        status["wall_sec"] = round(time.monotonic() - t0, 3)
    return status


def _write(out: dict, grader_failed=None) -> None:
    rewards = {"reward": float(out["reward"]),
               "mean_return": float(out["metric"]) if out["metric"] is not None else 0.0,
               "errors": list(out.get("errors") or [])}
    REWARD_DIR.mkdir(parents=True, exist_ok=True)
    (REWARD_DIR / "reward.txt").write_text(f"{out['reward']}\n", encoding="utf-8")
    (REWARD_DIR / "reward.json").write_text(
        json.dumps(_reward_json_numeric(rewards, grader_failed=grader_failed)), encoding="utf-8")
    (REWARD_DIR / "grade_debug.json").write_text(json.dumps(out), encoding="utf-8")
    # Human-readable text must land in score_details.json.
    (REWARD_DIR / "score_details.json").write_text(json.dumps(out), encoding="utf-8")


def main() -> None:
    REWARD_DIR.mkdir(parents=True, exist_ok=True)
    out = {"metric": None, "reward": 0.0, "correctness": False, "errors": []}
    run_errors: list = []   # submission-side diagnostics, NEVER `grader failed:`
    try:
        base, frontier, upper = anchors()   # sealed material, read as root BEFORE any spawn
        seeds = heldout_seeds()

        with _phase("submission"):
            if not (SUBMISSION_DIR / "solver.py").exists():
                raise ValueError(f"no solver.py under {SUBMISSION_DIR}")
            # the gate used to be `(MODEL_DIR / "model.pt").exists()`. The filename
            # "model.pt" appears NOWHERE in the agent-visible docs — instruction.md only asks for "save
            # everything needed to reload it into out_dir" and explicitly invites replacing the algorithm
            # entirely — so a submission that saved policy.pt / weights.npz / a subdirectory satisfied
            # the documented contract and still scored 0, BEFORE the child was ever started (the policy
            # never got to run). The gate is now "the checkpoint directory exists and is non-empty";
            # whether the bytes in it are loadable is the submitted Policy.__init__'s problem, and a
            # genuine load failure surfaces as a child error with a real traceback.
            model_dir = Path(MODEL_DIR)
            if not model_dir.is_dir() or not any(model_dir.iterdir()):
                raise ValueError(f"no checkpoint under {MODEL_DIR} — did the agent run train()? "
                                 "(the directory must exist and contain at least one file)")
        # prove the unprivileged solver account can really READ the graded artifacts
        # BEFORE handing them over. If it cannot, the child simply produces nothing and the
        # run scores 0 as though the submission were bad -- a silent, unattributable
        # failure. Raising here makes the cause land in the reward file instead.
        assert_artifacts_readable_by_solver(SUBMISSION_DIR, MODEL_DIR)

        # Manual cleanup: a cleanup failure must never become a grader failure.
        tmp = tempfile.mkdtemp(prefix="ara_tf_")
        try:
            # Scratch dir handed to the de-privileged child: the ONLY path it may write (traces +
            # its own caches). The trusted parent, still root, reads the traces back out of it.
            os.chown(tmp, SOLVER_UID, SOLVER_GID)
            os.chmod(tmp, 0o700)
            traces = Path(tmp) / "traces.json"
            # Submission side from here until the scorer call.
            _set_phase("submission")
            status = _run_child(traces, Path(tmp), seeds)  # untrusted code runs ONLY here

            # Record WHY before deciding whether there is anything to score.
            if status["timed_out"]:
                run_errors.append(f"submission timed out after {CHILD_TIMEOUT}s")
            rc = status.get("returncode")
            if rc not in (None, 0):
                hint = ""
                if rc in (-9, 137, -6, 134):     # SIGKILL / SIGABRT: the OOM-killer signature
                    hint = (" (SIGKILL/SIGABRT — the usual cause is the kernel OOM killer; "
                            "compare the checkpoint size against [verifier] memory_mb)")
                elif rc in (-7, 135):
                    hint = (" (SIGBUS — almost always a write into a /dev/shm-backed shared "
                            "segment larger than the container's 64 MiB /dev/shm: "
                            "multiprocessing.shared_memory, torch tensor sharing across "
                            "DataLoader workers, or joblib memmapping. The segment is created "
                            "fine and the FIRST WRITE kills the process with no output, which "
                            "looks exactly like a policy that produced nothing. Keep the "
                            "inference path single-process, or size shared buffers under "
                            "/dev/shm)")
                run_errors.append(f"submission subprocess exited with returncode {rc}{hint}")
            if "spawn_error" in status:
                run_errors.append(f"submission subprocess could not be run: {status['spawn_error']}")

            # SOFT TRUNCATION: score whatever episodes landed; a missing seed scores 0.0.
            # Only a genuinely empty trace file is fatal.
            if not traces.exists() or traces.stat().st_size == 0:
                raise ValueError("policy produced no action traces"
                                 + (" — " + "; ".join(run_errors) if run_errors else ""))
            # Size gate before parsing: the file size is submission-controlled.
            _tr_bytes = traces.stat().st_size
            if _tr_bytes > MAX_TRACES_BYTES:
                raise ValueError(
                    f"the policy's action-trace file is too large ({_tr_bytes} bytes > "
                    f"{MAX_TRACES_BYTES}); a complete honest run writes roughly 4 MB")
            try:
                parsed_traces = json.loads(traces.read_text())
            except ValueError:
                raise ValueError("the policy's action-trace file is not valid JSON"
                                 + (" — " + "; ".join(run_errors) if run_errors else ""))
            if not isinstance(parsed_traces, dict):
                raise ValueError("the policy's action-trace file is not a JSON object")
            # Action shape and finiteness are checked on the SUBMISSION side.
            n_done = 0
            for _k in [str(s) for s in seeds]:   # seeds order, not set order: reproducible text
                _tr = parsed_traces.get(_k)
                if _tr is None or (isinstance(_tr, list) and not _tr):
                    continue
                if not isinstance(_tr, list):
                    raise ValueError(f"seed {_k}: the action trace is not a JSON list")
                if len(_tr) > MAX_TRACE_STEPS:
                    raise ValueError(f"seed {_k}: the action trace has {len(_tr)} steps, more than "
                                     f"the {MAX_TRACE_STEPS} an episode can possibly consume")
                for _i, _a in enumerate(_tr):
                    if not isinstance(_a, list) or len(_a) != ACT_DIM:
                        raise ValueError(f"seed {_k} step {_i}: action trace is not a finite "
                                         f"{ACT_DIM}-vector (not a {ACT_DIM}-element list)")
                    for _x in _a:
                        if isinstance(_x, bool) or not isinstance(_x, (int, float)) \
                                or not math.isfinite(_x):
                            raise ValueError(
                                f"seed {_k} step {_i}: action trace is not a finite {ACT_DIM}-vector "
                                f"(saw {_x!r}) — the policy emitted a NaN/inf or non-numeric action, "
                                "which usually means training diverged")
                n_done += 1
            if n_done == 0:
                raise ValueError("policy produced no action traces"
                                 + (" — " + "; ".join(run_errors) if run_errors else ""))
            if n_done < len(seeds):
                run_errors.append(
                    f"scored on partial results: {n_done}/{len(seeds)} sealed episodes produced a "
                    f"trace; the rest count as return 0.0 (soft truncation, not a zeroed run)")

            # Back to the grader side: the sealed scorer is ours, its failures are infra.
            _set_phase("grader")
            # the scorer no longer receives the anchors — it only replays traces
            # and reports the raw metric. The mapping happens once, below, in this trusted parent.
            proc = subprocess.run(
                [sys.executable, "-B", str(SCORER), str(traces)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=SCORE_TIMEOUT)
            if proc.returncode == INVALID_TRACE_RC:
                # The scorer says the trace data itself is invalid -> submission side.
                with _phase("submission"):
                    raise ValueError(
                        "the replay scorer rejected the policy's action trace: "
                        + proc.stderr[-300:].strip())
            if proc.returncode != 0:
                raise RuntimeError(f"scorer failed: {proc.stderr[-300:].strip()}")
            raw = json.loads(proc.stdout)
        finally:
            # A cleanup failure must never become a grader failure.
            try:
                shutil.rmtree(tmp, ignore_errors=True)
            except BaseException:      # noqa: BLE001
                pass

        # The reward is computed HERE, in the trusted parent, from the replayed metric only.
        # the retired duplicate mapping in tests/trifinger_score.py is gone, so
        # this really is the only reward_of left in the chain.
        metric = float(raw["metric"])
        # a non-finite metric is a FAILED run, not a great one. It can only arise
        # from NaN/inf actions surviving the replay (trifinger_score.replay_episode now rejects
        # those at the source, this is the second gate), and it used to walk straight through
        # reward_of's segment chain into `return 1.0`. Raising drops into the except below -> 0.0
        # with the cause recorded in errors[], instead of a silent perfect score.
        with _phase("submission"):
            if not math.isfinite(metric):
                raise ValueError(f"scorer returned a non-finite metric ({raw['metric']!r}) — "
                                 "the policy's action trace was not numerically valid")
        out = {"metric": metric,
               "reward": round(reward_of(metric, base, frontier, upper), 6),
               # `correctness` stays True for a soft-truncated run: it DID produce a scoreable
               # result. The truncation itself is reported in errors[], where the aggregation layer
               # can see it without the run being mistaken for an infrastructure failure.
               "correctness": True, "errors": list(run_errors)}
    except Exception as exc:  # noqa: BLE001 — any failure -> reward 0, never crash
        # Prefix and exit code follow the phase the exception was raised in.
        grader_side = (_PHASE == "grader")
        detail = f"{type(exc).__name__}: {exc}"
        out = {"metric": None, "reward": 0.0, "correctness": False,
               "grader_side": grader_side,
               "errors": list(run_errors) + [("grader failed: " + detail) if grader_side else detail]}
        if grader_side:
            print("FATAL grader failed: " + detail, file=sys.stderr, flush=True)

    _write(out)
    print(json.dumps(out))
    # Non-zero exit only so test.sh reaches the same verdict; reward is already on disk.
    if out.get("grader_side"):
        sys.exit(1)


if __name__ == "__main__":
    # "A grader that dies never grades": every exit path must leave a reward on disk.
    # BaseException, not Exception, so SystemExit / KeyboardInterrupt / MemoryError are covered too;
    # only an uncatchable SIGKILL gets past this, and test.sh's exit-code gate catches that.
    try:
        main()
    except SystemExit:
        # The reward files already carry the prefix; pass the exit code through untouched.
        raise
    except BaseException as exc:  # noqa: BLE001
        try:
            # Unconditional grader_failed: getting here means main() did not catch it.
            _write({"metric": None, "reward": 0.0, "correctness": False,
                    "errors": [f"grader failed: {type(exc).__name__}: {str(exc)[:500]}"]},
                   grader_failed=True)
        except BaseException:  # noqa: BLE001 — even the write can fail (read-only /logs); still exit 1
            pass
        sys.exit(1)  # keep the non-zero code so test.sh's gate reaches the same verdict
