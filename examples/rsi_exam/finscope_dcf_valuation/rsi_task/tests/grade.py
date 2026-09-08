"""Harbor verifier for finscope_dcf_valuation.

Runs AFTER the agent finishes (Harbor copies tests/ into the box at verify time, so the
held-out under tests/heldout/ was never visible to the agent). Two-phase, answer-key-isolated:

  1. read the sealed held-out truth into memory, then DELETE it from disk;
  2. run the SUBMITTED forecaster in a subprocess (tests/solve.py) that only sees the OBSERVABLE
     panel (tests/features/, no truth) and emits drivers.json — the submission code never runs
     in this trusted process, and cannot open() the answer key (it is gone from disk);
  3. recompute each holdout company's candidate value V_hat (from the submitted drivers) and the
     perfect-foresight value V* (from the in-memory realized drivers) with the SEALED fin-compute
     engine, score the panel median absolute valuation error (MAVE) on the TEST tier, and map it
     to an anchored reward.

Reward (RL/eval scalar), 2-anchor piecewise on the lower-is-better MAVE axis, anchored at
baseline=0, sota=0.6, then a soft cap that only APPROACHES 1.0 (there is no theoretical upper
anchor on this task -- see _reward()). Any failure -> 0.
"""

from __future__ import annotations

import json
import math
import os
import signal
import stat
import statistics
import subprocess
import sys
import threading
from pathlib import Path


_RJ_TEXT_KEYS = ("error", "errors", "status", "reason", "detail", "details",
                 "message", "messages", "_details", "note", "notes")


GRADER_FAILED_PREFIX = "grader failed: "
_PHASE = "grader"          # grader side from process start
_GRADER_FAILED = False     # set by main(); drives the non-zero exit


def _set_phase(phase: str) -> None:
    """Switch _PHASE from any scope."""
    global _PHASE
    _PHASE = phase


_RJ_KEEP_PREFIXES = ("grader_failed", "submission_")
_RJ_SPILL: dict = {}          # metrics moved out of reward.json


def _rj_split(payload):
    """(kept in reward.json, spilled to score_details); spill also lands in _RJ_SPILL."""
    keep = {k: v for k, v in payload.items()
            if k == "reward" or k.startswith(_RJ_KEEP_PREFIXES)}
    spill = {k: v for k, v in payload.items() if k not in keep}
    _RJ_SPILL.update(spill)
    return keep, spill


def _rj_merge(details):
    """Merge spilled metrics into score_details; never overwrite an existing key."""
    merged = dict(details or {})
    for k, v in _RJ_SPILL.items():
        merged[k if k not in merged else "reward_json_" + k] = v
    return merged

def _reward_json_numeric(payload, grader_failed=None):
    """reward.json payload with finite numbers only (bool -> int, non-numeric dropped).

    grader_failed None -> inferred: only a string that STARTS with GRADER_FAILED_PREFIX counts.
    """
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
        """Yield strings one by one; the prefix test must not see them concatenated."""
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
        if k in _RJ_TEXT_KEYS:
            low = _flatten(v).lower()
            _marker_texts.extend(_strings(v))
            if "timed out" in low or "timeout" in low:
                flags["submission_timed_out"] = 1
            if "returncode" in low or "killed by signal" in low:
                flags["submission_child_failed"] = 1

    if grader_failed is None:
        _pfx = (globals().get("GRADER_FAILED_PREFIX")
                or globals().get("GRADER_FAIL_PREFIX")
                or "grader failed: ")
        grader_failed = (globals().get("_PHASE", "grader") == "grader"
                         and any(s.startswith(_pfx) for s in _marker_texts))
    if grader_failed:
        flags["grader_failed"] = 1

    out.update(flags)
    if "reward" not in out:
        out["reward"] = 0.0
    return out


# --- Anchors: single source of truth is the SEALED file tests/anchors.json (root, mode 0400).
#   BASELINE = MAVE of the weak baseline  -> reward 0
#   SOTA     = MAVE of the frontier solve -> reward 0.6, with a soft cap below it
# Lower MAVE is better, so BASELINE > SOTA > 0.
# They are NOT read from os.environ: task.toml [verifier.env] lands in PID 1's environ, which the
# submitted code reads out of /proc/1/environ (measured). No literal fallbacks either -- a missing
# or broken anchor file must fail loudly (-> reward 0), never silently grade against a stale
# default. This process reads the file as root BEFORE dropping to SOLVER_UID for the child.
ANCHOR_FILE = Path("/tests/anchors.json")
ANCHOR_KEYS = ("BASELINE", "SOTA")
BASELINE = SOTA = None   # populated by _load_anchors() inside main()

# --- Privilege drop for the UNTRUSTED submission subprocess (must exist in tests/Dockerfile).
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
_READ_PROBE_MAX_DIRS = 512        # ditto for directories (bounds the walk itself)
_READ_PROBE_MAX_PROBLEMS = 12     # enough to diagnose, short enough to log

_PROBE_ENV = "env"                # environment fault -> grader phase
_PROBE_SUB = "submission"         # fault inside the submission dir -> submission phase


class ArtifactUnreadableError(RuntimeError):
    """Graded artifacts are not readable by the unprivileged solver account."""


def _is_link(path):
    try:
        return stat.S_ISLNK(os.lstat(path).st_mode)
    except OSError:
        return False


def _lstat_kind(path):
    """(is_dir, is_regular_file) via lstat -- does not follow symlinks."""
    try:
        st = os.lstat(path)
    except OSError:
        return False, False
    return stat.S_ISDIR(st.st_mode), stat.S_ISREG(st.st_mode)


def _probe_one_as_solver(path, uid, want_exec):
    """Inside the setuid'd probe child: report why `path` is not usable.

    Returns [(kind, text), ...]; kind is _PROBE_ENV or _PROBE_SUB.
    """
    problems = []
    parts = [p for p in path.split(os.sep) if p]
    for i in range(len(parts)):                     # every ancestor, "/" first
        d = os.sep + os.sep.join(parts[:i])
        try:
            os.stat(os.path.join(d, "."))           # needs +x on d itself
        except OSError as exc:
            return [(_PROBE_ENV,
                     "uid %d cannot traverse %s (%s) on the way to %s"
                     % (uid, d, exc.strerror, path))]
    if _is_link(path):
        return []
    is_dir, is_file = _lstat_kind(path)
    if is_dir:
        stack, seen, ndirs = [path], 0, 0
        while stack and seen < _READ_PROBE_MAX_FILES and ndirs < _READ_PROBE_MAX_DIRS:
            d = stack.pop()
            ndirs += 1
            try:
                names = sorted(os.listdir(d))       # needs +r and +x on d
            except OSError as exc:
                problems.append((_PROBE_SUB,
                                 "uid %d cannot list directory %s (%s)"
                                 % (uid, d, exc.strerror)))
                if len(problems) >= _READ_PROBE_MAX_PROBLEMS:
                    return problems
                continue
            for name in names:
                p = os.path.join(d, name)
                if _is_link(p):                     # never follow symlinks
                    continue
                p_dir, p_file = _lstat_kind(p)
                if p_dir:
                    stack.append(p)
                    continue
                if not p_file:                      # sockets/fifos/devices
                    continue
                seen += 1
                if seen > _READ_PROBE_MAX_FILES:
                    break
                problems.extend(_probe_file_as_solver(p, uid, False, _PROBE_SUB))
                if len(problems) >= _READ_PROBE_MAX_PROBLEMS:
                    return problems
    elif is_file:
        problems.extend(_probe_file_as_solver(path, uid, want_exec, _PROBE_SUB))
    return problems


def _probe_file_as_solver(path, uid, want_exec, kind=_PROBE_SUB):
    problems = []
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        return [(kind, "uid %d cannot open %s for reading (%s)"
                       % (uid, path, exc.strerror))]
    try:
        os.read(fd, 1)                              # a mode bit is not a read
    except OSError as exc:
        problems.append((kind, "uid %d cannot read %s (%s)"
                               % (uid, path, exc.strerror)))
    finally:
        os.close(fd)
    if want_exec and not os.access(path, os.X_OK):  # real uid == uid here
        problems.append((kind, "uid %d cannot execute %s" % (uid, path)))
    return problems


def assert_artifacts_readable_by_solver(*targets, **kw):
    """Fail LOUDLY if the solver uid could not read the graded artifacts.

    Called as root before any privilege drop. Targets that do not exist at all
    are left alone -- a missing submission is the caller's own error to report,
    not a permission problem.

    "LOUDLY" is not "infra failure": only an untraversable ancestor or a dead probe is a
    grading-environment fault; anything inside the target dir switches to the submission phase.
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
                found.append((_PROBE_ENV,
                              "privilege drop to uid %d did not take" % uid))
            else:
                for p in paths:
                    found.extend(_probe_one_as_solver(p, uid, want_exec))
        except BaseException as exc:                # noqa: BLE001
            found.append((_PROBE_ENV,
                          "probe failed: %s: %s" % (type(exc).__name__, exc)))
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
        problems = [[_PROBE_ENV, "the readability probe died without a verdict "
                                 "(wait status %d)" % status]]
    env_side = False
    texts = []
    for _p in problems:
        if isinstance(_p, (list, tuple)) and len(_p) == 2:
            _kind, _text = _p[0], str(_p[1])
        else:                                       # unknown kind -> treat as env
            _kind, _text = _PROBE_ENV, str(_p)
        env_side = env_side or (_kind != _PROBE_SUB)
        texts.append(_text)
    if texts:
        # Also shout into the verifier log: some callers truncate or summarise the
        # exception text, and this failure must never be reconstructible only from
        # a bare "reward 0".
        print("FATAL: " + "; ".join(texts), file=sys.stderr, flush=True)
        if not env_side:
            _set_phase("submission")
        problems = texts
        raise ArtifactUnreadableError(
            # Details FIRST: several graders truncate the exception text when they
            # record it, and the path+errno is the part that is actually actionable.
            "graded artifacts unreadable by solver uid %d/gid %d: %s -- the "
            "submission is executed as that account, so this would otherwise have "
            "been a silent no-output run scored 0 as a bad submission. Fix the "
            "artifact permissions (test.sh does `chmod -R a+rX`) and re-grade."
            % (SOLVER_UID, SOLVER_GID, "; ".join(problems)))
CHILD_TIMEOUT_SEC = 1200.0

# --- Paths (PHASE-0 VERIFY against Harbor's actual mounts) -----------------------------------
# Agent workspace inside the box: the agent edits methods/main/forecaster.py at /app.
SUBMISSION_DIR = Path("/app/methods/main")
# Observable mirror (no labels) the submission's solve() runs against; travels with tests/.
FEATURES = Path("/tests/features")
# Held-out truth + sealed engine; land only at verify time.
HELDOUT = Path("/tests/heldout")
REWARD_DIR = Path("/logs/verifier")
SOLVE = Path("/tests/solve.py")
# Public DCF bridge for the solve subprocess: lets a forecaster that imports `valuation` /
# `fin_compute_engine` for its own checks run at grade time (only /app/methods is uploaded).
LIBPUB = Path("/tests/libpub")

APE_CAP = 2.0

_STDERR_TAIL_CHARS = 64 * 1024
_SUB_FILE_MAX_BYTES = 32 * 1024 * 1024


def _read_submission_file(path, max_bytes=_SUB_FILE_MAX_BYTES):
    """Read a SUBMISSION-OWNED file; any failure is a SubmissionRunError.

    O_NOFOLLOW + O_NONBLOCK + regular-file check + byte cap: a FIFO or a huge file left in the
    scratch dir must not hang or OOM the grader.
    """
    p = str(path)
    try:
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise SubmissionRunError(
            f"cannot read submission artifact {p} ({exc.strerror})") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise SubmissionRunError(
                f"submission artifact {p} is not a regular file "
                f"(st_mode 0o{st.st_mode:o}) -- refusing to read it")
        if st.st_size > max_bytes:
            raise SubmissionRunError(
                f"submission artifact {p} is {st.st_size} bytes, over the "
                f"{max_bytes} byte cap")
        os.set_blocking(fd, True)
        chunks, got = [], 0
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            got += len(block)
            if got > max_bytes:
                raise SubmissionRunError(
                    f"submission artifact {p} grew past the {max_bytes} byte cap "
                    f"while being read")
            chunks.append(block)
    finally:
        os.close(fd)
    return b"".join(chunks).decode("utf-8", "replace")


def _load_anchors():
    """Read the sealed anchors. Raises (-> reward 0) if the file is missing a key."""
    raw = json.loads(ANCHOR_FILE.read_text(encoding="utf-8"))
    missing = [k for k in ANCHOR_KEYS if k not in raw]
    if missing:
        raise RuntimeError(f"anchors missing from {ANCHOR_FILE}: {missing}")
    return tuple(float(raw[k]) for k in ANCHOR_KEYS)


class SubmissionRunError(RuntimeError):
    """Submission-side run failure (timeout / killed / no output); written verbatim."""


# memory_mb / storage_mb / gpus / gpu_types / tpu / build_timeout_sec / docker_image / os，
_SIGBUS_NOTE = (" (SIGBUS -- typically a write into a shared-memory segment larger than the "
                "container's 64 MiB /dev/shm; multiprocessing/joblib/shared_memory handing "
                "big arrays between workers is the usual cause, and it kills the solver with "
                "NO output, which looks identical to a submission that produced nothing)")


def _describe_child_exit(rc) -> str:
    if rc is None:
        return "child exit status unknown"
    if rc < 0:
        sig = -rc
        note = (" (SIGKILL -- the usual signature of a cgroup/container OOM kill)"
                if sig == 9 else _SIGBUS_NOTE if sig == 7 else "")
        return f"child killed by signal {sig}{note}"
    if rc == 137:
        return "child exited 137 (128+SIGKILL -- the usual signature of an OOM kill)"
    if rc == 135:
        return "child exited 135 (128+SIGBUS)" + _SIGBUS_NOTE
    if rc > 128:
        return f"child exited {rc} (128+signal {rc - 128})"
    return f"child exited with returncode {rc}"


def _run_child(argv, env, cwd):
    """Run the submission harness as the unprivileged solver uid, in its own session.

    Returns (rc, stdout, stderr, timed_out) and never lets a solver-side failure propagate:
      * user=/group= drop the privileges that make holes A/B/D/E exploitable;
      * start_new_session=True puts the child in its own process group so (a) it cannot signal
        the grader's group and (b) killpg reaps grandchildren the submission forked — plain
        subprocess.run(timeout=) only kills the direct child and would leave them running long
        enough to overwrite the reward file we are about to write;
      * TimeoutExpired is turned into a `timed_out` flag instead of an exception, so no code
        path can leave the grader dead with an attacker-seeded reward still on disk.
    """
    proc = subprocess.Popen(
        argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        env=env, cwd=cwd, start_new_session=True,
        user=SOLVER_UID, group=SOLVER_GID, extra_groups=[],
    )
    global _PHASE
    _PHASE = "submission"
    pgid = proc.pid          # start_new_session=True => pgid == pid
    tail = [""]

    def _drain(stream, sink):
        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                sink[0] = (sink[0] + chunk)[-_STDERR_TAIL_CHARS:]
        except Exception:  # noqa: BLE001
            pass

    drainer = threading.Thread(target=_drain, args=(proc.stderr, tail), daemon=True)
    drainer.start()
    try:
        proc.wait(timeout=CHILD_TIMEOUT_SEC)
        drainer.join(timeout=10)
        return proc.returncode, "", tail[0], False
    except subprocess.TimeoutExpired:
        return None, "", f"submission subprocess exceeded {CHILD_TIMEOUT_SEC}s and was killed", True
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.stderr.close()
        except Exception:  # noqa: BLE001
            pass


def _reward(mave: float) -> float:
    """baseline->0, sota->0.6, then a soft cap approaching 1."""
    b, s = BASELINE, SOTA                        # b > s > 0
    m = max(float(mave), 0.0)
    if m >= b:
        return 0.0
    if m >= s:
        return float(0.6 * (b - m) / max(1e-12, b - s))          # baseline..sota -> 0..0.6
    # tau pins the reward density continuous at sota: 0.4 / tau = 0.6 / (b - s)
    tau = (2.0 / 3.0) * max(1e-12, b - s)
    return float(1.0 - 0.4 * math.exp(-(s - m) / tau))           # sota..0 -> 0.6 -> ~1 (never 1)


def _score(drivers, truth, VAL):
    """Recompute dev/test MAVE from the submitted drivers + in-memory realized truth.

    Returns (test_mave, dev_mave, errors); both medians are None on any structural violation."""
    holdout_ids = set(truth)
    if not isinstance(drivers, dict) or not drivers:
        return None, None, ["solve() returned no non-empty drivers mapping"]
    submitted = set(drivers)
    if holdout_ids - submitted:
        return None, None, [f"missing drivers for {len(holdout_ids - submitted)} companies"]
    if submitted - holdout_ids:
        return None, None, [f"drivers for {len(submitted - holdout_ids)} unknown/non-holdout companies"]

    ape = {"dev": [], "test": []}
    for cid in sorted(holdout_ids):
        d = drivers[cid]
        if VAL.validate_drivers(d):
            return None, None, [f"{cid}: invalid drivers"]
        facts = truth[cid]["facts"]
        vhat, _ = VAL.intrinsic_value(d, facts)
        vstar, r_star = VAL.intrinsic_value(truth[cid]["realized_drivers"], facts)
        if r_star is not None or vstar is None or vstar <= 0:
            return None, None, [f"{cid}: truth recompute failed"]
        a = APE_CAP if vhat is None else min(abs(vhat - vstar) / abs(vstar), APE_CAP)
        ape[truth[cid]["split"]].append(a)
    if not ape["test"] or not ape["dev"]:
        return None, None, ["no scorable companies in a split"]
    return float(statistics.median(ape["test"])), float(statistics.median(ape["dev"])), []


def main() -> None:
    global _PHASE, _GRADER_FAILED
    _PHASE = "grader"     # everything before the child starts is grader side
    REWARD_DIR.mkdir(parents=True, exist_ok=True)
    out = {"metric": None, "reward": 0.0, "correctness": False, "dev_mave": None, "errors": []}
    try:
        # 0) anchors from the sealed file (read as root, before any untrusted code runs).
        global BASELINE, SOTA
        BASELINE, SOTA = _load_anchors()
        # prove the unprivileged solver account can really READ the graded artifacts
        # BEFORE handing them over. If it cannot, the child simply produces nothing and the
        # run scores 0 as though the submission were bad -- a silent, unattributable
        # failure. Raising here makes the cause land in the reward file instead.
        assert_artifacts_readable_by_solver(SUBMISSION_DIR)

        # 1) read the answer key into memory, then SEAL it from disk (answer-key isolation):
        # the submission subprocess below must not be able to open the realized drivers.
        truth_path = HELDOUT / "holdout_truth.json"
        truth = json.loads(truth_path.read_text(encoding="utf-8"))["companies"]
        truth_path.unlink()

        # trusted, sealed engine (tests/heldout/_valuation.py -> _fin_compute_engine.py).
        sys.path.insert(0, str(HELDOUT))
        import _valuation as VAL  # noqa: E402

        # 2) run the submission in an isolated subprocess on the OBSERVABLE panel only.
        # The child runs as SOLVER_UID, so its scratch dir must be handed to it explicitly:
        # root creates it, chowns it to the solver and makes it private (0700). Nothing else in
        # the box is writable by that uid, which is the point.
        solve_out = Path("/tmp/finscope_solve")
        solve_out.mkdir(parents=True, exist_ok=True)
        os.chown(solve_out, SOLVER_UID, SOLVER_GID)
        os.chmod(solve_out, 0o700)
        # Minimal env: do NOT hand the grader's own environment to untrusted code (it carries
        # VERIFIER_LOG_DIR, and historically the anchors as well). HOME/TMPDIR point at the
        # scratch dir so anything the submission caches has somewhere writable to land.
        # BLAS/OpenMP thread count pinned to a CONSTANT (= [verifier] cpus) (deliberately NOT
        # os.environ.get(...) with a default -- that hands control to whatever host runs the
        # grader). Threaded reductions change the summation order, so the last bits of a large-K
        # reduction (syrk / covariance / ridge / PCA) depend on the thread count; measured
        # bit-different between OMP=1 and OMP=16 on the 96-core host. The agent is free to pip
        # install numpy/pandas/sklearn into its submission, so this matters even though the
        # verifier image itself is thin.
        solve_env = {"PATH": "/usr/local/bin:/usr/bin:/bin",
                     "HOME": str(solve_out), "TMPDIR": str(solve_out),
                     "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                     "MKL_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2",
                     "LANG": "C.UTF-8"}
        if LIBPUB.is_dir():  # make `import valuation` resolve to the public bridge in the solve box
            solve_env["PYTHONPATH"] = str(LIBPUB)
        for _p, _what in ((SOLVE, "solve harness"), (FEATURES, "observable feature panel")):
            if not _p.exists():
                raise RuntimeError(f"grading harness incomplete: {_what} missing at {_p}")
        rc, _cout, cerr, timed_out = _run_child(
            [sys.executable, str(SOLVE), "--submission", str(SUBMISSION_DIR),
             "--features", str(FEATURES), "--out", str(solve_out)],
            solve_env, str(solve_out))
        drivers_path = solve_out / "drivers.json"
        if timed_out:
            raise SubmissionRunError(
                f"submission timed out after {CHILD_TIMEOUT_SEC:g}s")
        if not drivers_path.exists():
            meta = solve_out / "meta.json"
            try:
                why = (_read_submission_file(meta)[:2000] if meta.exists()
                       else "no meta.json")
            except Exception as _mexc:  # noqa: BLE001
                why = f"meta.json unreadable: {type(_mexc).__name__}: {_mexc}"
            tail = (cerr or "").strip()[-1200:]
            raise SubmissionRunError(
                f"submission produced no drivers.json ({why}); "
                f"{_describe_child_exit(rc)}"
                + (f" [child stderr: {tail}]" if tail else ""))
        drivers = json.loads(_read_submission_file(drivers_path))

        _PHASE = "grader"

        # 3) score on the sealed TEST tier; the DEV tier is diagnostics only.
        test_mave, dev_mave, errs = _score(drivers, truth, VAL)
        if test_mave is None:
            out = {"metric": None, "reward": 0.0, "correctness": False, "dev_mave": None, "errors": errs}
        else:
            out = {"metric": round(test_mave, 6), "reward": round(_reward(test_mave), 6),
                   "correctness": True, "dev_mave": round(dev_mave, 6), "errors": []}
    except SubmissionRunError as exc:
        out = {"metric": None, "reward": 0.0, "correctness": False, "dev_mave": None,
               "errors": [str(exc)]}
    except Exception as exc:  # noqa: BLE001 — any failure -> reward 0, never crash
        msg = f"{type(exc).__name__}: {exc}"
        if _PHASE != "submission":
            _GRADER_FAILED = True
            msg = GRADER_FAILED_PREFIX + msg
        out = {"metric": None, "reward": 0.0, "correctness": False, "dev_mave": None,
               "errors": [msg]}

    # Harbor reward.json must be a flat {key: number} map. Numeric keys only; diagnostics go to
    # stdout + grade_debug.json / score_details.json.
    rewards = {"reward": float(out["reward"]),
               "valuation_mave": float(out["metric"]) if out["metric"] is not None else 0.0}
    if _GRADER_FAILED:
        rewards["errors"] = out["errors"]
    (REWARD_DIR / "reward.txt").write_text(f"{out['reward']}\n", encoding="utf-8")
    _rj, _ = _rj_split(_reward_json_numeric(rewards))
    (REWARD_DIR / "reward.json").write_text(json.dumps(_rj), encoding="utf-8")
    (REWARD_DIR / "grade_debug.json").write_text(json.dumps(out), encoding="utf-8")
    (REWARD_DIR / "score_details.json").write_text(json.dumps(_rj_merge(out)), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    # Last-ditch fail-closed: EVERY exit path must leave a reward file on disk, otherwise a
    # reward seeded by the submission before it crashed us survives and gets reported.
    # BaseException (not Exception) so SystemExit / KeyboardInterrupt / MemoryError are covered
    # too; only an uncatchable SIGKILL gets past this, and test.sh's exit-code gate catches that.
    try:
        main()
    except BaseException as exc:  # noqa: BLE001
        try:
            REWARD_DIR.mkdir(parents=True, exist_ok=True)
            (REWARD_DIR / "reward.txt").write_text("0.0\n", encoding="utf-8")
            (REWARD_DIR / "reward.json").write_text(
                json.dumps(_rj_split(_reward_json_numeric(
                    {"reward": 0.0, "valuation_mave": 0.0, "grader_failed": 1}))[0]),
                encoding="utf-8")
            _doc = {"metric": None, "reward": 0.0, "correctness": False, "dev_mave": None,
                    "errors": [f"{GRADER_FAILED_PREFIX}{type(exc).__name__}: {str(exc)[:500]}"]}
            (REWARD_DIR / "grade_debug.json").write_text(json.dumps(_doc), encoding="utf-8")
            (REWARD_DIR / "score_details.json").write_text(json.dumps(_doc), encoding="utf-8")
        except BaseException:  # noqa: BLE001
            pass
        sys.exit(1)   # keep the non-zero exit code so test.sh's gate agrees with the file
    if _GRADER_FAILED:
        sys.exit(1)
