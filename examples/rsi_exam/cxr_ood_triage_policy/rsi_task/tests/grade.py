"""TWO-PHASE Harbor verifier (PARENT / trusted side) for cxr_ood_triage_policy.

Runs AFTER the agent finishes (Harbor copies tests/ into the box at verify time, so the held-out
under tests/heldout/ was never visible to the agent). The verifier RE-RUNS the agent's submitted
triage() on the sealed held-out studies, so untrusted submission code executes inside the verifier
container. Per docs/anti-cheat.md, that uses the TWO-PHASE pattern:

    PARENT (this file, trusted — NEVER imports the submission):
      1. load the held-out grade studies, keep the gold_critical labels in memory, write a
         GOLD-STRIPPED copy of the studies + the (labeled, allowed) calibration resources for the
         child, and stash the gold-bearing grade file out of reach;
      2. spawn a CHILD (`python3 child_triage.py`, a fresh process — NOT a fork) in its own
         session/process group; the child imports + runs triage() and writes ONLY the per-study
         risks, then SIGKILL the whole child process group;
      3. restore the grade file, recompute the raw triage_utility_pct from the gold kept in
         memory, normalize against the anchors, and write reward.json + reward.txt.

Because the score is computed by THIS trusted process from gold the child never saw, a submission
cannot read the labels or monkeypatch the scoring. (The labeled calibration sample is intentionally
handed to triage as a resource — it is NOT the protected gold.)

HARDENING. Process isolation alone is not enough when both sides share uid 0, so:
  * the child is dropped to a dedicated unprivileged account (SOLVER_UID/GID, created in
    tests/Dockerfile). As uid 4242 it cannot read the sealed gold or anchors (0400 root), cannot
    write anything under /tests, cannot write /logs/verifier, cannot signal the root parent, and
    cannot read /proc/1/environ. That single change closes four separate full-score bypasses;
  * the anchors are read from the sealed /tests/anchors.json, NOT from the environment: anything
    the harness injects as an env var is world-readable through PID 1's /proc/1/environ. There is
    deliberately no literal fallback — a missing anchor must fail loudly, never grade silently
    against a stale default;
  * every exit path writes a reward. The child's timeout is handled, its whole process group is
    killed, and __main__ has a BaseException backstop, so "the grader crashed" can never leave a
    submission-planted reward file on disk (test.sh's exit-code gate is the outer layer).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path


# ---------------------------------------------------------------------------
#
#
# ---------------------------------------------------------------------------
_RJ_TEXT_KEYS = ("error", "errors", "status", "reason", "detail", "details",
                 "message", "messages", "_details", "note", "notes")


def _reward_json_numeric(payload, grader_failed=None):
    """reward.json payload with finite numbers only; grader_failed inferred when None."""
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
        """Yield the strings one by one; never join before the prefix test."""
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


HELDOUT = Path("/tests/heldout")
SUBMISSION_DIR = Path("/app/methods/main")
AGENT_TREE = Path("/app/methods")
REWARD_DIR = Path("/logs/verifier")
CHILD = Path("/tests/child_triage.py")

GRADE = HELDOUT / "grade.jsonl"
CALIB = HELDOUT / "calibration.jsonl"
CHILD_TIMEOUT = 1800.0

# Unprivileged account the untrusted child runs as (created in tests/Dockerfile).
SOLVER_UID = 4242
SOLVER_GID = 4242


# ---------------------------------------------------------------------------
#
#
#
#
# ---------------------------------------------------------------------------
GRADER_FAILED_PREFIX = "grader failed: "
_PHASE = "grader"
_GRADER_FAILED = False


def _set_phase(phase: str) -> None:
    global _PHASE
    assert phase in ("grader", "submission")
    _PHASE = phase


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


class ProbeInfraError(RuntimeError):
    """The readability probe itself, or the image below AGENT_TREE, is broken."""


# ---------------------------------------------------------------------------
#
#
#
# ---------------------------------------------------------------------------

def _under_agent_tree(path) -> bool:
    root = os.path.abspath(str(AGENT_TREE))
    p = os.path.abspath(str(path))
    return p == root or p.startswith(root + os.sep)


def _probe_one_as_solver(path, uid, want_exec):
    """Inside the setuid'd probe child: report why `path` is not usable.

    Returns a list of ``[offending_path, message]`` pairs (JSON-friendly): the caller needs the
    path to decide whose fault the problem is (see the N1-b block above).
    """
    problems = []
    parts = [p for p in path.split(os.sep) if p]
    for i in range(len(parts)):                     # every ancestor, "/" first
        d = os.sep + os.sep.join(parts[:i])
        try:
            os.stat(os.path.join(d, "."))           # needs +x on d itself
        except OSError as exc:
            return [[d, "uid %d cannot traverse %s (%s) on the way to %s"
                     % (uid, d, exc.strerror, path)]]
    if os.path.isdir(path):
        stack, seen = [path], 0
        while stack and seen < _READ_PROBE_MAX_FILES:
            d = stack.pop()
            try:
                names = sorted(os.listdir(d))       # needs +r and +x on d
            except OSError as exc:
                problems.append([d, "uid %d cannot list directory %s (%s)"
                                 % (uid, d, exc.strerror)])
                if len(problems) >= _READ_PROBE_MAX_PROBLEMS:
                    return problems
                continue
            for name in names:
                p = os.path.join(d, name)
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
        return [[path, "uid %d cannot open %s for reading (%s)"
                 % (uid, path, exc.strerror)]]
    try:
        os.read(fd, 1)                              # a mode bit is not a read
    except OSError as exc:
        problems.append([path, "uid %d cannot read %s (%s)" % (uid, path, exc.strerror)])
    finally:
        os.close(fd)
    if want_exec and not os.access(path, os.X_OK):  # real uid == uid here
        problems.append([path, "uid %d cannot execute %s" % (uid, path)])
    return problems


def assert_artifacts_readable_by_solver(*targets, **kw):
    """Fail LOUDLY if the solver uid could not read the graded artifacts.

    Called as root before any privilege drop. Targets that do not exist at all
    are left alone -- a missing submission is the caller's own error to report,
    not a permission problem.

    N1-b (2026-08-03): raises ArtifactUnreadableError when the unreadable path is inside
    AGENT_TREE (submission's own doing -> real 0, no infra_failure) and ProbeInfraError when it is
    not, or when the probe machinery itself broke (grader's doing -> `grader failed:` + rc 1).
    """
    want_exec = bool(kw.get("want_exec", False))
    uid, gid = SOLVER_UID, SOLVER_GID
    paths = [os.path.abspath(str(t)) for t in targets if t is not None]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths or uid == 0 or os.geteuid() != 0:
        return                                      # authoring-host run: no drop
    try:
        r_fd, w_fd = os.pipe()
        pid = os.fork()
    except OSError as exc:
        raise ProbeInfraError(
            "the readability probe could not start: %s: %s" % (type(exc).__name__, exc))
    if pid == 0:                                    # ---- probe child ----
        found = []
        try:
            os.close(r_fd)
            os.setgroups([])
            os.setgid(gid)
            os.setuid(uid)
            if os.getuid() != uid or os.geteuid() != uid:
                found.append(["", "privilege drop to uid %d did not take" % uid])
            else:
                for p in paths:
                    found.extend(_probe_one_as_solver(p, uid, want_exec))
        except BaseException as exc:                # noqa: BLE001
            found.append(["", "probe failed: %s: %s" % (type(exc).__name__, exc)])
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
        problems = [["", "the readability probe died without a verdict "
                         "(wait status %d)" % status]]
    norm = []
    for item in problems:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            norm.append((str(item[0]), str(item[1])))
        else:
            norm.append(("", str(item)))
    if norm:
        msgs = [m for _, m in norm]
        # Also shout into the verifier log: some callers truncate or summarise the
        # exception text, and this failure must never be reconstructible only from
        # a bare "reward 0".
        print("FATAL: " + "; ".join(msgs), file=sys.stderr, flush=True)
        infra = [m for p, m in norm if not _under_agent_tree(p)]
        if infra:
            raise ProbeInfraError(
                "the graded-artifact readability probe hit a problem OUTSIDE the submission "
                "tree %s, i.e. in the verifier image itself: %s"
                % (AGENT_TREE, "; ".join(infra)))
        raise ArtifactUnreadableError(
            # Details FIRST: several graders truncate the exception text when they
            # record it, and the path+errno is the part that is actually actionable.
            "graded artifacts unreadable by solver uid %d/gid %d: %s -- the "
            "submission is executed as that account, so this would otherwise have "
            "been a silent no-output run scored 0 as a bad submission. Everything "
            "reported here lives under %s, which the submission side owns (modes, "
            "ACLs and symlink targets alike), so this is a SUBMISSION failure scored "
            "0, not a grading-infrastructure failure."
            % (SOLVER_UID, SOLVER_GID, "; ".join(msgs), AGENT_TREE))

# Reward anchors (raw triage_utility_pct). Sole source of truth: the sealed, root-0400
# /tests/anchors.json, read HERE as root before the child is spawned. NOT the environment —
# task.toml [verifier.env] lands in PID 1's environ and the submission can just read it.
ANCHOR_FILE = Path("/tests/anchors.json")
ANCHOR_KEYS = ("CXR_BASELINE", "CXR_REFERENCE", "CXR_SOFTCAP_SOTA_FRAC")
BASELINE = None   # set by load_anchors() at the top of main()
REFERENCE = None
SOTA_FRAC = None

# Reward landing: baseline -> 0, sota point -> 0.6, upper -> 1.0.
UPPER = 100.0     # structural ceiling of triage_utility_pct = 100*clip(...,0,1); hard anchor
R_SOTA = 0.6


def load_anchors() -> None:
    """Populate BASELINE/REFERENCE/SOTA_FRAC from the sealed anchor file.

    No env lookup and no literal fallback on purpose: a missing/renamed anchor has to raise
    (-> reward 0 + a recorded error), never grade quietly against a plausible-looking stale
    constant.
    """
    global BASELINE, REFERENCE, SOTA_FRAC
    with open(ANCHOR_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    missing = [k for k in ANCHOR_KEYS if k not in raw]
    if missing:
        raise RuntimeError(f"anchors missing from {ANCHOR_FILE}: {missing}")
    BASELINE = float(raw["CXR_BASELINE"])
    REFERENCE = float(raw["CXR_REFERENCE"])
    SOTA_FRAC = float(raw["CXR_SOFTCAP_SOTA_FRAC"])
    if not REFERENCE > BASELINE:
        raise RuntimeError("anchors not ordered: need CXR_REFERENCE > CXR_BASELINE")
    if not SOTA_FRAC > 0:
        raise RuntimeError("CXR_SOFTCAP_SOTA_FRAC must be > 0")
    if not REFERENCE + SOTA_FRAC * (REFERENCE - BASELINE) < UPPER:
        raise RuntimeError("anchors not ordered: the 0.6 calibration point must be below UPPER")

# Metric constants (must match the task's scoring; mirrored from the source evaluator).
CRITICAL = ["Effusion", "Cardiomegaly", "Edema", "Pneumonia",
            "Consolidation", "Atelectasis", "Pneumothorax", "Mass"]
BUDGET = 0.20
N_BINS = 15
LAMBDA_CAL = 1.0
LAMBDA_GAP = 0.5
SEED = 20260626


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def strip_gold(case):
    return {k: v for k, v in case.items() if not k.startswith("gold_")}


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def brier(pairs):
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs) if pairs else 0.0


def score(cases, risk_by_uid):
    """Raw triage_utility_pct from gold_critical (trusted; never touches submission code)."""
    sites = sorted({c["site_id"] for c in cases})
    n = len(cases)
    k = math.ceil(BUDGET * n)
    order = sorted(cases, key=lambda c: (-risk_by_uid[c["study_uid"]], str(c["study_uid"])))
    referred = {c["study_uid"] for c in order[:k]}

    sens_list, cal_list, per_site = [], [], {}
    for s in sites:
        sc = [c for c in cases if c["site_id"] == s]
        crit = [c for c in sc if int(c["gold_critical"]) == 1]
        pairs = [(risk_by_uid[c["study_uid"]], int(c["gold_critical"])) for c in sc]
        site_cal = brier(pairs)
        cal_list.append(site_cal)
        if crit:
            sens = sum(1 for c in crit if c["study_uid"] in referred) / len(crit)
            sens_list.append(sens)
            per_site[str(s)] = {"n": len(sc), "n_critical": len(crit),
                                "sensitivity": round(sens, 4), "brier": round(site_cal, 4)}
        else:
            per_site[str(s)] = {"n": len(sc), "n_critical": 0,
                                "sensitivity": None, "brier": round(site_cal, 4)}
    if not sens_list:
        raise ValueError("no critical studies in any site")
    mean_sens, min_sens, mean_cal = mean(sens_list), min(sens_list), mean(cal_list)
    u = mean_sens - LAMBDA_CAL * mean_cal - LAMBDA_GAP * (mean_sens - min_sens)
    metric = 100.0 * max(0.0, min(1.0, u))
    details = {"mean_sensitivity": round(mean_sens, 4), "min_sensitivity": round(min_sens, 4),
               "mean_brier": round(mean_cal, 4), "budget": BUDGET, "n": n, "referred": k,
               "per_site": per_site}
    return metric, details


def sota_point():
    """Raw value that lands on 0.6, placed by the calibration constant."""
    return REFERENCE + SOTA_FRAC * (REFERENCE - BASELINE)


def normalize(raw):
    """baseline -> 0, sota point -> 0.6, upper (100) -> 1; linear on both segments."""
    sota = sota_point()
    if raw <= BASELINE:
        return 0.0
    if raw <= sota:
        return R_SOTA * (raw - BASELINE) / max(1e-9, sota - BASELINE)
    if raw >= UPPER:
        return 1.0
    return R_SOTA + (1.0 - R_SOTA) * (raw - sota) / max(1e-9, UPPER - sota)


def _describe_rc(rc: int) -> str:
    """Turn a child return code into a readable cause of death."""
    if rc == 0:
        return ""
    if rc is None:
        return "submission subprocess did not report an exit status"
    if rc < 0 or rc > 128:
        sig = -rc if rc < 0 else rc - 128
        name = signal.Signals(sig).name if sig in {s.value for s in signal.Signals} else f"signal {sig}"
        extra = " -- SIGKILL with no timeout recorded is the signature of an OOM kill" \
            if sig == signal.SIGKILL else ""
        if sig == signal.SIGBUS:
            extra = (" -- SIGBUS on first touch of a mapping is the signature of exhausting the "
                     "container's /dev/shm (64 MiB by default): multiprocessing.shared_memory / "
                     "mmap let you CREATE a larger segment and only fault when you write it. "
                     "This is a crash of the submission, not an empty submission")
        return f"submission subprocess was killed by {name} (returncode {rc}){extra}"
    return f"submission subprocess exited with returncode {rc}"


# ---------------------------------------------------------------------------
#
#
#
# ---------------------------------------------------------------------------
_CHILD_LOG_RETAIN_BYTES = 1024 * 1024
_CHILD_LOG_LIMIT_BYTES = 256 * 1024 * 1024
_CHILD_REAP_GRACE = 10.0


class _CappedReader(threading.Thread):
    """Drain the child pipe to EOF with a hard resident cap; on_overflow past `limit`."""

    def __init__(self, stream, cap, limit, on_overflow):
        threading.Thread.__init__(self)
        self.daemon = True
        self._stream, self._cap, self._limit = stream, cap, limit
        self._on_overflow = on_overflow
        self._buf = bytearray()
        self.total = 0
        self.overflowed = False

    def run(self) -> None:
        try:
            while True:
                chunk = self._stream.read(65536)
                if not chunk:
                    break
                self.total += len(chunk)
                self._buf.extend(chunk)
                if len(self._buf) > self._cap:
                    del self._buf[:len(self._buf) - self._cap]
                if self.total > self._limit and not self.overflowed:
                    self.overflowed = True
                    try:
                        self._on_overflow()
                    except BaseException:                      # noqa: BLE001
                        pass
        except BaseException:                                  # noqa: BLE001
            pass
        finally:
            try:
                self._stream.close()
            except BaseException:                              # noqa: BLE001
                pass

    def text(self) -> str:
        return bytes(self._buf).decode("utf-8", "replace")


def run_child(cases_json, resources_json, out_json, scratch) -> list:
    """Run the submission as SOLVER_UID, in its own session, with a scrubbed environment.

    Returns a list of diagnostic strings (timeout fact + subprocess returncode) for the caller to merge into `errors`. Previously this function
    printed the timeout to stderr and returned nothing, so the reward file recorded a bare
    "submission produced no predictions" whether the submission was killed at the wall,
    OOM-killed, or genuinely wrote nothing.

    `user=`/`group=`/`extra_groups=[]` are the whole ballgame: as uid 4242 the submission can no
    longer read the sealed gold or anchors, write /tests or /logs/verifier, signal this process,
    or read /proc/1/environ. `env=` is a fixed allow-list so nothing the harness handed the
    verifier (anchors, paths, tokens) is inherited; note this is hygiene, not a boundary — the
    boundary is the uid, because env scrubbing alone is defeated by /proc/1/environ.
    """
    proc = subprocess.Popen(
        [sys.executable, str(CHILD), str(SUBMISSION_DIR),
         str(cases_json), str(resources_json), str(out_json)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True,          # own process group, so killpg below reaps grandchildren
        user=SOLVER_UID, group=SOLVER_GID, extra_groups=[],
        cwd=str(scratch),
        env={"PATH": "/usr/local/bin:/usr/bin:/bin",
             "HOME": str(scratch), "TMPDIR": str(scratch),
             "XDG_CACHE_HOME": str(scratch), "MPLCONFIGDIR": str(scratch),
             # thread count pinned to a CONSTANT (= [verifier] cpus) (never read from
             # os.environ). Threaded reductions change the summation order, so the last bits of a
             # large reduction depend on the thread count -- measured bit-different between OMP=1
             # and OMP=16 on a 96-core host. This task is currently immune -- the verifier image
             # is stdlib-only, so a submission that imports numpy fails to import at all -- so
             # these four are DEFENSIVE, not load-bearing: they cost nothing, they cannot change
             # today's anchors, and they mean that adding a numeric package to the image later
             # cannot silently make the score a function of the grading host's core count.
             "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
             "MKL_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2",
             "PYTHONDONTWRITEBYTECODE": "1", "LC_ALL": "C.UTF-8"})
    _set_phase("submission")
    notes = []
    try:
        _pgid = os.getpgid(proc.pid)
    except OSError:
        _pgid = proc.pid

    def _killpg() -> None:
        try:
            os.killpg(_pgid, signal.SIGKILL)
        except OSError:
            pass

    reader = _CappedReader(proc.stdout, _CHILD_LOG_RETAIN_BYTES,
                           _CHILD_LOG_LIMIT_BYTES, _killpg)
    reader.start()
    joined = [False]
    try:
        proc.wait(timeout=CHILD_TIMEOUT)
        _killpg()
        reader.join(_CHILD_REAP_GRACE)
        joined[0] = True
        stdout = reader.text()
        if stdout:
            print(stdout, file=sys.stderr)
        if reader.overflowed:
            notes.append(
                f"submission wrote more than {_CHILD_LOG_LIMIT_BYTES} bytes to "
                f"stdout/stderr and was killed; the grader caps how much of an "
                f"untrusted child's output it buffers")
        else:
            # capture the rc BEFORE the `finally` killpg below, which would otherwise
            # overwrite a clean exit status with the signal we ourselves sent.
            desc = _describe_rc(proc.returncode)
            if desc:
                notes.append(desc)
    except subprocess.TimeoutExpired:
        # Must be caught: an uncaught TimeoutExpired would kill the grader mid-run, and a grader
        # that dies without writing is exactly how a planted reward file survives.
        # the timeout gets its OWN sentence, in the exact library-wide wording, so the
        # aggregation layer can tell "ran out of wall clock" from "produced nothing".
        notes.append(f"submission timed out after {CHILD_TIMEOUT:g}s")
        print(notes[-1], file=sys.stderr)
    except BaseException as exc:  # noqa: BLE001 — a failing submission never fails the grader
        notes.append(f"submission subprocess could not be waited on: "
                     f"{type(exc).__name__}: {exc}")
        print(notes[-1], file=sys.stderr)
    finally:
        # killpg, not proc.kill(): the submission may have forked grandchildren that would
        # otherwise outlive us and race the reward file.
        _killpg()
        try:
            proc.wait(timeout=10)
        except BaseException:  # noqa: BLE001
            pass
        try:
            if reader.is_alive() and not joined[0]:
                reader.join(_CHILD_REAP_GRACE)
        except BaseException:  # noqa: BLE001
            pass
    return notes


def main() -> None:
    global _GRADER_FAILED
    REWARD_DIR.mkdir(parents=True, exist_ok=True)
    out = {"metric": None, "reward": 0.0, "correctness": False, "errors": []}
    child_notes = []          # A1(a) (2026-08-01): timeout / returncode facts, merged below
    _set_phase("grader")
    try:
        load_anchors()          # as root, before any untrusted code exists
        # prove the unprivileged solver account can really READ the graded artifacts
        # BEFORE handing them over. If it cannot, the child simply produces nothing and the
        # run scores 0 as though the submission were bad -- a silent, unattributable
        # failure. Raising here makes the cause land in the reward file instead.
        #
        try:
            assert_artifacts_readable_by_solver(SUBMISSION_DIR)
        except ArtifactUnreadableError:
            _set_phase("submission")
            raise
        grade = read_jsonl(GRADE)
        calib = read_jsonl(CALIB)
        if not grade:
            raise ValueError("no held-out studies")
        sites = sorted({c["site_id"] for c in grade})
        valid_uids = {c["study_uid"] for c in grade}

        tmp = tempfile.mkdtemp(prefix="ara_cxr_")
        try:
            # The dropped child has to CREATE preds.json in here, so the scratch dir is handed to
            # SOLVER_UID (0700 solver-owned: no other unprivileged process can look in, and root
            # reads it back regardless). This is the only place the child may write; the inputs
            # below stay root-owned 0644 so they are readable but not forgeable.
            os.chown(tmp, SOLVER_UID, SOLVER_GID)
            os.chmod(tmp, 0o700)
            cases_json = Path(tmp) / "cases.json"
            resources_json = Path(tmp) / "resources.json"
            preds_json = Path(tmp) / "preds.json"
            cases_json.write_text(json.dumps([strip_gold(c) for c in grade]))
            resources_json.write_text(json.dumps(
                {"budget": BUDGET, "labels": list(CRITICAL), "calibration": calib,
                 "sites": sites, "n_bins": N_BINS}))
            for p in (cases_json, resources_json):
                os.chmod(p, 0o644)   # root-owned, child-readable, child-unwritable

            # Remove the gold-bearing grade file from DISK before any untrusted code runs.
            # N6 fix 2026-07-27: this used to rename() it to grade.jsonl.stashed and rename it
            # back afterwards. Renaming is not hiding -- the 1600 gold labels stayed plainly
            # readable at a trivially guessable path (and os.listdir("/tests/heldout") handed
            # the new name over for free) for the entire window in which the submission runs.
            # unlink() is safe because the labels are ALREADY in memory: `grade` was read at the
            # top of main() and every later consumer (valid_uids, score(), the length checks)
            # reads that list, never the file. Nothing re-opens GRADE after this point, so there
            # is deliberately no restore -- the verifier image is discarded after each trial,
            # and "restore the answer key" is not a property worth keeping.
            # The child still gets everything it legitimately needs: the gold-stripped cases
            # via cases.json, and calibration (allowed) via resources.json.
            GRADE.unlink()
            # untrusted code runs ONLY here
            child_notes.extend(run_child(cases_json, resources_json, preds_json, tmp))

            if not preds_json.exists() or preds_json.stat().st_size == 0:
                raise ValueError("submission produced no predictions")
            preds = json.loads(preds_json.read_text())
        finally:
            try:
                shutil.rmtree(tmp, ignore_errors=True)
            except BaseException:  # noqa: BLE001
                pass

        if not isinstance(preds, list) or len(preds) != len(grade):
            raise ValueError("triage must return a list with one prediction per study")
        risk_by_uid = {}
        for p in preds:
            if not isinstance(p, dict) or "study_uid" not in p or "risk" not in p:
                raise ValueError("each prediction needs study_uid and risk")
            uid = p["study_uid"]
            if uid not in valid_uids:
                raise ValueError(f"unknown study_uid in prediction: {uid}")
            r = float(p["risk"])
            if not math.isfinite(r):
                raise ValueError("risk must be finite")
            risk_by_uid[uid] = min(1.0, max(0.0, r))
        if len(risk_by_uid) != len(valid_uids):
            raise ValueError("predictions must cover every study exactly once")

        _set_phase("grader")
        raw, details = score(grade, risk_by_uid)
        reward = normalize(raw)
        out = {"metric": float(raw), "reward": round(float(reward), 6),
               "correctness": True, "errors": list(child_notes),
               "details": {**details, "baseline": BASELINE, "reference": REFERENCE,
                           "sota_point": sota_point()}}
    except Exception as exc:  # noqa: BLE001 — any failure -> reward 0, never crash
        # the timeout / returncode facts come FIRST. Without them this line
        # was a bare "ValueError: submission produced no predictions" for three completely
        # different causes (wall-clock kill, OOM kill, genuinely empty output).
        detail = f"{type(exc).__name__}: {exc}"
        if _PHASE == "grader":
            _GRADER_FAILED = True
            detail = GRADER_FAILED_PREFIX + detail
        out = {"metric": None, "reward": 0.0, "correctness": False,
               "errors": list(child_notes) + [detail],
               "grader_failed": bool(_GRADER_FAILED), "phase": _PHASE}
    # no `finally` restore any more -- GRADE is unlinked on purpose and never re-read.

    rewards = {"reward": float(out["reward"]),
               "triage_utility_pct": float(out["metric"]) if out["metric"] is not None else 0.0}
    if _GRADER_FAILED:
        rewards["grader_failed"] = 1
    (REWARD_DIR / "reward.txt").write_text(f"{out['reward']}\n", encoding="utf-8")
    (REWARD_DIR / "reward.json").write_text(json.dumps(_reward_json_numeric(rewards)), encoding="utf-8")
    (REWARD_DIR / "grade_debug.json").write_text(json.dumps(out), encoding="utf-8")
    (REWARD_DIR / "score_details.json").write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps(out))


def _emergency_zero(exc: BaseException) -> None:
    """Last-resort reward writer. Every exit path must leave a reward on disk, otherwise a
    reward file planted by the submission before it crashed the grader would be what survives.
    """
    doc = {"metric": None, "reward": 0.0, "correctness": False,
           "errors": [f"{GRADER_FAILED_PREFIX}{type(exc).__name__}: {str(exc)[:500]}"],
           "grader_failed": True, "phase": _PHASE}
    try:
        REWARD_DIR.mkdir(parents=True, exist_ok=True)
        (REWARD_DIR / "reward.txt").write_text("0.0\n", encoding="utf-8")
        (REWARD_DIR / "reward.json").write_text(
            json.dumps(_reward_json_numeric({"reward": 0.0, "triage_utility_pct": 0.0},
                                            grader_failed=True)), encoding="utf-8")
        (REWARD_DIR / "grade_debug.json").write_text(json.dumps(doc), encoding="utf-8")
        (REWARD_DIR / "score_details.json").write_text(json.dumps(doc), encoding="utf-8")
    except BaseException:  # noqa: BLE001 — /logs itself may be gone; nothing left to do
        pass


if __name__ == "__main__":
    # BaseException, not Exception: SystemExit / KeyboardInterrupt / MemoryError / a full disk
    # must also land on 0. Only an uncatchable SIGKILL gets past here, and test.sh's exit-code
    # gate catches that.
    try:
        main()
    except BaseException as _exc:  # noqa: BLE001
        _emergency_zero(_exc)
        sys.exit(1)   # keep the non-zero rc so test.sh's gate agrees with what is on disk
    if _GRADER_FAILED:
        sys.exit(1)
