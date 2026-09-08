#!/usr/bin/env python3
"""Sealed grader for feeder_phase_impedance_inversion_v1 (trusted parent).

Two-phase sealed protocol (c201/c107 pattern):
  1. Parent (this process; stdlib+numpy, never imports agent code) loads
     the baked sealed fleets (AMI records + record-error truth, generated
     once at authoring time by tests/heldout/heldout_gen.py from secret
     seeds) entirely INTO MEMORY, then -- inside the verifier image --
     DELETES the sealed directory from disk before any submitted code
     runs.
  2. For each feeder the parent writes a truth-stripped record (byte-
     identical to environment/data/assessment/records.json) to a fresh
     temp dir, copies run_one.py next to it, and runs the submitted
     method there in an isolated subprocess (scrubbed env, single-thread
     BLAS pinned, hard wall budget). The parent alone compares the
     prediction to the in-memory truth.

Metric per feeder, in [0,1], HIGHER better (0.0 also the score for a
crashed / over-budget / malformed run):
    0.45 * phase score   (accuracy over metered customers, rescaled so
                          copying the shipped GIS labels = 0, perfect = 1)
  + 0.40 * impedance score (1 - ||log path-cumulative R1/X1 errors|| /
                          ||log errors of the records-as-shipped||, over
                          head-to-metered-bus paths, clipped to [0,1])
  + 0.15 * tap score     (1 - |tap step error| / 4, clipped)
Family metric = mean over its feeders; aggregate = mean over families.
Reward = canonical piecewise map through the sealed landmarks read from
tests/anchors.json (ANCHOR_BASELINE -> 0, ANCHOR_SOTA -> 0.6,
ANCHOR_UPPER -> 1.0), never hardcoded here. Both segments are linear in
the raw metric and reward is CLAMPED at 1.0 above ANCHOR_UPPER, which is
the structural bound of the metric.

Determinism: scoring is pure numpy on baked bytes; the submitted method's
subprocess runs with OMP/OPENBLAS/MKL_NUM_THREADS pinned to a constant (discrete searches
amplify BLAS float-context differences; pinning is a spike-measured
requirement for exact reproducibility).
"""
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


# reward.json may carry finite numbers only; text diagnostics go to score_details.json.
_RJ_TEXT_KEYS = ("error", "errors", "status", "reason", "detail", "details",
                 "message", "messages", "_details", "note", "notes")


# Phase gate: a "grader" fault gets the grader-failed prefix + nonzero exit, a "submission" one does not.
GRADER_FAIL_PREFIX = "grader failed: "
_PHASE = "grader"


def _set_phase(phase: str) -> None:
    global _PHASE
    _PHASE = phase


def _is_grader_fault(text: str) -> bool:
    return str(text).strip().lower().startswith(GRADER_FAIL_PREFIX.strip().lower())


# reward.json keeps only reward and failure flags; metrics move to score_details.json.
_RJ_KEEP_PREFIXES = ("grader_failed", "submission_")
_RJ_SPILL: dict = {}          # metrics moved out of reward.json


def _rj_split(payload):
    """(kept in reward.json, spilled to score_details); the spill is recorded in _RJ_SPILL."""
    keep = {k: v for k, v in payload.items()
            if k == "reward" or k.startswith(_RJ_KEEP_PREFIXES)}
    spill = {k: v for k, v in payload.items() if k not in keep}
    _RJ_SPILL.update(spill)
    return keep, spill


def _rj_merge(details):
    """Merge spilled metrics into score_details; an existing key wins."""
    merged = dict(details or {})
    for k, v in _RJ_SPILL.items():
        merged[k if k not in merged else "reward_json_" + k] = v
    return merged

def _reward_json_numeric(payload, grader_failed=None):
    """reward.json payload with finite numbers only (bool -> int, non-numeric keys dropped).

    grader_failed None means: infer it from marker strings that START with the prefix.
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
        """Yield the strings one by one; prefix matching must not see them concatenated."""
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
        # Non-numeric: drop the key (never coerce a null metric to 0.0) after distilling flags.
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
        # Two gates: prefix at string start, and the current phase is grader-side.
        grader_failed = (globals().get("_PHASE", "grader") == "grader"
                         and any(s.startswith(_pfx) for s in _marker_texts))
    if grader_failed:
        flags["grader_failed"] = 1

    out.update(flags)
    if "reward" not in out:
        out["reward"] = 0.0
    return out


HERE = Path(__file__).resolve().parent
SUBMISSION_DIR = Path("/app/methods/main")
# Hardcoded: an env override of either path would weaken grading.
HELDOUT_DIR = Path("/heldout")
LOG_DIR = Path("/logs/verifier")
# the three constants below are LIBRARY-UNIFORM across all six
# budget-enforcing tasks -- GRACE_SEC = 15.0 and HARD_CAP_SEC = TIME_BUDGET_SEC + 30.0.
# GRACE is the slack before a run is *flagged* over budget; HARD_CAP is the wall at which
# the child is *killed*. They must differ: if HARD_CAP equalled BUDGET+GRACE the soft
# check could never fire, because the child is killed exactly at the threshold it is
# tested against. Only TIME_BUDGET_SEC is task-specific.
TIME_BUDGET_SEC = 300.0
GRACE_SEC = 15.0
HARD_CAP_SEC = TIME_BUDGET_SEC + 30.0

# the submitted method's subprocess is dropped to this dedicated unprivileged
# account (created in tests/Dockerfile). It cannot write /logs or /tests, cannot signal
# this root parent, and cannot read /proc/1/environ or the 0400 anchors -- holes A/B/D/E.
SOLVER_UID = 4242
SOLVER_GID = 4242
# Anchors live in a root-only 0400 file, never in the environment: anything the
# harness injects as env also shows up in PID 1's /proc/1/environ for any submission to read.


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
ANCHOR_FILE = HERE / "anchors.json"

EVAL_ORDER = ["indist", "indist_hard"]
THREAD_PINS = {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
               "MKL_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2"}


def load_anchors() -> dict:
    """Read the sealed anchors from the root-only file as the trusted parent (before any
    privilege drop). No environment fallback and no hardcoded default: a missing anchor
    must fail loudly (-> reward 0 via the outer guard), never silently score against a
    stale constant."""
    with open(ANCHOR_FILE) as f:
        return json.load(f)


def anchor(anchors: dict, key: str, required: bool = True):
    if key not in anchors or key.startswith("_"):
        if required:
            raise RuntimeError(f"anchor {key} missing from {ANCHOR_FILE}")
        return None
    return float(anchors[key])


# Root-only sentinel that only the image build can create; absent on the authoring host.
VERIFIER_IMAGE_MARKER = HERE / ".verifier_image"


def seal_disk() -> None:
    """Inside the verifier image, remove the sealed sources (generator
    with secret seeds + baked truth) from disk so the submitted method's
    subprocess can never read them. Local (authoring-host) runs keep the
    repo intact: only the image build creates VERIFIER_IMAGE_MARKER;
    nothing the agent controls can."""
    if VERIFIER_IMAGE_MARKER.exists():
        shutil.rmtree(HELDOUT_DIR, ignore_errors=True)


def validate_pred(pred, feats):
    """Return the normalized prediction dict or None if malformed.

    N1 complementary fix (2026-08-03, wave 2). This function is called back in the
    GRADER phase (see run_solver), so an exception escaping it is reported as a grader
    fault. That is only sound if nothing the SUBMISSION controls can raise out of here,
    hence the split:

      * this wrapper touches only `feats` -- sealed, grader-owned data. A KeyError /
        TypeError here really is the grader's own bug (or corrupted sealed data) and
        must surface as `grader failed:` + grader_failed=1, not as "the submission
        scored 0";
      * `_validate_pred_untrusted` touches only the submission's bytes, and EVERY
        exception it raises is swallowed into `None` == "malformed prediction".
        Without that blanket catch, moving this call into the grader phase would hand
        the submission a wash-score channel (e.g. a ragged `z_rx` makes np.asarray
        raise ValueError, `phase: [1e999]` makes int(inf) raise OverflowError) --
        it could turn its own garbage into an infra failure and delete the 0.
    """
    L = len(feats["loads"])
    S = len(feats["segments"])
    seg_codes = [s["rec_code"] for s in feats["segments"]]
    catalog = feats["catalog"]
    try:
        return _validate_pred_untrusted(pred, L, S, seg_codes, catalog)
    except Exception:       # noqa: BLE001 -- see docstring: submission-controlled input
        return None


def _validate_pred_untrusted(pred, L, S, seg_codes, catalog):
    """Normalize the UNTRUSTED prediction. Never let an exception escape to the caller
    with a meaning other than "malformed" -- the caller converts any escape into None."""
    if not isinstance(pred, dict):
        return None
    try:
        phase = [int(p) for p in pred["phase"]]
        tap = float(pred["tap_steps"])
    except (KeyError, TypeError, ValueError):
        return None
    if len(phase) != L or any(p not in (0, 1, 2) for p in phase):
        return None
    if not np.isfinite(tap):
        return None
    out = {"phase": phase, "tap_steps": tap}
    z_rx = pred.get("z_rx")
    if z_rx is not None:
        arr = np.asarray(z_rx, float)
        if arr.shape != (S, 2) or not np.all(np.isfinite(arr)):
            return None
        out["z_rx"] = arr.tolist()
        out["code"] = list(seg_codes)
        out["scale"] = [1.0] * S
        return out
    try:
        code = [str(c) for c in pred["code"]]
        scale = [float(x) for x in pred["scale"]]
    except (KeyError, TypeError, ValueError):
        return None
    if len(code) != S or len(scale) != S:
        return None
    if any(c not in catalog for c in code):
        return None
    if not all(np.isfinite(x) and x > 0 for x in scale):
        return None
    out["code"] = code
    out["scale"] = scale
    return out


def score_instance(inst, pred):
    """Identical definition to environment/selfcheck.py (raw metric)."""
    tr = inst["truth"]
    met = inst["meter_load_idx"]
    tp = np.array(tr["phase"])[met]
    gp = np.array([ld["gis_phase"] for ld in inst["loads"]])[met]
    pp = np.array(pred["phase"])[met]
    acc = float((pp == tp).mean())
    acc_gis = float((gp == tp).mean())
    phase_score = float(np.clip((acc - acc_gis) /
                                max(1.0 - acc_gis, 0.05), 0.0, 1.0))

    segs = inst["segments"]
    parent = {s["tb"]: (s["fb"], s["id"]) for s in segs}

    def path_ids(bus):
        out = []
        b = bus
        while b in parent:
            fb, sid = parent[b]
            out.append(sid)
            b = fb
        return out

    met_paths = [path_ids(inst["loads"][li]["bus"]) for li in met]

    def seg_rx(codes, scales, z_rx=None):
        if z_rx is not None:
            arr = np.clip(np.array(z_rx, float), 1e-4, None)
            return arr[:, 0], arr[:, 1]
        r = np.array([inst["catalog"][c]["r1"] * s["km"] * sc
                      for s, c, sc in zip(segs, codes, scales)])
        x = np.array([inst["catalog"][c]["x1"] * s["km"] * sc
                      for s, c, sc in zip(segs, codes, scales)])
        return r, x

    def pathvec(r, x):
        out = []
        for pth in met_paths:
            out += [sum(r[i] for i in pth), sum(x[i] for i in pth)]
        return np.array(out)

    z_true = pathvec(*seg_rx(tr["code"], tr["scale"]))
    z_pred = pathvec(*seg_rx(pred["code"], pred["scale"],
                             pred.get("z_rx")))
    z_nom = pathvec(*seg_rx([s["rec_code"] for s in segs],
                            np.ones(len(segs))))
    e_pred = np.log(z_pred / z_true)
    e_nom = np.log(z_nom / z_true)
    imp_score = float(np.clip(
        1.0 - np.linalg.norm(e_pred) / max(np.linalg.norm(e_nom), 1e-9),
        0.0, 1.0))

    dt = abs(float(pred["tap_steps"]) - tr["tap_steps"])
    tap_score = float(np.clip(1.0 - dt / 4.0, 0.0, 1.0))

    total = 0.45 * phase_score + 0.40 * imp_score + 0.15 * tap_score
    return {"phase_score": phase_score, "imp_score": imp_score,
            "tap_score": tap_score, "score": total}


def _describe_rc(rc: int) -> str:
    """Turn a raw child return code into something a reader can act on.

    Signal deaths are the interesting case: Popen reports them as a NEGATIVE returncode, and a
    shell-style 128+N is what a wrapper would report. -9 / 137 is what an OOM kill looks like from
    here. SIGBUS (-7 / 135) gets its own wording: it is what writing past the container's 64 MiB
    /dev/shm looks like, which must never read as "the submission produced nothing".
    """
    _SHM_HINT = (" -- this is what writing past the container's 64 MiB /dev/shm looks like "
                 "(e.g. a multiprocessing/joblib shared_memory buffer); the segment is created "
                 "successfully and only the first write faults, so it is NOT 'the submission "
                 "produced nothing'")
    if rc < 0:
        sig = -rc
        name = signal.Signals(sig).name if sig in {s.value for s in signal.Signals} else f"signal {sig}"
        extra = " (typical of an out-of-memory kill)" if sig == signal.SIGKILL else ""
        if sig == signal.SIGBUS:
            extra = _SHM_HINT
        return f"submission child was killed by {name} (returncode {rc}){extra}"
    if rc > 128:
        sig = rc - 128
        extra = " (typical of an out-of-memory kill)" if sig == signal.SIGKILL else ""
        if sig == signal.SIGBUS:
            extra = _SHM_HINT
        return f"submission child exited {rc} = killed by signal {sig}{extra}"
    return f"submission child exited with returncode {rc}"


# Reads of submission-owned paths must be bounded, non-blocking and must not follow symlinks.
SUBMISSION_OUTPUT_LIMIT_BYTES = 64 << 20   # a legit pred.json is KBs
CHILD_ERR_TAIL_BYTES = 4096


def _open_regular_ro(path) -> int:
    fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"{path} is not a regular file")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _read_submission_text(path, limit=SUBMISSION_OUTPUT_LIMIT_BYTES) -> str:
    """Bounded read of submission output; callers fold any exception into malformed output."""
    fd = _open_regular_ro(path)
    try:
        chunks, got = [], 0
        while got < limit:
            block = os.read(fd, min(1 << 20, limit - got))
            if not block:
                break
            chunks.append(block)
            got += len(block)
    finally:
        os.close(fd)
    return b"".join(chunks).decode("utf-8", "replace")


def _child_stream_tail(path) -> str:
    """Last few KB of the child stderr, diagnostics only; degrades silently to an empty string."""
    try:
        fd = _open_regular_ro(path)
    except (OSError, ValueError):
        return ""
    try:
        size = os.fstat(fd).st_size
        if size > CHILD_ERR_TAIL_BYTES:
            os.lseek(fd, size - CHILD_ERR_TAIL_BYTES, os.SEEK_SET)
        raw = os.read(fd, CHILD_ERR_TAIL_BYTES)
    except OSError:
        return ""
    finally:
        os.close(fd)
    return raw.decode("utf-8", "replace")


def _drop_priv():
    """setuid preexec for the child: drop the group first, wipe supplementary groups,
    then the user. Runs in the forked child before exec. (Kept as an explicit preexec_fn
    rather than the user=/group= kwargs so the ordering is unambiguous and auditable.)"""
    os.setgroups([])
    os.setgid(SOLVER_GID)
    os.setuid(SOLVER_UID)


def run_solver(features: dict) -> "tuple[dict | None, list]":
    """Run one feeder in an isolated, UNPRIVILEGED subprocess. Returns (pred_or_None, notes).

    `notes` is a LIST of independent facts, not one blended string: the timeout fact, the
    child's returncode (via _describe_rc) and "produced nothing" each get their own line.

    Hardening vs. the original subprocess.run(capture_output=True):
      * child dropped to uid 4242 (holes A/B/D/E): it can neither seed /logs, tamper /tests,
        kill this parent, nor read the sealed anchors / /proc/1/environ;
      * child stdout/stderr go to FILES, not pipes -- a grandchild the solver detaches with
        setsid can no longer hold the read end open and hang the grader (PATTERN pit #5);
      * start_new_session=True + killpg in finally reaps any surviving grandchildren;
      * the child temp dir is chown'd to 4242 so the dropped child can still write pred.json,
        while /logs and /tests stay root-owned and unwritable.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        inst_f = td / "record.json"
        out_f = td / "pred.json"
        inst_f.write_text(json.dumps(features))
        runner = td / "run_one.py"
        runner.write_text(RUN_ONE_SRC)
        # the dropped child owns its scratch dir (so it can write pred.json + caches) but
        # nothing else; record/runner stay root-owned and world-readable.
        os.chmod(inst_f, 0o644)
        os.chmod(runner, 0o644)
        os.chown(td, SOLVER_UID, SOLVER_GID)
        os.chmod(td, 0o700)
        child_out = td / "_stdout.txt"
        child_err = td / "_stderr.txt"
        env = {k: v for k, v in os.environ.items()
               if k not in ("HELDOUT_DIR", "VERIFIER_LOG_DIR",
                            "SUBMISSION_DIR", "VERIFIER_IMAGE")
               and not k.startswith("ANCHOR_")}
        env.update(THREAD_PINS)
        # give the unprivileged child writable cache/tmp locations inside its own scratch dir
        env.update({"HOME": str(td), "TMPDIR": str(td),
                    "XDG_CACHE_HOME": str(td), "MPLCONFIGDIR": str(td)})
        t0 = time.monotonic()
        fout = open(child_out, "w")
        ferr = open(child_err, "w")
        try:
            proc = subprocess.Popen(
                [sys.executable, str(runner),
                 "--solver-dir", str(SUBMISSION_DIR),
                 "--record", str(inst_f), "--out", str(out_f)],
                stdin=subprocess.DEVNULL, stdout=fout, stderr=ferr,
                cwd=str(td), env=env,
                start_new_session=True, preexec_fn=_drop_priv)
        except Exception as exc:  # noqa: BLE001 -- spawn failure is a failed grade, not a crash
            fout.close(); ferr.close()
            # Child failed to spawn: grader-side, so this round is an infra failure, not a 0.
            return None, [f"{GRADER_FAIL_PREFIX}submission child spawn failed: "
                          f"{type(exc).__name__}: {exc}"]
        # The submission window covers only waiting on the child and reading its pred.json.
        notes: list = []
        _set_phase("submission")
        try:
            rc = proc.wait(timeout=HARD_CAP_SEC)
        except subprocess.TimeoutExpired:
            rc = 124
            # The timeout fact gets its own line in the mandated wording; no grader prefix.
            notes.append(f"submission timed out after {HARD_CAP_SEC:.0f}s")
        finally:
            # Reaping, and everything after it, is grader-side again.
            _set_phase("grader")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
            fout.close(); ferr.close()
        wall = time.monotonic() - t0
        # Submission-owned path, so stay in the submission phase; the helper never raises.
        _set_phase("submission")
        try:
            tail = _child_stream_tail(child_err)
        finally:
            _set_phase("grader")
        # rc 124 is our own timeout marker, already reported; anything else writes the returncode out.
        if rc != 0 and rc != 124:
            notes.append(_describe_rc(rc))
            if tail.strip():
                notes.append(f"submission child stderr tail: {tail.strip()[-300:]}")
        elif not notes and wall > TIME_BUDGET_SEC + GRACE_SEC:
            notes.append(f"submission ran {wall:.0f}s, over the {TIME_BUDGET_SEC:.0f}s "
                         f"per-feeder budget (+{GRACE_SEC:.0f}s grace)")
        if rc != 0 or wall > TIME_BUDGET_SEC + GRACE_SEC:
            return None, notes          # already back in the grader phase
        # Reading and parsing the submission's own pred.json: submission window.
        _set_phase("submission")
        try:
            pred = json.loads(_read_submission_text(out_f))["pred"]
        except Exception:   # noqa: BLE001
            # N1 complementary fix (2026-08-03, wave 2): the catch used to be
            # (OSError, ValueError, KeyError), which let a few shapes of a hostile /
            # broken pred.json escape as an exception instead of a verdict -- a
            # top-level JSON list makes the ["pred"] subscript raise TypeError, and a
            # multi-GB pred.json makes read_text raise MemoryError. Those escapes landed
            # in the submission phase, so they were scored 0 rather than washed, but the
            # window has to be exact, not accidentally-correct: everything decoded from
            # the child's own bytes is a submission verdict, full stop.
            #
            # Reached only when the child exited 0 within budget: it really did produce
            # nothing usable. This is the ONE case that is genuinely "submission produced
            # no output", and it is now distinguishable from every case above.
            notes.append("submission exited 0 within budget but produced no valid pred.json")
            return None, notes
        finally:
            # The output is in memory; from here on (validate_pred included) it is grader code.
            _set_phase("grader")
        out = validate_pred(pred, features)
        if out is None:
            notes.append("submission produced a pred.json that failed validate_pred "
                         "(shape / range / finiteness / catalog membership)")
        return out, notes


def piecewise(metric, base, sota, upper):
    """baseline->0, sota->0.6, upper->1."""
    pts = [(base, 0.0), (sota, 0.6), (upper, 1.0)]
    higher_better = pts[-1][0] > pts[0][0]
    x = metric if higher_better else -metric
    pts = [(v if higher_better else -v, rw) for v, rw in pts]
    if x <= pts[0][0]:
        return 0.0
    for (v0, r0), (v1, r1) in zip(pts, pts[1:]):
        if x <= v1:
            return float(r0 + (r1 - r0) * (x - v0) / max(1e-12, v1 - v0))
    return 1.0


def main() -> int:
    """Return the process exit code; a sys.exit here would be swallowed by the outer guard."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # phase 1: everything sealed goes to memory; disk copy removed in-image
    fleets = json.loads((HELDOUT_DIR / "instances_sealed.json")
                        .read_text())["families"]
    features, truths = {}, {}
    k = 0
    for fam in EVAL_ORDER:
        features[fam], truths[fam] = [], []
        for inst in fleets[fam]:
            feats = {kk: v for kk, v in inst.items() if kk != "truth"}
            feats["id"] = f"f{k:02d}"
            features[fam].append(feats)
            truths[fam].append(inst)
            k += 1
    seal_disk()

    # anchors read from the sealed root-only file, never the environment
    anchors = load_anchors()
    # prove the unprivileged solver account can really READ the graded artifacts
    # BEFORE handing them over. If it cannot, the child simply produces nothing and the
    # run scores 0 as though the submission were bad -- a silent, unattributable
    # failure. Raising here makes the cause land in the reward file instead.
    assert_artifacts_readable_by_solver(SUBMISSION_DIR)
    upper = anchor(anchors, "ANCHOR_UPPER")
    per_condition, cond_metrics, details = {}, [], {}
    errors: list = []
    for fam in EVAL_ORDER:
        vals, per_inst = [], []
        for feats, inst in zip(features[fam], truths[fam]):
            pred, notes = run_solver(feats)
            for n in notes:
                errors.append(f"{fam}/{feats['id']}: {n}")
            if pred is None:
                per_inst.append(None)   # invalid/over-budget: worst case
                vals.append(0.0)
                continue
            sc = score_instance(inst, pred)
            per_inst.append({kk: round(float(v), 6)
                             for kk, v in sc.items()})
            vals.append(float(sc["score"]))
        m = round(float(np.mean(vals)), 6)
        cond_metrics.append(m)
        details[fam] = per_inst
        # Per-family METRIC only; no per-family reward is emitted.
        per_condition[fam] = {"metric": m}

    metric = round(float(np.mean(cond_metrics)), 6)
    base = anchor(anchors, "ANCHOR_BASELINE")
    sota = anchor(anchors, "ANCHOR_SOTA")
    reward = round(piecewise(metric, base, sota, upper), 6)
    # reward.json must stay a flat dict of numbers; nested views and error strings go to score_details.
    flat = {"reward": reward, "metric": metric}
    for fam, entry in per_condition.items():
        flat[f"{fam}_metric"] = entry["metric"]
    # A prefixed note with no exception is still an infra failure: set the flag and exit nonzero.
    grader_fault = any(_is_grader_fault(e.split(": ", 1)[-1]) or _is_grader_fault(e)
                       for e in errors)
    if grader_fault:
        flat["grader_failed"] = 1
    _rj, _ = _rj_split(_reward_json_numeric(flat))
    (LOG_DIR / "reward.json").write_text(json.dumps(_rj))
    sd = {"per_condition": per_condition,
          "per_instance": details,
          "errors": errors,
          "anchors": {"baseline": base, "sota": sota, "upper": upper}}
    dbg = {"metric": metric, "reward": reward, "errors": errors}
    if grader_fault:
        # Aggregation matches the prefix at string start, so mint a top-level error string too.
        mark = GRADER_FAIL_PREFIX + "; ".join(
            e for e in errors if _is_grader_fault(e.split(": ", 1)[-1]) or _is_grader_fault(e))
        sd["error"] = mark
        dbg["error"] = mark
    (LOG_DIR / "score_details.json").write_text(json.dumps(_rj_merge(sd)))
    (LOG_DIR / "grade_debug.json").write_text(json.dumps(dbg))
    for e in errors:
        print("ERROR " + e, file=sys.stderr, flush=True)
    print(f"metric={metric:.6f} reward={reward:.6f} errors={len(errors)}")
    return 1 if grader_fault else 0


RUN_ONE_SRC = r'''#!/usr/bin/env python3
"""Run one feeder calibration in an isolated process (grader's child)."""
import argparse
import importlib.util
import json
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver-dir", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.solver_dir)
    spec = importlib.util.spec_from_file_location(
        "solver", a.solver_dir + "/solver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rec = json.load(open(a.record))
    pred = mod.calibrate(rec)
    out = {"phase": [int(p) for p in pred["phase"]],
           "tap_steps": float(pred["tap_steps"])}
    if pred.get("z_rx") is not None:
        out["z_rx"] = [[float(a2), float(b2)] for a2, b2 in pred["z_rx"]]
    else:
        out["code"] = [str(c) for c in pred["code"]]
        out["scale"] = [float(s) for s in pred["scale"]]
    json.dump({"pred": out}, open(a.out, "w"))


if __name__ == "__main__":
    main()
'''


if __name__ == "__main__":
    # Hole C defence-in-depth: every exit path must leave a reward file. BaseException (not
    # Exception) so SystemExit / KeyboardInterrupt / MemoryError are covered too. Only an
    # uncatchable SIGKILL slips past -- caught by test.sh's exit-code gate. This task's
    # TimeoutExpired is already handled inside run_solver; this is the last-resort net.
    try:
        _rc = main()
    except BaseException as exc:  # noqa: BLE001
        # Whether the grader-failed prefix is added depends on the current phase.
        _grader_side = (_PHASE == "grader")
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            # Last-resort reward.json: numbers only; the text goes to score_details.json.
            (LOG_DIR / "reward.json").write_text(json.dumps(
                {"reward": 0.0, "grader_failed": 1 if _grader_side else 0}))
            try:
                _pfx = GRADER_FAIL_PREFIX if _grader_side else ""
                _txt = f"{_pfx}{type(exc).__name__}: {str(exc)[:400]}"
                (LOG_DIR / "score_details.json").write_text(json.dumps({"error": _txt}))
                (LOG_DIR / "grade_debug.json").write_text(json.dumps(
                    {"reward": 0.0, "phase": _PHASE, "error": _txt}))
            except BaseException:
                pass
        except BaseException:
            pass
        # Grader-side: nonzero exit. Submission-side: 0, so that reward 0 counts as a score.
        sys.exit(1 if _grader_side else 0)
    sys.exit(_rc or 0)
