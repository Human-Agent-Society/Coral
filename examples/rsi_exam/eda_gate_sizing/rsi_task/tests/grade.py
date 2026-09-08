"""TWO-PHASE Harbor verifier for eda_gate_sizing (PARENT / trusted side).

Runs AFTER the agent finishes. Harbor copies tests/ into the box at verify time,
so the sealed held-out under tests/heldout/ was never visible to the agent. The
verifier RE-RUNS the agent's submitted solve() on the held-out cases, i.e. it
executes untrusted code — so it uses the two-phase pattern:

  PARENT (this file, trusted — NEVER imports the submission):
    1. read the sealed anchor table as root, BEFORE dropping privileges,
    2. spawn a CHILD (`python3 child_solve.py`) as an unprivileged uid, in its
       own session, that imports + runs solve() and writes one `<case>.size`
       per held-out case; the child never sees the scorer, the anchors or the
       reward path, and cannot write /logs,
    3. SIGKILL the child's process group, then for each case:
         a. structural legality gate (tools/legality.check_case) — trusted,
         b. score.score_case(...) — power under timing/DRV via OpenROAD,
    4. map EACH case to reward first, then average, and write
       /logs/verifier/reward.{txt,json} + score_details.json.

Reward band (lower-is-better metric). Three anchors per
case: the do-nothing template (B -> 0.0), the hidden expert solution (S -> 0.6)
and the theoretical bound (U = 0 -> 1.0). There is no 0.3 reference tier: no
measured solution earns one; leave an anchor out rather than invent it.

The band is piecewise linear in u(m) = log(m + shift):

    m >= B        -> 0.0                                  at or below the start
    B  > m >= S   -> 0.6 * (u(B) - u(m)) / (u(B) - u(S))   start -> expert
    S  > m        -> 0.6 + 0.4 * (u(S) - u(m)) / (u(S) - u(U))   expert -> bound
    result clamped to [0, 1]

Why log: the metric spans ~3 decades between B and S, so a linear map on the raw
scale spends most of the band on the first, easiest decade and leaves everything
past the expert worth ~0.001 reward. On a log scale every 10x improvement is
priced roughly the same.

Why the shift: the upper anchor is at m = 0 and log(0) is undefined, so the scale
is log(m + shift). The shift is what makes the theoretical bound a FINITE point on
the axis; it also sets how much resolution the top segment gets, which is why it
lives in the sealed anchor table rather than in this file.
"""
from __future__ import annotations

import json
import math
import os
import pwd
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))
import legality  # noqa: E402  (trusted, ships in tests/tools)
import score as scorer  # noqa: E402  (metric lives here; swappable)

# --- HARD-CODED constants (never os.environ.get — the host must not be
#     able to weaken the grader by setting a container env var) ---------------
ANCHOR_FILE = Path("/tests/anchors.json")     # root-owned, 0400
SUBMISSION_DIR = Path("/app/methods/main")
HELDOUT = Path("/tests/heldout")
REWARD_DIR = Path("/logs/verifier")
CHILD = Path("/tests/child_solve.py")
SOLVER_UID = 65534                            # nobody
SOLVER_GID = 65534
# Time budget promised to the submission, the slack before it counts as over
# budget, and the wall that actually kills it (the three must differ, or
# the soft path is dead code).
TIME_BUDGET_SEC = 2400.0                      # per case, enforced by the child
GRACE_SEC = 15.0
HARD_CAP_SEC = TIME_BUDGET_SEC * 2 + 120.0    # 2 held-out cases + startup
# thread count is a constant equal to [verifier] cpus, not a function of
# how many cores the host happens to have. Both Dockerfiles set the same ENV.
THREADS = "8"
CHILD_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/tmp",
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": THREADS,
    "OPENBLAS_NUM_THREADS": THREADS,
    "MKL_NUM_THREADS": THREADS,
    "NUMEXPR_NUM_THREADS": THREADS,
}
GRADER_FAIL = "grader failed: "               # infra failure marker


def _load_anchors() -> dict:
    """Read the sealed table as root, before any privilege drop."""
    a = json.loads(ANCHOR_FILE.read_text())
    for case, v in a["cases"].items():
        if not (v["baseline"] > v["sota"] > a["upper"]):
            raise ValueError(f"anchors out of order for {case}: {v}")
    if not a["log_shift"] > 0:
        raise ValueError("log_shift must be positive (it keeps log(upper) finite)")
    return a


def _list_cases() -> list[str]:
    return sorted(p.name for p in HELDOUT.iterdir()
                  if p.is_dir() and (p / "IR_Tables").exists())


def _assert_readable_as_solver(path: Path) -> None:
    """The dropped uid must really be able to read the submission. Checking
    st_mode is not enough (parent-dir x bits, ACLs, mount options), so fork a
    child that setuids and actually opens it. A silent failure here looks
    exactly like 'the agent submitted a broken solver'."""
    pid = os.fork()
    if pid == 0:  # child
        try:
            os.setgroups([])
            os.setgid(SOLVER_GID)
            os.setuid(SOLVER_UID)
            (path / "solver.py").open("rb").close()
            os._exit(0)
        except Exception:  # noqa: BLE001
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    if os.waitstatus_to_exitcode(status) != 0:
        raise PermissionError(
            f"{path} is not readable by uid {SOLVER_UID}; the artifacts dir must "
            f"be world-readable or the grader silently scores 0")


def _drop_privileges() -> None:
    """Drop to the solver uid. setgroups() FIRST and it is not optional: setgid()
    replaces only the primary group, the SUPPLEMENTARY list is inherited, and the
    parent runs as root — so without this the child keeps group 0 and can write
    every root:root 0664 file in the image. Found by the escape test,
    which wrote to /tests/score.py from the "unprivileged" child."""
    os.setgroups([])
    os.setgid(SOLVER_GID)
    os.setuid(SOLVER_UID)


def _run_child(out_dir: Path) -> None:
    proc = subprocess.Popen(
        [sys.executable, str(CHILD), str(SUBMISSION_DIR), str(HELDOUT),
         str(out_dir), str(TIME_BUDGET_SEC), str(GRACE_SEC)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True, env=CHILD_ENV, cwd="/tmp",
        preexec_fn=_drop_privileges,
    )
    try:
        stdout, _ = proc.communicate(timeout=HARD_CAP_SEC)
        if stdout:
            print(stdout, file=sys.stderr)
    except subprocess.TimeoutExpired:
        print(f"child hit the hard cap after {HARD_CAP_SEC}s; scoring whatever "
              f"it managed to write", file=sys.stderr)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _score_case(case: str, out_dir: Path) -> tuple[float, dict]:
    """Trusted: legality gate then real score + its per-term breakdown.
    Raises on illegal / failure."""
    size_path = out_dir / f"{case}.size"
    rep = legality.check_case(HELDOUT / case / "IR_Tables", size_path)
    if not rep["legal"]:
        raise ValueError(f"illegal sizing: {rep['errors'][:3]}")
    return scorer.score_case_detailed(HELDOUT / case, size_path)


def _case_reward(metric: float, baseline: float, sota: float, upper: float,
                 shift: float, sota_reward: float) -> float:
    """Piecewise-linear in log(m + shift) for one case. See the module docstring."""
    def u(v: float) -> float:
        return math.log(max(v + shift, 1e-12))

    if metric >= baseline:
        return 0.0
    if metric >= sota:
        return sota_reward * (u(baseline) - u(metric)) / (u(baseline) - u(sota))
    top = (1.0 - sota_reward) * (u(sota) - u(metric)) / (u(sota) - u(upper))
    return min(1.0, sota_reward + top)


def main() -> None:
    REWARD_DIR.mkdir(parents=True, exist_ok=True)
    details: dict = {"cases": {}, "errors": []}
    reward = 0.0
    metric = None
    try:
        anchors = _load_anchors()          # as root, before the drop
        cases = _list_cases()
        if not cases:
            raise FileNotFoundError("no held-out cases found")
        missing = [c for c in cases if c not in anchors["cases"]]
        if missing:
            raise KeyError(f"no anchors for held-out cases {missing}")
        _assert_readable_as_solver(SUBMISSION_DIR)

        with tempfile.TemporaryDirectory(prefix="gs_out_") as tmp:
            out_dir = Path(tmp)
            os.chown(out_dir, SOLVER_UID, SOLVER_GID)
            os.chmod(out_dir, 0o700)
            _run_child(out_dir)            # untrusted code runs ONLY here

            rewards, metrics = [], []
            for case in cases:
                a = anchors["cases"][case]
                brk = None
                try:
                    m, brk = _score_case(case, out_dir)
                    r = _case_reward(m, a["baseline"], a["sota"],
                                     anchors["upper"], anchors["log_shift"],
                                     anchors["sota_reward"])
                except Exception as exc:  # noqa: BLE001 — submission-side failure
                    m, r = None, 0.0
                    details["errors"].append(f"{case}: {exc}")
                details["cases"][case] = {
                    "metric": (round(m, 6) if m is not None else None),
                    "reward": round(r, 6),
                    "baseline": a["baseline"], "sota": a["sota"],
                    # which of the four terms the score came from (F5 diagnosis)
                    "breakdown": ({k: round(v, 6) for k, v in brk.items()}
                                  if brk is not None else None),
                }
                rewards.append(r)
                if m is not None:
                    metrics.append(m)

        # map per case FIRST, then average. The two held-out designs differ
        # by ~3x in score, so averaging raw scores would let the larger one
        # decide the whole reward.
        reward = sum(rewards) / len(rewards)
        metric = sum(metrics) / len(metrics) if len(metrics) == len(cases) else None
        details["correctness"] = len(metrics) == len(cases)
    except Exception as exc:  # noqa: BLE001 — GRADER-side failure, not the submission's
        # mark infra failures so a broken grading image is not silently
        # recorded as "every agent failed". reward stays 0.0.
        reward, metric = 0.0, None
        details["errors"].append(f"{GRADER_FAIL}{type(exc).__name__}: {exc}")
        details["correctness"] = False

    payload = {"reward": round(reward, 6),
               "mean_score": float(metric) if metric is not None else 0.0}
    if details["errors"]:
        payload["error"] = details["errors"][0]
    (REWARD_DIR / "reward.json").write_text(json.dumps(payload))
    (REWARD_DIR / "reward.txt").write_text(f"{round(reward, 6)}\n")
    (REWARD_DIR / "score_details.json").write_text(json.dumps(details, indent=2))
    print(json.dumps({"reward": payload["reward"], "mean_score": payload["mean_score"],
                      "errors": details["errors"]}))


if __name__ == "__main__":
    main()
