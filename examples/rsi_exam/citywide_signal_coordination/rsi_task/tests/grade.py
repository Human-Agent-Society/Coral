"""Trusted parent grader (never imports the agent's controller).

For each sealed (scale, seed) case in heldout/protocol.json it spawns run_eval.py as a
child that runs the SUMO sim with the agent's solution and writes tripinfo to a path
THIS process owns.  The parent parses that tripinfo and computes the frozen Cost
itself, then NORMALISES each case by that case's own no-op baseline Cost (stored in
protocol.json as `noop_baseline`) and averages the normalised costs.  Per-case
normalisation removes the hidden weighting a raw mean gives to the heaviest case.

Trust boundary, concretely:
  * tls.add.xml is snapshotted and whitelist-checked HERE, before any child exists, by
    eval_core.freeze_and_validate_additional().  This process never imports submitted
    code, so the whitelist cannot be monkeypatched away — a child doing
    `eval_core.ADD_ALLOWED_TAGS.add("rerouter")` changes nothing, because the child never
    runs the validator.  Only the already-validated snapshot path is passed down.
  * tripinfo files live in a private root-owned 0700 mkdtemp() directory, NOT under
    /logs/verifier: nothing that runs submitted code may write the numbers it is scored
    from, nor the reward directory.
  * the child's environment is CONSTRUCTED from constants (see CHILD_ENV), not filtered
    from this process's own: it never sees the BASELINE/REFERENCE/SOTA/UPPER anchors.
  * run_eval.py is TRUSTED and imports no submitted code.  It
    runs SUMO itself and starts `controller.py` as a further child through
    sim_rpc.ControllerHost, dropped to uid/gid 65534 via Popen(user=/group=) — after
    fork and before exec, so no submitted code ever runs as root.  That process reaches
    SUMO only through the sim_rpc policy filter (read-only calls + trafficlight setters),
    never learns the tripinfo path, and cannot write the directory holding it.  Before
    the merge the submission was imported INTO the SUMO process and was handed the
    tripinfo path in argv, so it could both author its own tripinfo and empty the network
    with `conn.vehicle.remove()`.
    tests/heldout/ (0500) and tests/anchors.json (0400) are root-owned and therefore
    unreadable by uid 65534; sealing_preflight() proves this by forking, setuid-ing and
    really open()-ing them.

Anchors (mean normalised Cost over the hidden protocol; lower is better; the no-op
baseline is exactly 1.0 by construction):
    BASELINE -> 0.0   (inherited no-op controller: normalised mean == 1.0)
    SOTA     -> 0.6   (release reference frontier, measured)
    UPPER    -> 1.0   (theoretical bound; deliberately unset here)
BASELINE -> SOTA is linear; at/above BASELINE the reward floors at 0.  With no UPPER,
above SOTA the reward follows the standard exponential tail 1 - 0.4 * exp(-(p - p0)/tau),
tau fixed by slope continuity at SOTA: strictly increasing, bounded by 1, never 1.
Unset anchors are reported as null in score_details.json — never 0.0, which on this
minimisation task would read as a theoretical-optimum anchor value.
"""
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_core import cost_from_tripinfo, freeze_and_validate_additional  # noqa: E402

# Every constant below is HARD-CODED on purpose.  There is no
# environment read anywhere in this file: anything an operator can set with `-e` is
# something that can weaken grading from the outside.  REWARD_DIR decides where the reward
# lands (a host-settable VERIFIER_OUT would have let `-e VERIFIER_OUT=/tmp` stop the
# grader ever overwriting a planted reward.json), ANCHOR_FILE decides what a submission
# is scored against, and SCENARIO_DATA decides which network is simulated.
REWARD_DIR = "/logs/verifier"
SOLUTION = "/app/methods/main"
ANCHOR_FILE = os.path.join(HERE, "anchors.json")
SCENARIO_DATA = "/data/scenario"

# The child's environment is CONSTRUCTED, not filtered: nothing of this process's own
# environment reaches submitted code.  The thread counts are pinned to "1" so a
# submission cannot make its own run non-deterministic or starve the box.
CHILD_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin",
             "HOME": "/tmp",
             "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
             "SCENARIO_DATA": SCENARIO_DATA,
             "PYTHONHASHSEED": "0",
             "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
             "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
             "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}

# --- privilege drop + sealing pre-flight ---------------------------
# Until 2026-08-05 the process that imports submitted code ran as the SAME uid as this
# grader (root), so the 0400 on tests/anchors.json and the 0500 on tests/heldout/
# excluded nobody -- a submission could simply open() the hidden cases and tune against
# them.  It is now dropped to uid/gid 65534 (nobody/nogroup) after fork and BEFORE exec,
# so not one line of submitted code has run as root.  Since the trust-boundary merge
# the process being dropped is the sim_rpc controller child (spawned inside run_eval.py
# with Popen(user=DROP_UID, group=DROP_GID)), not run_eval.py itself, which is trusted
# and must stay root to own the tripinfo directory.  _drop_privileges() below is kept
# for the pre-flight, which has to become that uid for real in order to prove the seals.
#
# The pre-flight below PROVES the drop, by doing it: it forks, setuid()s for real, and
# really open()s the files.  It deliberately does not use os.access() (which answers for
# the calling uid, and under root answers "yes" to almost everything) and does not read
# stat().st_mode on its own (a mode bit says what the file asks for, not whether the
# drop happened).
DROP_UID = 65534
DROP_GID = 65534


def _drop_privileges():
    """preexec_fn for the solver child: runs after fork, before exec."""
    os.setgroups([])
    os.setgid(DROP_GID)
    os.setuid(DROP_UID)


def _first_file(root):
    """Any regular file under `root`, for the positive half of the pre-flight."""
    for dirpath, _dirs, files in os.walk(root):
        for name in sorted(files):
            p = os.path.join(dirpath, name)
            if os.path.isfile(p):
                return p
    return None


def sealing_preflight(must_read, must_not_read):
    """Fork, really drop to DROP_UID, really open() the paths, and raise loudly on any
    disagreement.  A failure here is a GRADER fault (`grader failed:` -> infra failure,
    re-run), never a silent 0: a verifier image whose seals do not hold must not grade."""
    if os.getuid() != 0:
        print("[grade] pre-flight: not root, the privilege drop is a no-op on this host "
              "and sealing is NOT proven (local smoke run?)", file=sys.stderr)
        return
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:                                     # child: never returns
        problems = []
        try:
            os.close(r)
            _drop_privileges()
            if os.getuid() != DROP_UID or os.getgid() != DROP_GID:
                problems.append("setuid/setgid did not take effect")
            for p in must_read:
                try:
                    with open(p, "rb") as fh:
                        fh.read(1)
                except OSError as exc:
                    problems.append(f"submission artefact unreadable after the drop: "
                                    f"{p} ({type(exc).__name__})")
            for p in must_not_read:
                try:
                    with open(p, "rb") as fh:
                        fh.read(1)
                    problems.append(f"sealed artefact STILL READABLE after the drop: {p}")
                except OSError:
                    pass
            os.write(w, "\n".join(problems).encode()[:4000])
        except BaseException as exc:                 # noqa: BLE001
            try:
                os.write(w, f"pre-flight child raised {type(exc).__name__}: "
                            f"{exc}".encode()[:4000])
            except OSError:
                pass
        finally:
            os._exit(0)
    os.close(w)
    msg = b""
    while True:
        chunk = os.read(r, 4096)
        if not chunk:
            break
        msg += chunk
    os.close(r)
    os.waitpid(pid, 0)
    if msg.strip():
        raise RuntimeError("sealing pre-flight failed: " + msg.decode(errors="replace"))


def _kill_group(proc):
    """SIGKILL the child's whole process group (it is a session leader, see Popen below)."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def child_env():
    """Minimal environment for run_eval.py: no anchors, no host knobs, pinned threads."""
    return dict(CHILD_ENV)


def anchors():
    """Anchors on the NORMALISED cost scale, from the sealed root-owned mode-0400
    `tests/anchors.json` and NOWHERE ELSE.  There is deliberately no environment
    fallback: a fallback is exactly the hole `-e SOTA=...` walks through,
    and a host-side re-anchor run edits the file and rebuilds the image like everything
    else that decides a score.

    A missing or malformed file is a GRADER fault, not a zero: raise, and let main()
    tag it `grader failed:`.  BASELINE is 1.0 by construction; SOTA must be
    measured; UPPER is None when unset (never a 0.0 sentinel, which on this
    minimisation task would read as a theoretical-optimum anchor value)."""
    with open(ANCHOR_FILE) as fh:
        src = {str(k).upper(): v for k, v in json.load(fh).items()}
    # Accept the UPPER_BOUND spelling: unset is "key absent", so a rename would
    # silently drop the top anchor instead of failing.
    if "UPPER_BOUND" in src and "UPPER" not in src:
        src["UPPER"] = src.pop("UPPER_BOUND")

    def g(k):
        v = src.get(k, "")
        v = str(v).strip() if v is not None else ""
        return float(v) if v else None

    baseline = g("BASELINE")
    if baseline is None:
        baseline = 1.0  # exact under per-case normalisation
    sota = g("SOTA")
    if sota is None:
        raise RuntimeError(
            "SOTA anchor is unset: the release reference frontier has not been "
            "measured, so grading would use no measured anchor at all.  Measure it "
            "on a SUMO machine first — see _dev/reanchor_2026-08/RE_ANCHOR.md.")
    upper = g("UPPER")
    if not 0.0 < sota < baseline:
        # The tail divides by (baseline - sota); a non-positive width would turn the
        # whole band into nonsense silently instead of failing loudly.  And
        # sota >= baseline means the frontier stopped beating the no-op on the
        # current window — a re-anchor bug, not a gradable state.
        raise RuntimeError(
            f"anchor band is degenerate: need 0 < SOTA ({sota}) < BASELINE "
            f"({baseline}).  Re-measure on the v2 window before grading — see "
            "_dev/reanchor_2026-08/RE_ANCHOR.md.")
    # the anchors must be strictly ordered on the metric (lower cost is better), or
    # the piecewise map silently inverts a segment.  Checked over whatever is SET.
    ordered = [v for v in (baseline, sota, upper) if v is not None]
    if any(a <= b for a, b in zip(ordered, ordered[1:])):
        raise RuntimeError(f"anchor order violated (E3): {ordered}")
    return baseline, sota, upper


def reward_of(norm_cost, a):
    """baseline->0, frontier(SOTA)->0.6, upper->1."""
    baseline, sota, upper = a
    if norm_cost >= baseline:
        return 0.0
    # y counts BASELINE->SOTA widths; y == 1 exactly at SOTA.
    y = (baseline - norm_cost) / max(1e-9, baseline - sota)
    if norm_cost >= sota:
        return 0.6 * y
    if upper is not None:
        c = (baseline - sota) / max(1e-9, baseline - upper)
        p = (baseline - norm_cost) / max(1e-9, baseline - upper)
        return min(1.0, 0.6 + 0.4 * (p - c) / max(1e-9, 1.0 - c))
    # No UPPER: standard exponential tail 1 - 0.4 * exp(-(p - p0)/tau).  tau is fixed by
    # slope continuity at SOTA, which reduces the exponent to 1.5 * (y - 1).
    return 1.0 - 0.4 * math.exp(-1.5 * (y - 1.0))


def write_reward(reward, extra=None):
    """Both sinks, always together: reward.json AND reward.txt.  Which of the two the
    harness treats as authoritative is not settled, and an attacker who plants one wins
    outright if the grader only ever writes the other."""
    os.makedirs(REWARD_DIR, exist_ok=True)
    payload = {"reward": round(float(reward), 4)}
    if extra:
        payload.update(extra)
    with open(os.path.join(REWARD_DIR, "reward.json"), "w") as fh:
        json.dump(payload, fh)
    with open(os.path.join(REWARD_DIR, "reward.txt"), "w") as fh:
        fh.write(f"{payload['reward']:.4f}\n")
    return payload


def grade():
    os.makedirs(REWARD_DIR, exist_ok=True)
    protocol = json.load(open(os.path.join(HERE, "heldout", "protocol.json")))
    cases = protocol["cases"]
    missing = [c for c in cases if not c.get("noop_baseline")]
    if missing:
        raise RuntimeError(
            f"protocol.json: noop_baseline missing for {len(missing)} case(s) — the "
            "v2 (2026-08) re-anchor has not been run yet.  Measure the per-case no-op "
            "baselines on a SUMO machine first — see _dev/reanchor_2026-08/RE_ANCHOR.md.")
    a = anchors()

    # prove the seals hold against the uid the solver child is about to run as,
    # BEFORE any submitted code exists.  `must_read` is the declared artifacts dir
    # (task.toml `artifacts = ["/app/methods"]`) -- the child has to be able to import
    # the submission; `must_not_read` is everything that decides the score.
    _sub = _first_file(os.path.dirname(SOLUTION.rstrip("/")))
    sealing_preflight([p for p in (_sub,) if p],
                      [ANCHOR_FILE, os.path.join(HERE, "heldout", "protocol.json")])

    # Whitelist-check the additional file HERE, in the process that never imports the
    # submission, and hand the children only the validated snapshot.
    add_frozen, add_error = None, None
    try:
        add_frozen = freeze_and_validate_additional(SOLUTION)
        # The snapshot stays in its 0700 root-owned mkdtemp().  Since the trust-boundary merge the
        # only process that opens it is the trusted run_eval.py child (root), so it needs
        # no loosening at all -- and the submitted controller, which now runs as uid
        # 65534 in a different process, cannot swap the file after validation.
    except ValueError as exc:
        add_error = f"illegal tls.add.xml: {str(exc)[:200]}"

    # Two scratch directories, and the split IS the trust boundary:
    #   trip_dir     root-owned 0700 -- SUMO's tripinfo lives here.  The controller uid
    #                cannot write it, so even if it learned the path (e.g. by reading
    #                /proc/<ppid>/cmdline of the trusted run_eval process) it cannot
    #                author the file the Cost is parsed from.  Before the trust-boundary merge this
    #                was chmod 0777 because the process writing it was the one importing
    #                the submission.
    #   ctrl_scratch owned by DROP_UID 0700 -- HOME/TMPDIR/cwd for the controller
    #                process, the one place it may write.
    trip_dir = tempfile.mkdtemp(prefix="ara_trip_")
    os.chmod(trip_dir, 0o700)
    ctrl_scratch = tempfile.mkdtemp(prefix="ara_ctrl_")
    try:
        os.chown(ctrl_scratch, DROP_UID, DROP_GID)
    except (PermissionError, OSError):
        # not root (local smoke run): the drop is a no-op there anyway
        os.chmod(ctrl_scratch, 0o700)
    else:
        os.chmod(ctrl_scratch, 0o700)
    env = child_env()
    per, norms = [], []
    try:
        for i, case in enumerate(cases):
            scale, seed = case["scale"], case["seed"]
            base = float(case["noop_baseline"])
            if add_error:
                # A rejected additional file makes every case invalid; each one scores
                # that case's baseline instead of crashing the grader.
                per.append({"scale": scale, "seed": seed, "cost": None,
                            "noop_baseline": base, "norm_cost": 1.0,
                            "error": add_error})
                norms.append(1.0)
                continue
            trip = os.path.join(trip_dir, f"_trip_{i}.xml")
            # The 5th argument is ALWAYS passed, "" meaning "no additional file at all",
            # so the child never falls back to freezing/validating on its own side of
            # the boundary (a controller from an earlier case could otherwise have
            # dropped a tls.add.xml into the solution dir).
            argv = [sys.executable, os.path.join(HERE, "run_eval.py"),
                    SOLUTION, str(scale), str(seed), trip, add_frozen or "",
                    str(DROP_UID), str(DROP_GID), ctrl_scratch]
            # start_new_session: the child becomes its own process group leader, so the
            # timeout below kills any grandchild it forked as well -- and since the trust-boundary
            # merge the sim_rpc controller process is exactly such a grandchild.
            # Otherwise a process spawned from controller.py outlives the backstop and is
            # still running -- able to rewrite /logs/verifier/reward.json -- after grading
            # has finished.
            # No preexec_fn here any more: run_eval.py is TRUSTED (it imports no submitted
            # code and owns the tripinfo), and the drop to 65534 happens one level down,
            # in sim_rpc.ControllerHost, for the process that does import the submission.
            proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                env=env, start_new_session=True)
            try:
                stdout, stderr = proc.communicate(
                    timeout=protocol.get("case_timeout_s", 1200))
            except subprocess.TimeoutExpired:
                _kill_group(proc)
                proc.communicate()
                # Wall-clock backstop only (the deterministic budget is simulation time,
                # enforced inside eval_core).  A case that overruns scores the baseline
                # for that case; letting the exception escape used to kill grading
                # outright and write no reward at all.
                per.append({"scale": scale, "seed": seed, "cost": None,
                            "noop_baseline": base, "norm_cost": 1.0,
                            "error": "timeout"})
                norms.append(1.0)
                continue
            # Also on the success path: the direct child can exit cleanly while leaving a
            # forked grandchild alive, and that grandchild must not outlive grading.
            _kill_group(proc)
            counters = None
            if proc.returncode == 0 and stdout.strip():
                try:
                    counters = json.loads(stdout.strip().splitlines()[-1])
                except (json.JSONDecodeError, ValueError, TypeError, IndexError):
                    # A stray print() from the controller is enough to make the last line
                    # unparseable; degrade this case, never crash grading.
                    counters = None
            if not counters or not counters.get("ok") or not os.path.exists(trip):
                # a broken solution scores the baseline for this case
                # (never crashes grading)
                per.append({"scale": scale, "seed": seed, "cost": None,
                            "noop_baseline": base, "norm_cost": 1.0,
                            "error": (counters or {}).get("error", (stderr or "")[-300:])})
                norms.append(1.0)
                continue
            res = cost_from_tripinfo(trip, counters)
            os.remove(trip)
            norm = res["cost"] / base
            res.update(scale=scale, seed=seed, noop_baseline=base,
                       norm_cost=round(norm, 6))
            per.append(res)
            norms.append(norm)
    finally:
        shutil.rmtree(trip_dir, ignore_errors=True)
        shutil.rmtree(ctrl_scratch, ignore_errors=True)
        if add_frozen:
            shutil.rmtree(os.path.dirname(add_frozen), ignore_errors=True)

    mean_norm_cost = sum(norms) / len(norms)
    reward = reward_of(mean_norm_cost, a)
    out = write_reward(reward, {"mean_norm_cost": round(mean_norm_cost, 6)})
    json.dump({"mean_norm_cost": round(mean_norm_cost, 6), "per_case": per,
               "anchors": {"baseline": a[0], "sota": a[1], "upper": a[2]}},
              open(os.path.join(REWARD_DIR, "score_details.json"), "w"), indent=2)
    print(f"reward = {out['reward']}   mean_norm_cost = {out['mean_norm_cost']:.6f}")


def main():
    try:
        grade()
    except BaseException as exc:  # noqa: BLE001 — a crashed grader must still leave a 0
        try:
            write_reward(0.0, {"error": f"grader failed: {type(exc).__name__}: "
                                       f"{str(exc)[:300]}"})
        except BaseException:  # noqa: BLE001
            pass
        raise


if __name__ == "__main__":
    main()
