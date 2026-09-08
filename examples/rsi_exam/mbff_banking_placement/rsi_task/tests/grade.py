"""TWO-PHASE Harbor verifier for mbff_banking_placement (PARENT / trusted side).

Runs AFTER the agent finishes (Harbor copies tests/ into the box at verify time, so the
held-out under tests/heldout/ was never visible to the agent). This task's verifier must
RE-RUN the agent's submitted solve() on the sealed held-out testcases, which means untrusted
submission code executes inside the verifier container. Per docs/anti-cheat.md, that requires
the TWO-PHASE pattern:

    PARENT (this file, trusted — NEVER imports the submission):
      1. spawn a CHILD (`python3 child_solve.py`, a fresh process — NOT a fork) in its own
         session/process group; the child imports + runs the submission's solve() and writes
         a placement output per case into a private 0700 dir. The child never sees the scoring
         binaries or the reward path,
      2. SIGKILL the whole child process group, then for each held-out case run the bundled
         C++ checker/evaluator pipeline (sanity -> placement_checker -> preliminary-evaluator)
         on the child's output and parse "Final score",
      3. average the per-case scores into mean_final_score (lower is better), map to the Harbor
         reward, write /logs/verifier/reward.json + reward.txt.

Because the score is computed by THIS trusted process from the C++ tools, a submission cannot
monkeypatch the scoring or fake a "Final score".

Reward (the RL/eval scalar), lower-is-better, three anchors: weak template -> 0, the strong
C++ expert solver -> 0.6, and the relaxation lower bound of the cost -> 1.0. The bound is a
real one (see _reward), so 1.0 means "provably optimal", not "best we ever measured". Any
failure / illegal-on-any-case -> 0.
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

# --- Anchors: sealed file, NOT the environment and NOT literals in this file.
# The path is hard-coded for the same reason every other path here is: a stray
# `-e ANCHORS=/tmp/mine.json` would let the caller supply their own band. The file is 0400
# root:root and this module reads it at import, i.e. as root, long before _drop_privileges()
# hands the submission a uid that cannot open it.
#   baseline = mean_final_score of the weak template on tests/heldout          -> reward 0
#   sota     = mean_final_score of the strong C++ expert solver (solution/)    -> reward 0.6
#   upper    = relaxation lower bound of the cost itself                       -> reward 1.0
# upper is a PROVED bound, not a measured best; the derivation lives in anchors.json's
# _provenance.
ANCHORS_PATH = Path("/tests/anchors.json")
if not ANCHORS_PATH.exists():                 # ad hoc runs from the repo tree, outside the image
    ANCHORS_PATH = Path(__file__).resolve().parent / "anchors.json"
_ANCHORS = json.loads(ANCHORS_PATH.read_text())
BASELINE = float(_ANCHORS["baseline"])
SOTA = float(_ANCHORS["sota"])
UPPER = float(_ANCHORS["upper"])

# --- Paths and limits: ALL hard-coded on purpose ----------------------------------------------
# Nothing here reads the environment. A host-side `-e HELDOUT_DIR=/empty` (or TOOLS_DIR,
# CHILD_SCRIPT, SOLVER_UID, ...) would silently change what gets graded while still producing a
# perfectly normal-looking reward file. To grade a local copy, edit a copy of this file.
SUBMISSION_DIR = Path("/app/methods/main")
HELDOUT = Path("/tests/heldout")
TOOLS = Path("/tests/tools")
REWARD_DIR = Path("/logs/verifier")
CHILD = Path("/tests/child_solve.py")

CHILD_TIMEOUT = 3000.0
TOOL_TIMEOUT = 1800.0

# Unprivileged account the submission runs as (created in tests/Dockerfile). The submission and
# this grader share a filesystem, so without the drop the submitted code runs as root and can
# simply write /logs/verifier/reward.json (which Harbor reads in preference to reward.txt) and
# kill this process. Hard-coded for the same reason as the paths above: `-e SOLVER_UID=0` would
# turn the whole sandbox off.
SOLVER_UID = 65010
SOLVER_GID = 65010

# BLAS/OpenMP thread count, pinned to a CONSTANT so results do not depend on the host's core
# count. N = [verifier] cpus = 8; both Dockerfiles set the same ENV so the agent's self-check and
# grading are identical. The bundled C++ reference uses OpenMP, so an unpinned thread count makes
# the SOTA anchor a function of whichever machine measured it.
THREAD_ENV = {"OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8",
              "MKL_NUM_THREADS": "8", "NUMEXPR_NUM_THREADS": "8"}
os.environ.update(THREAD_ENV)          # hard assignment, NOT setdefault: the image ENV is already
                                       # set, and setdefault would silently never fire.

GRADER_FAILED = "grader failed: "      # marks OUR breakage, never the submission's — see main()

# The bundled checkers/evaluator exit 0 even when they REJECT a solution, and the evaluator
# then prints "Final score:0". Since lower-is-better, a trusted 0 would make an illegal
# solution the best possible.
#
# A blacklist of rejection markers is NOT sufficient and cannot be proven complete (two of the
# three tools ship without source). It has already failed once in production: a submission whose
# CLK pin mapping was incomplete made the evaluator print
#     "Map fail reason:ERROR[Missing mapping] C102675/CLK , Final score:0"
# which matches none of the markers below ("Map fail reason" != "Fail, reason"), so the 0 was
# trusted as a legal score and the run scored higher than every legitimate submission.
# `sanity` is the same shape: it rejects with "Checking fail, get 0!" and still exits 0.
#
# So rejection detection is a WHITELIST — each tool must positively confirm it passed — with the
# marker list kept only as a second line of defence.
FAIL_MARKERS = ("Fail !!!", "Fail, reason", "Map fail reason", "Checking fail",
                "is overlap with", "ERROR[")
PASS_MARKERS = {"sanity": "Checking pass!",
                "placement_checker": "Pass placement checker!!!",
                "preliminary-evaluator": "Check pass! Scoring..."}


def _has_failure(text: str) -> bool:
    return any(m in text for m in FAIL_MARKERS)


def _run(cmd, timeout) -> tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=timeout)
    return p.returncode, p.stdout


def _parse_score(text: str) -> float:
    for pat in (r"Final score:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
                r"Score:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"):
        m = re.search(pat, text)
        if m:
            return float(m.group(1))
    raise ValueError("could not parse final score from preliminary-evaluator output")


def _list_cases(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.iterdir() if p.is_file())


def _drop_privileges() -> None:
    """preexec_fn: become SOLVER_UID with no supplementary groups. Runs in the forked child."""
    os.setgroups([])
    os.setgid(SOLVER_GID)
    os.setuid(SOLVER_UID)


def _assert_submission_readable() -> None:
    """Fail LOUDLY if the artifacts dir is unreadable once we drop privileges.

    Checking st_mode is not enough (parent-directory x bits, ACLs, mount options all bite), and
    os.access() answers for the REAL uid, so this forks a child that actually setuids and opens
    the file. Without this the failure mode is silent: the child reads nothing, produces no
    output, and the run looks exactly like a submission that scored 0.
    """
    pid = os.fork()
    if pid == 0:                                     # child: try it for real, exit(0) on success
        try:
            _drop_privileges()
            with (SUBMISSION_DIR / "solver.py").open("rb") as fh:
                fh.read(1)
            os._exit(0)
        except BaseException:                        # noqa: BLE001
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    if os.waitstatus_to_exitcode(status) != 0:
        raise PermissionError(
            f"{SUBMISSION_DIR}/solver.py is not readable by uid {SOLVER_UID}; the submission "
            f"would silently produce no output and be graded 0")


def _run_child(out_dir: Path) -> None:
    """PHASE 1: run the untrusted solve() unprivileged, in its own process group, then kill it."""
    child_env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp",
                 "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", **THREAD_ENV}
    proc = subprocess.Popen(
        [sys.executable, str(CHILD), str(SUBMISSION_DIR), str(HELDOUT), str(out_dir)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,  # own session/process group, so we can kill stragglers
        preexec_fn=_drop_privileges,  # noqa: PLW1509 — the whole point
        env=child_env,
        cwd="/tmp",
    )
    try:
        stdout, _ = proc.communicate(timeout=CHILD_TIMEOUT)
        if stdout:
            print(stdout, file=sys.stderr)
    except subprocess.TimeoutExpired:
        print(f"child timed out after {CHILD_TIMEOUT}s", file=sys.stderr)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _score_case(case: Path, out: Path) -> float:
    """PHASE 2 (trusted): legality + score for one case. Raises on illegal/missing/unparseable."""
    if not out.exists() or out.stat().st_size == 0:
        raise ValueError("solver produced no non-empty output")
    for tool_name in ("sanity", "placement_checker", "preliminary-evaluator"):
        rc, so = _run([str(TOOLS / tool_name), str(case), str(out)], TOOL_TIMEOUT)
        if rc != 0 or _has_failure(so):
            raise ValueError(f"{tool_name} rejected the placement: {so[-300:].strip()}")
        if PASS_MARKERS[tool_name] not in so:      # whitelist: no explicit pass -> rejected
            raise ValueError(f"{tool_name} did not confirm a pass: {so[-300:].strip()}")
    score = _parse_score(so)
    if not (score > 0.0):
        # Every legal placement has strictly positive power and area, so a non-positive score
        # can only come from a rejection path that printed "Final score:0".
        raise ValueError(f"implausible non-positive score {score}")
    return score


def _reward(metric: float) -> float:
    """Piecewise-linear on the cost, lower-is-better: BASELINE -> 0, SOTA -> 0.6, UPPER -> 1.0.

    Both segments are linear because the whole band spans well under one decade of cost
    (87.2M -> 29.9M, a factor of 2.9), so there is no cross-magnitude compression for a log
    scale to fix; linear is the flattest in-segment shape here.

    The previous band mapped SOTA -> 0.5 and then soft-capped with `1 - 0.5/x`. That form was
    unreachable: `reward -> 1` requires `x -> inf`, i.e. a cost of MINUS infinity, while cost is
    a sum of non-negative terms. Its supremum over the real domain was 0.694, so the whole
    0.694..1.0 range was dead and the only way to touch it was an illegal solution scored as 0.
    """
    if not (metric > 0.0) or metric >= BASELINE:
        return 0.0
    if metric >= SOTA:
        r = 0.6 * (BASELINE - metric) / (BASELINE - SOTA)
    else:
        r = 0.6 + 0.4 * (SOTA - metric) / (SOTA - UPPER)
    return max(0.0, min(1.0, r))


def main() -> None:
    REWARD_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(REWARD_DIR, 0o700)          # root-only: the unprivileged child must not seed a reward
    out = {"metric": None, "reward": 0.0, "correctness": False, "errors": []}
    per_case: dict[str, float | None] = {}
    try:
        cases = _list_cases(HELDOUT)
        if not cases:
            raise FileNotFoundError("no held-out cases found")
        _assert_submission_readable()

        with tempfile.TemporaryDirectory(prefix="mbff_out_") as tmp:
            out_dir = Path(tmp)
            os.chown(out_dir, SOLVER_UID, SOLVER_GID)   # the dropped child writes its output here
            os.chmod(out_dir, 0o700)                    # ... and only it and root can look
            _run_child(out_dir)  # untrusted code runs ONLY here

            scores, errors = [], []
            for case in cases:
                try:
                    score = _score_case(case, out_dir / f"{case.name}.out")
                    scores.append(score)
                    per_case[case.name] = score
                except Exception as exc:  # noqa: BLE001 — the SUBMISSION failed, not the grader
                    errors.append(f"{case.name}: {exc}")
                    per_case[case.name] = None

        # Anti-gaming: illegal on ANY held-out case -> not a valid solution (inf, not a good
        # mean from only its legal cases).
        correctness = not errors and len(scores) == len(cases)
        metric = sum(scores) / len(scores) if correctness else math.inf
        reward = _reward(metric) if math.isfinite(metric) else 0.0
        out = {"metric": (round(metric, 6) if math.isfinite(metric) else None),
               "reward": round(reward, 6), "correctness": correctness, "errors": errors}
    except Exception as exc:  # noqa: BLE001 — any failure -> reward 0, never crash
        # Reached only when the GRADER itself broke (missing held-out, unreadable artifacts,
        # bad image, ...). fail-closed guarantees we never hand out a score that was not earned,
        # but it does NOT guarantee anyone notices: without a marker this file is byte-shaped
        # exactly like "the agent submitted something that scores 0", and a broken grading image
        # silently turns a whole batch of runs into "every model failed". The prefix is what the
        # aggregation layer keys on to call the run an infra failure and force a re-run.
        out = {"metric": None, "reward": 0.0, "correctness": False,
               "errors": [f"{GRADER_FAILED}{type(exc).__name__}: {exc}"]}

    # Harbor reward.json must be a flat {key: number} map. Keep ONLY numeric keys; non-numeric
    # diagnostics go to stdout + score_details.json.
    rewards = {"reward": float(out["reward"]),
               "mean_final_score": float(out["metric"]) if out["metric"] is not None else 0.0}
    details = {**out, "per_case": per_case, "anchors": {"baseline": BASELINE, "sota": SOTA,
                                                        "upper": UPPER}}
    (REWARD_DIR / "reward.txt").write_text(f"{out['reward']}\n", encoding="utf-8")
    (REWARD_DIR / "reward.json").write_text(json.dumps(rewards), encoding="utf-8")
    (REWARD_DIR / "score_details.json").write_text(json.dumps(details), encoding="utf-8")
    print(json.dumps(details))


if __name__ == "__main__":
    main()
