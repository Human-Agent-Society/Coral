"""CHILD side of the two-phase verifier (UNTRUSTED — imports the submitted solver and runs its Policy).

Spawned by grade.py as a fresh process (NOT a fork), in its own session/group, and DROPPED to the
unprivileged uid/gid 4242 `solver`: it can write nothing but the scratch dir handed to it (cwd /
TMPDIR / HOME / OUTPUT_JSON all live there), cannot touch /logs/verifier or /tests, cannot signal the
root parent, and cannot read /proc/1/environ or any sealed 0400 file. It loads the submitted Policy
from SUBMISSION_DIR + the checkpoint the agent already trained under MODEL_DIR (artifact-eval — this
process never calls train(), never re-trains, never touches the network), runs each sealed held-out
episode closed-loop, and writes ONLY the per-episode ACTION TRACE (plain data) to OUTPUT_JSON. It
never computes the reward and never sees the reward anchors, so nothing it does to its own process can
affect the return the trusted parent recomputes by clean replay of these traces.

argv:  child_solve.py <submission_dir> <model_dir> <output_json> <tests_dir>
stdin: {"seeds": [...]} — the held-out episode seeds, handed over by the root parent, which read the
       sealed heldout_seeds.py itself. (The seeds necessarily become visible to this process at run
       time — it has to drive those episodes — but there is no longer a sealed FILE it can read.)
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


def main() -> int:
    submission_dir = Path(sys.argv[1])
    model_dir = sys.argv[2]
    output_json = Path(sys.argv[3])
    tests_dir = Path(sys.argv[4])

    seeds = [int(s) for s in json.loads(sys.stdin.read())["seeds"]]

    sys.path.insert(0, str(tests_dir))
    import trifinger_score as ts

    os.environ["MODEL_DIR"] = model_dir  # Policy.__init__ reads the checkpoint path from here

    solver_path = submission_dir / "solver.py"
    if not solver_path.exists():
        print(f"missing solver.py under {submission_dir}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(submission_dir))
    spec = importlib.util.spec_from_file_location("submitted_solver", str(solver_path))
    if spec is None or spec.loader is None:
        print(f"cannot import {solver_path}", file=sys.stderr)
        return 2
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)               # untrusted top-level code runs HERE
    if not hasattr(mod, "Policy"):
        print("solver.py must define class Policy(PolicyBase)", file=sys.stderr)
        return 2

    env = ts.make_env(data_dir=None)  # no dataset needed for rollout; sim assets ship in the wheel
    policy = mod.Policy(env.action_space, env.observation_space,
                         env.unwrapped.sim_env.episode_length)  # untrusted code runs HERE too

    # SOFT TRUNCATION. This used to be a single write AFTER the whole
    # 32-episode loop, so a wall-clock kill at episode 31 threw away 31 finished episodes: the
    # parent found no traces file at all and reported "policy produced no action traces" -> reward
    # 0.0, indistinguishable from a submission that never produced anything. That is the failure
    # mode the task forbids (not finishing is not a submission failure; it should be scored on
    # the best state reached so far) and the same shape that
    # discovery/symbolic_regression was ruled into soft truncation for.
    # The trace file is now re-published after EVERY episode, so whatever finished before the kill
    # is scored. replay_score.py already scores a missing/empty trace as return 0.0 for that seed,
    # so a truncated run degrades smoothly (k good episodes out of 32) instead of falling off a
    # cliff. A COMPLETE run writes exactly the same bytes as before -- same dict, same insertion
    # order -- so no anchor moves (re-measured).
    # The write is atomic (tmp + os.replace) because the parent may read this file at any instant,
    # including while it is being rewritten; a half-written file would be a corrupt-JSON failure
    # that looks like a bad submission.
    tmp_json = output_json.with_name(output_json.name + ".part")

    def _publish(obj) -> None:
        tmp_json.write_text(json.dumps(obj))
        os.replace(tmp_json, output_json)

    traces = {}
    _publish(traces)                       # exists from t=0: "killed early" != "never started"
    for seed_k in seeds:
        try:
            r = ts.run_policy_episode(env, seed_k, policy)
            traces[str(seed_k)] = r["actions"]
        except Exception as exc:  # noqa: BLE001 — a crash on one episode -> empty trace -> 0 there
            print(f"seed {seed_k}: policy error: {type(exc).__name__}: {exc}", file=sys.stderr)
            traces[str(seed_k)] = []
        _publish(traces)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
