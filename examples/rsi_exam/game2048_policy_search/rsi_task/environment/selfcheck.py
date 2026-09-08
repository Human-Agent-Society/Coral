#!/usr/bin/env python3
"""Run the current 2048 policy on all public seeds."""

from pathlib import Path
import json
import resource
import signal

from evaluate import evaluate


ROOT = Path(__file__).resolve().parent
POLICY_MAX_BYTES = 10_000_000  # same literal the grader enforces; keep the two in step
CPU_SEC_PER_GAME = 225         # same per-game budget the grader enforces


def _budget_exceeded(_signum: int, _frame: object) -> None:
    raise SystemExit(
        f"policy used more than {CPU_SEC_PER_GAME} s of CPU per game. The grader applies the "
        "same budget to the whole sealed suite with one shared process, so a policy this slow "
        "is killed partway through and every remaining game scores zero."
    )


def _arm_cpu_budget(games: int) -> None:
    budget = CPU_SEC_PER_GAME * games
    signal.signal(signal.SIGXCPU, _budget_exceeded)
    resource.setrlimit(resource.RLIMIT_CPU, (budget, budget + 20))


def main() -> None:
    policy = ROOT / "methods/main/policy.py"
    size = policy.stat().st_size
    if size > POLICY_MAX_BYTES:
        raise SystemExit(
            f"policy.py is {size:,} bytes, over the {POLICY_MAX_BYTES:,} byte limit. "
            "The grader rejects this outright and scores zero."
        )

    suite = ROOT / "visible_seeds.json"
    _arm_cpu_budget(len(json.loads(suite.read_text(encoding="utf-8"))["seeds"]))

    before = resource.getrusage(resource.RUSAGE_SELF)
    result = evaluate(policy, suite)
    after = resource.getrusage(resource.RUSAGE_SELF)
    games = len(result["instances"])
    used = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    result["cpu_seconds"] = round(used, 1)
    result["cpu_seconds_per_game"] = round(used / games, 1)
    result["cpu_budget_per_game"] = CPU_SEC_PER_GAME

    (ROOT / "visible_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))

    # Budget reminder. /app/budget.py is mounted by the harness; when it is not mounted the
    # whole block is skipped, so running this task standalone is unaffected. check=False so a
    # failing budget.py can never take selfcheck down with it. Placed at the end of main() so
    # the reminder prints after this run's score.
    import os as _os, subprocess as _sp, sys as _sys
    if _os.path.exists("/app/budget.py"):
        _sys.stdout.flush()          # without this the reminder prints before the score in a pipe
        _sp.run([_sys.executable, "/app/budget.py"], check=False)


if __name__ == "__main__":
    main()
