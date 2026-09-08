"""Exact visible-set self-check for the editable contraction planner."""
from __future__ import annotations

import json
from pathlib import Path

from evaluate_visible import evaluate


ROOT = Path(__file__).resolve().parent


def main() -> None:
    report = evaluate(
        ROOT / "methods" / "main",
        ROOT / "data" / "visible_cases.json",
    )
    print(json.dumps(report, indent=2))
    if not report["correct"]:
        raise SystemExit(1)

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
