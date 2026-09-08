"""Canonical complete visible-suite evaluator."""
from __future__ import annotations

import json
from pathlib import Path

from evaluator import SUBMISSION_DIR, evaluate, load_jsonl


def main():
    rows = load_jsonl(Path("/app/data/visible.jsonl"))
    if len(rows) != 54:
        raise RuntimeError(f"visible suite must contain exactly 54 rows, found {len(rows)}")
    result = evaluate(rows, SUBMISSION_DIR)
    Path("/app/selfcheck_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"visible compile accuracy = {result['n_correct']}/{result['n_total']} "
          f"= {result['accuracy_pct']:.2f}%")
    if result.get("invalid"):
        raise RuntimeError(result.get("error", "invalid submission"))

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
