"""Local self-check: score your current solution on the VISIBLE load settings and print
the raw Cost (lower is better). Visible = scale 0.3 (light) and scale 0.5 (the
over-saturation proxy). The hidden test uses heavier, unseen load/seed settings, so
tune for robustness, not for these two numbers.

    python /app/selfcheck.py            # both visible settings
    python /app/selfcheck.py --scale 0.5 --seed 43
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "sim"))
from eval_core import evaluate  # noqa: E402

SOLUTION = os.path.join(HERE, "methods", "main")
VISIBLE = [(0.25, 42), (0.37, 42)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cases = [(a.scale, a.seed)] if a.scale else VISIBLE
    for scale, seed in cases:
        out = os.path.join(HERE, f"_tripinfo_selfcheck_{scale}_{seed}.xml")
        res = evaluate(SOLUTION, scale, seed, out)
        if os.path.exists(out):
            os.remove(out)
        print(f"scale={scale} seed={seed}  Cost={res['cost']:,}  "
              f"(timeLoss={res['timeLoss']:,} teleports={res['teleports']} "
              f"stops={res['stops']:,} unfinished={res['unfinished']})  "
              f"[{res['wall_s']}s]")

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
