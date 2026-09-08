import importlib.util
import numpy as np
from problem import simulate


def load_solver():
    spec = importlib.util.spec_from_file_location("solver", "/app/methods/main/solver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    solver = load_solver()
    cases = np.load("/app/data/visible_cases.npz", allow_pickle=True)["cases"]
    rmses = []
    for i, case in enumerate(cases):
        field = np.clip(np.asarray(solver.estimate_logf(case), float), -3.0, 1.0)
        pred = simulate(field, case["validation_gauges"], case["depth"], case["forcing"])
        rmse = float(np.sqrt(np.mean(np.abs(pred - case["validation_heads"]) ** 2)))
        rmses.append(rmse)
        print(f"case {i}: validation_gauge_rmse={rmse:.6f}")
    print(f"mean_validation_gauge_rmse={np.mean(rmses):.6f}")

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
