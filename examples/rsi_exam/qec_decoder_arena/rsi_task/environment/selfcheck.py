"""Self-check: score your decoder on the VISIBLE color-code settings.

    python3 selfcheck.py [path/to/solver.py]

The visible settings carry the true eval observable flips (eval_obs), so this
runs the EXACT scoring the sealed grader uses — per setting: LER = mean
any-observable-mismatch, score = clip((ln ler_base - ln ler_agent)/
(ln ler_base - ln ler_sota), 0, CAP), LERs floored at 1/(2*shots). The graded
set is a SEALED sibling of the same four color settings (calibrated to the same
anchors) that you never see — improve the DECODER, not these four answers.

metric = mean setting score: 0 = plain pymatching on the published DEM,
1.0 = the reference level recorded with each setting."""
import importlib.util
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arena_harness as ah

VISIBLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings_visible")
SETTINGS = ["color_d5", "color_d5_Z", "color_d7", "color_d9"]


def load(path):
    spec = importlib.util.spec_from_file_location("solver", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "methods", "main", "solver.py")
    mod = load(path)
    scores = []
    print(f"{'setting':<12}{'ler_agent':>11}{'ler_base':>10}{'ler_sota':>10}"
          f"{'score':>8}   note")
    for name in SETTINGS:
        setting = ah.load_setting(VISIBLE, name)
        meta = setting["meta"]
        truth = setting["eval_obs"]
        t0 = time.time()
        try:
            pred = np.asarray(mod.decode(setting, setting["eval_dets"]), dtype=bool)
            ler = ah.ler_of(pred, truth)
            s = ah.setting_score(ler, meta, meta["n_eval"])
            scores.append(s)
            print(f"{name:<12}{ler:>11.5f}{meta['ler_base']:>10.5f}"
                  f"{meta['ler_sota']:>10.5f}{s:>8.3f}   ({time.time()-t0:.0f}s)")
        except Exception as e:
            scores.append(0.0)
            print(f"{name:<12}{'-':>11}{meta['ler_base']:>10.5f}"
                  f"{meta['ler_sota']:>10.5f}{0.0:>8.3f}   FAILED: "
                  f"{type(e).__name__}: {e}")
    print(f"\nvisible metric (mean setting score) = {np.mean(scores):.4f}")
    print("This is a visible development metric; the sealed evaluation uses held-out settings.")

    import subprocess as _sp
    if os.path.exists("/app/budget.py"):
        sys.stdout.flush()
        _sp.run([sys.executable, "/app/budget.py"], check=False)
