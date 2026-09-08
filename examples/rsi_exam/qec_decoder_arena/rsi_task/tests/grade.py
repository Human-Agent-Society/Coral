"""Trusted grading parent (sealed) — QEC color-code decoder arena.

The agent submits a DECODER `decode(setting, dets) -> predictions` ([n_shots,
n_observables] bool). For each sealed color-code setting the parent:
  1. loads the setting via arena_harness (which exposes eval_dets but NOT the
     truth), and separately np.loads eval_truth.npz (the hidden observable flips)
     — truth is read ONLY here in the parent;
  2. builds a sanitized per-setting root that carries eval_dets but no
     eval_truth.npz, and runs the agent's decode() in an ISOLATED SUBPROCESS
     with a per-setting wall-clock timeout = meta["decode_budget_sec"];
  3. computes ler_agent = arena_harness.ler_of(pred, truth), then the setting
     score = arena_harness.setting_score(ler_agent, meta, meta["n_eval"]) —
     s = clip((ln ler_base - ln ler_agent)/(ln ler_base - ln ler_sota), 0, CAP),
     LERs floored at 1/(2*shots), anchors baked into meta at calibration time.

metric = mean of setting scores (0 = plain pymatching on the published DEM,
1.0 = the published open-decoder portfolio). A crash / timeout / invalid-shape
on a setting scores 0 for THAT setting, never a whole-run failure. reward is a
piecewise-linear map through the sealed tests/anchors.json BASELINE->0, SOTA->0.6,
UPPER_BOUND->1.0, capped at 1.0."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arena_harness as ah

HERE = os.path.dirname(os.path.abspath(__file__))
# Anchors live in a sealed root-owned file, never in [verifier.env]: anything
# injected there is readable from /proc/1/environ by the submitted code.
with open(os.path.join(HERE, "anchors.json")) as _f:
    _A = json.load(_f)
BASELINE = float(_A["BASELINE"])
SOTA = float(_A["SOTA"])
UPPER_BOUND = float(_A["UPPER_BOUND"])
SOLVER = os.environ.get("SOLVER_PATH", "/app/methods/main/solver.py")
OUT = os.environ.get("REWARD_PATH", "/logs/verifier/reward.json")
SEALED = os.environ.get("SEALED_ROOT", os.path.join(HERE, "settings_sealed"))
DEFAULT_BUDGET = 600.0

# The four sealed color-code settings (order fixed for reproducibility).
SETTINGS = ["color_d5", "color_d5_Z", "color_d7", "color_d9"]


def reward_of(metric):
    pts = [(BASELINE, 0.0), (SOTA, 0.6), (UPPER_BOUND, 1.0)]
    if metric <= pts[0][0]:
        return 0.0
    for (v0, r0), (v1, r1) in zip(pts, pts[1:]):
        if metric <= v1:
            return float(r0 + (r1 - r0) * (metric - v0) / max(1e-12, v1 - v0))
    return 1.0  # metric cannot exceed UPPER_BOUND: every setting score is capped at SCORE_CAP


def _sanitized_root(name, workdir):
    """Copy the sealed setting into a fresh root carrying eval_dets but NEVER
    eval_truth.npz, so the untrusted child cannot see the ground truth."""
    src = os.path.join(SEALED, name)
    dst = os.path.join(workdir, name)
    os.makedirs(dst, exist_ok=True)
    for fn in ("meta.json", "model.dem", "model.stim", "dev.npz", "eval.npz"):
        shutil.copy(os.path.join(src, fn), os.path.join(dst, fn))
    # explicitly do NOT copy eval_truth.npz
    return workdir


def grade_one(name, workdir):
    meta = json.load(open(os.path.join(SEALED, name, "meta.json")))
    truth = np.load(os.path.join(SEALED, name, "eval_truth.npz"))["obs"]
    n_eval = int(meta.get("n_eval", len(truth)))
    budget = float(meta.get("decode_budget_sec", DEFAULT_BUDGET))

    child_root = _sanitized_root(name, workdir)
    opath = os.path.join(workdir, f"pred_{name}.npy")
    if os.path.exists(opath):
        os.remove(opath)
    try:
        subprocess.run([sys.executable, os.path.join(HERE, "run_decode.py"),
                        SOLVER, child_root, name, opath],
                       check=True, timeout=budget)
        pred = np.load(opath)
        ler_agent = ah.ler_of(pred, truth)   # asserts matching shape
        score = ah.setting_score(ler_agent, meta, n_eval)
        detail = {"name": name, "ler_agent": round(ler_agent, 6),
                  "ler_base": meta["ler_base"], "ler_sota": meta["ler_sota"],
                  "score": round(score, 4), "valid": True, "reason": None}
    except subprocess.TimeoutExpired:
        score = 0.0
        detail = {"name": name, "ler_agent": None, "ler_base": meta["ler_base"],
                  "ler_sota": meta["ler_sota"], "score": 0.0, "valid": False,
                  "reason": f"decode exceeded budget {budget:g}s"}
    except Exception as e:
        score = 0.0
        detail = {"name": name, "ler_agent": None, "ler_base": meta["ler_base"],
                  "ler_sota": meta["ler_sota"], "score": 0.0, "valid": False,
                  "reason": f"decode child failed: {type(e).__name__}: {e}"}
    return score, detail


def main():
    scores, details = [], []
    for name in SETTINGS:
        if not os.path.isdir(os.path.join(SEALED, name)):
            continue
        workdir = tempfile.mkdtemp(prefix=f"qec_{name}_")
        try:
            s, d = grade_one(name, workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        scores.append(s)
        details.append(d)
        print(f"{name:<12} ler_agent={d['ler_agent']} score={d['score']} "
              f"valid={d['valid']}" + (f"  ({d['reason']})" if d["reason"] else ""))

    metric = float(np.mean(scores)) if scores else 0.0
    n_valid = sum(1 for d in details if d["valid"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # harbor requires reward.json to be FLAT {str: float|int}
    payload = {"reward": round(reward_of(metric), 6)}
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(os.path.dirname(OUT), "score_details.json"), "w") as f:
        json.dump({"anchors": {"baseline": BASELINE, "sota": SOTA,
                               "upper_bound": UPPER_BOUND},
                   "score_cap": ah.SCORE_CAP,
                   "metric": round(metric, 4),
                   "n_settings": len(scores), "n_valid": n_valid,
                   "per_setting": details}, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
