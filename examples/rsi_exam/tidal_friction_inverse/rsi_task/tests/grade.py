import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import problem
from problem import simulate

ANCHORS = json.loads((Path(__file__).resolve().parent / "anchors.json").read_text())
BASELINE = float(ANCHORS["BASELINE"])
UPPER_BOUND = float(ANCHORS["UPPER_BOUND"])
REFERENCE = float(ANCHORS["REFERENCE"])
FRONTIER = {k: float(v) for k, v in ANCHORS["FRONTIER"].items()}  # per sealed case


def reward_of(metric, frontier):
    """baseline->0, reference->0.3, frontier->0.6, upper->1; linear in -log10(RMSE)."""
    import math
    anchors = [(BASELINE, 0.0), (REFERENCE, 0.3), (frontier, 0.6), (UPPER_BOUND, 1.0)]
    x = -math.log10(max(metric, 1e-12))
    pts = [(-math.log10(v), rw) for v, rw in anchors]  # ascending in x as metric drops
    if x <= pts[0][0]:
        return 0.0
    for (v0, r0), (v1, r1) in zip(pts, pts[1:]):
        if x <= v1:
            return float(r0 + (r1 - r0) * (x - v0) / max(1e-12, v1 - v0))
    return 1.0


def find_solver():
    candidates = [Path("/app/methods/main/solver.py"), Path("/logs/artifacts/methods/main/solver.py")]
    candidates += list(Path("/root").glob("**/methods/main/solver.py"))
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError("submitted methods/main/solver.py not found")


def run_one(solver_path, public_case):
    with tempfile.TemporaryDirectory() as td:
        os.chmod(td, 0o755)
        shutil.copy(problem.__file__, Path(td) / "problem.py")
        inp = Path(td) / "case.npz"
        out = Path(td) / "field.npy"
        np.savez_compressed(inp, case=np.array(public_case, dtype=object))
        runner = Path(td) / "run.py"
        runner.write_text(
            "import importlib.util,numpy as np,sys\n"
            "s=importlib.util.spec_from_file_location('submission',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
            "c=np.load(sys.argv[2],allow_pickle=True)['case'].item();z=np.asarray(m.estimate_logf(c),float)\n"
            "assert z.shape==(24,24) and np.isfinite(z).all();np.save(sys.argv[3],np.clip(z,-3.0,1.0))\n"
        )
        out.touch()
        os.chmod(str(out), 0o666)
        subprocess.run([sys.executable, str(runner), str(solver_path), str(inp), str(out)],
                       check=True, timeout=75, cwd=str(td), user="solver")
        return np.load(out)


def main():
    solver = find_solver()
    sealed = np.load("/heldout/cases.npz", allow_pickle=True)["cases"]
    cases = [dict(c) for c in sealed]              # pull answers into memory, then...
    os.remove("/heldout/cases.npz")                # ...seal them off disk before any solver runs
    details = []
    for i, secret in enumerate(cases):
        public = {k: secret[k] for k in ("depth", "forcing", "gauges", "obs_heads")}
        field = run_one(solver, public)
        pred = simulate(field, secret["score_gauges"], secret["depth"], secret["forcing"])
        rmse = float(np.sqrt(np.mean(np.abs(pred - secret["score_heads"]) ** 2)))
        cid = f"sealed_{i:02d}"
        details.append({"id": cid, "rmse": rmse, "floor": BASELINE,
                        "reference": REFERENCE, "frontier": FRONTIER[cid], "upper": UPPER_BOUND,
                        "score": reward_of(rmse, FRONTIER[cid])})
    reward = float(np.mean([d["score"] for d in details]))
    out = Path("/logs/verifier")
    out.mkdir(parents=True, exist_ok=True)
    (out / "score_details.json").write_text(json.dumps({"metric": "gauge_rmse", "instances": details}, indent=2))
    (out / "reward.json").write_text(json.dumps({"reward": reward, "mean_gauge_rmse": float(np.mean([d["rmse"] for d in details]))}))
    print(json.dumps({"reward": reward, "instances": details}, indent=2))


if __name__ == "__main__":
    main()
