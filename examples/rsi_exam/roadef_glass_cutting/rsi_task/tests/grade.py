#!/usr/bin/env python3
"""Sealed grader for ROADEF/EURO 2018 glass cutting (glass_cutting_v1).

Protocol
--------
The submission is a language-agnostic program living in SUBMISSION_DIR
(default /app/solution):

  * optional  solution/build.sh  -- run ONCE before grading (compile step),
  * required  solution/run.sh     -- run per case as
        bash run.sh <batch.csv> <defects.csv> <global_param.csv> <out.csv>
    reading the three instance CSVs and writing a cutting-tree solution
    to <out.csv>.

For every sealed instance in HELDOUT_DIR/in the grader runs the submission,
then validates + measures the produced solution with the OFFICIAL ROADEF
checker (HELDOUT_DIR/checker, the same binary contestants' solutions were
scored with). The absolute objective is the WASTE area

    waste = ((nPlates-1)*W + (W - widthResidual)) * H - item_area

(W,H = plate size; the reusable residual on the last plate is NOT waste). This
matches the challenge objective and the reproduced strong reference solver's own reported
waste exactly. A solution the checker rejects (validSolution != 1) scores 0 on
that case.

Per case:  rel(c) = ref_waste(c) / your_waste(c)   (lower waste = better =
higher rel; ref_waste = frozen reproduced-the strong reference objective, an honest field-
minimum PROXY). The aggregate metric is the mean of rel over the sealed cases
(HIGHER better), mapped to reward in [0,1] by a piecewise-log curve
frozen in tests/anchors.json (never present in the agent image):

    baseline metric -> 0.0     (shipped greedy starter)
    reference metric -> 0.6     (reproduced strong reference, 30s budget)
    upper     metric -> 1.0     (metric 1.5 ceiling)

reward.json is written flat (float leaves only) for harbor's VerifierResult.
"""
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBMISSION_DIR = Path(os.environ.get("SUBMISSION_DIR", "/app/methods/main"))
HELDOUT_DIR = Path(os.environ.get("HELDOUT_DIR", "/heldout"))
LOG_DIR = Path(os.environ.get("VERIFIER_LOG_DIR", "/logs/verifier"))

CHECKER = HELDOUT_DIR / "checker"
IN_DIR = HELDOUT_DIR / "in"
GLOBAL_PARAM = HELDOUT_DIR / "global_param.csv"
REF_WASTE = json.loads((HELDOUT_DIR / "ref_lengths.json").read_text())

PER_CASE_TIMEOUT = float(os.environ.get("PER_CASE_TIMEOUT_SEC", "40"))
BUILD_TIMEOUT = float(os.environ.get("BUILD_TIMEOUT_SEC", "900"))

# Sealed scoring band; unlinked at import so the submission can never read it.
_ANCHORS_PATH = Path(__file__).resolve().parent / "anchors.json"
_ANCHORS = json.loads(_ANCHORS_PATH.read_text())
try:
    _ANCHORS_PATH.unlink()
except OSError:
    pass
A_BASELINE = float(_ANCHORS["baseline"])
A_REFERENCE = float(_ANCHORS["reference"])
A_UPPER = float(_ANCHORS["upper"])

STAT_RE = re.compile(r"[^;]+;(\d+);(\d+);[^;]*;([\d.]+)")


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def piecewise_log(metric):
    """baseline->0, reference->0.6, upper->1; linear in log(metric) per segment."""
    if metric <= 0 or A_BASELINE <= 0 or metric <= A_BASELINE:
        return 0.0
    knots = [(A_BASELINE, 0.0), (A_REFERENCE, 0.60), (A_UPPER, 1.00)]
    for (v0, r0), (v1, r1) in zip(knots, knots[1:]):
        if metric <= v1:
            return _clip(r0 + (r1 - r0) * math.log(metric / v0) / math.log(v1 / v0), r0, r1)
    return 1.0


def plate_dims():
    W, H = 6000, 3210
    with open(GLOBAL_PARAM) as f:
        r = csv.reader(f, delimiter=';')
        next(r, None)
        for row in r:
            if not row:
                continue
            if row[0] == "widthPlates":
                W = int(row[1])
            elif row[0] == "heightPlates":
                H = int(row[1])
    return W, H


def item_area(batch):
    tot = 0
    with open(batch) as f:
        r = csv.reader(f, delimiter=';')
        next(r, None)
        for row in r:
            if row and row[0].strip():
                tot += int(row[1]) * int(row[2])
    return tot


def maybe_build():
    build = SUBMISSION_DIR / "build.sh"
    if build.exists():
        try:
            subprocess.run(["bash", str(build)], cwd=str(SUBMISSION_DIR),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=BUILD_TIMEOUT)
        except subprocess.SubprocessError:
            pass


def check_waste(idx, batch, defects, sol_path, W, H):
    """Validate + measure with the official checker. Return waste or None."""
    work = tempfile.mkdtemp()
    inst = Path(work) / "instances_checker"
    logs = Path(work) / "logs"
    inst.mkdir(); logs.mkdir()
    (inst / f"{idx}_batch.csv").write_bytes(Path(batch).read_bytes())
    (inst / f"{idx}_defects.csv").write_bytes(Path(defects).read_bytes())
    (inst / "global_param.csv").write_bytes(GLOBAL_PARAM.read_bytes())
    (inst / f"{idx}_solution.csv").write_bytes(Path(sol_path).read_bytes())
    try:
        subprocess.run([str(CHECKER), idx], cwd=work, input="0\n",
                       capture_output=True, text=True, timeout=120)
    except subprocess.SubprocessError:
        return None
    stat = logs / f"{idx}_statistics.csv"
    if not stat.exists():
        return None
    lines = stat.read_text().strip().splitlines()
    if len(lines) < 2:
        return None
    parts = lines[1].split(";")
    try:
        valid = int(parts[1]); nplates = int(parts[2]); resid = int(float(parts[4]))
    except (ValueError, IndexError):
        return None
    if valid != 1:
        return None
    used_w = (nplates - 1) * W + (W - resid)
    return used_w * H - item_area(batch)


def run_case(idx, batch, defects, W, H):
    run = SUBMISSION_DIR / "run.sh"
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
        out_path = tf.name
    try:
        try:
            subprocess.run(["bash", str(run), batch, defects, str(GLOBAL_PARAM), out_path],
                           cwd=str(SUBMISSION_DIR), stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=PER_CASE_TIMEOUT)
        except subprocess.SubprocessError:
            return None
        if os.path.getsize(out_path) == 0:
            return None
        return check_waste(idx, batch, defects, out_path, W, H)
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    maybe_build()
    W, H = plate_dims()

    batches = sorted(IN_DIR.glob("*_batch.csv"))
    rels, per_case = [], {}
    for b in batches:
        idx = b.name[:-len("_batch.csv")]
        defects = IN_DIR / f"{idx}_defects.csv"
        r1 = REF_WASTE.get(idx)
        waste = run_case(idx, str(b), str(defects), W, H)
        if waste is None or waste <= 0 or r1 is None:
            rel = 0.0
        else:
            rel = r1 / waste
        rels.append(rel)
        per_case[idx] = {"ref_waste": r1, "your_waste": waste, "rel": round(rel, 4)}

    metric = sum(rels) / len(rels) if rels else 0.0
    reward = round(piecewise_log(metric), 6)
    flat = {"reward": reward, "metric": round(metric, 4),
            "n_cases": len(batches),
            "n_valid": sum(1 for v in per_case.values() if v["your_waste"])}
    (LOG_DIR / "reward.json").write_text(json.dumps(flat))
    (LOG_DIR / "score_details.json").write_text(json.dumps({
        "per_case": per_case,
        "anchors": {"baseline": A_BASELINE,
                    "reference": A_REFERENCE, "super": A_UPPER},
        "aggregate": flat}))
    print(f"metric={metric:.4f} reward={reward:.6f} "
          f"valid={flat['n_valid']}/{flat['n_cases']}")


if __name__ == "__main__":
    main()
