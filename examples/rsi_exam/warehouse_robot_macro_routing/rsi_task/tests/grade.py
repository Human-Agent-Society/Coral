#!/usr/bin/env python3
"""Sealed grader for AHC066 Macro Controller (macro_controller_v1).

Protocol
--------
The submission is a language-agnostic stdin->stdout program living in
SUBMISSION_DIR (default /app/solution):

  * optional  solution/build.sh  -- run ONCE before grading (compile step),
  * required  solution/run.sh     -- run per test case; reads one instance on
    stdin, writes the F/R/L/S/M/P button sequence on stdout.

For every sealed instance in HELDOUT_DIR/in the grader pipes the instance into
the submission, scores the produced button sequence with the OFFICIAL AtCoder
visualiser (HELDOUT_DIR/vis, the same binary contestants used), and computes a
per-case relative score

    rel(c) = rank1_len(c) / your_absolute_score(c)

where rank1_len(c) is the frozen per-case absolute score of the reproduced
contest rank-1 solution (an honest approximation of the field MIN, not the true
all-participants minimum) and your_absolute_score(c) is what the official
visualiser reports for the submission on case c (button-sequence length when all
balls are basketed, or the T*(M-V) penalty otherwise -- lower is better, so an
invalid/incomplete/timed-out case drives rel(c) toward 0).

The aggregate metric is the mean of rel(c) over all sealed cases (HIGHER is
better). It is mapped to reward in [0,1] by a 3-anchor curve frozen in the
verifier environment:

    baseline metric -> 0.0     (shipped greedy starter, no macros)
    rank1    metric -> 0.6     (reproduced contest rank-1)
    super    metric -> 1.0     (metric 1.5 upper bound)

reward.json is written flat (float leaves only) for harbor's VerifierResult.
"""
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBMISSION_DIR = Path(os.environ.get("SUBMISSION_DIR", "/app/methods/main"))
HELDOUT_DIR = Path(os.environ.get("HELDOUT_DIR", "/heldout"))
LOG_DIR = Path(os.environ.get("VERIFIER_LOG_DIR", "/logs/verifier"))

VIS = HELDOUT_DIR / "vis"
IN_DIR = HELDOUT_DIR / "in"
RANK1_LEN = json.loads((HELDOUT_DIR / "rank1_lengths.json").read_text())

PER_CASE_TIMEOUT = float(os.environ.get("PER_CASE_TIMEOUT_SEC", "20"))
BUILD_TIMEOUT = float(os.environ.get("BUILD_TIMEOUT_SEC", "900"))

# 3 frozen anchors (metric values), read from verifier env; defaults = sealed
# measurement so the grader is runnable standalone.
# Anchors live in a sealed root-owned file, never in [verifier.env]:
# anything injected there is readable from /proc/1/environ by submitted code.
ANCHOR_PATH = Path(__file__).resolve().with_name("anchors.json")
with ANCHOR_PATH.open(encoding="utf-8") as _f:
    _ANCHORS = json.load(_f)
A_BASELINE = float(_ANCHORS["BASELINE"])
A_RANK1 = float(_ANCHORS["RANK1"])
A_SUPER = float(_ANCHORS["SUPER"])

SCORE_RE = re.compile(r"Score\s*=\s*(-?\d+)")


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def piecewise_log(metric):
    """baseline->0, rank1->0.6, super->1."""
    b, r1, sup = A_BASELINE, A_RANK1, A_SUPER
    if metric <= 0 or b <= 0:
        return 0.0
    if metric <= b:
        return 0.0
    if metric <= r1:                        # baseline .. rank1  -> 0.0 .. 0.6
        denom = math.log(r1 / b)
        return _clip(0.60 * math.log(metric / b) / denom, 0.0, 0.60)
    if metric <= sup:                       # rank1 .. super     -> 0.6 .. 1.0, linear
        return _clip(0.60 + 0.40 * (metric - r1) / (sup - r1), 0.60, 1.00)
    return 1.0


def maybe_build():
    build = SUBMISSION_DIR / "build.sh"
    if build.exists():
        try:
            subprocess.run(["bash", str(build)], cwd=str(SUBMISSION_DIR),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=BUILD_TIMEOUT)
        except subprocess.SubprocessError:
            pass


def run_case(in_file):
    """Run the submission on one instance, return official absolute score or None."""
    run = SUBMISSION_DIR / "run.sh"
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as tf:
        out_path = tf.name
    try:
        with open(in_file) as fi, open(out_path, "w") as fo:
            try:
                subprocess.run(["bash", str(run)], cwd=str(SUBMISSION_DIR),
                               stdin=fi, stdout=fo, stderr=subprocess.DEVNULL,
                               timeout=PER_CASE_TIMEOUT)
            except subprocess.SubprocessError:
                return None
        r = subprocess.run([str(VIS), str(in_file), out_path],
                           capture_output=True, text=True, timeout=60)
        m = SCORE_RE.search(r.stdout + r.stderr)
        return int(m.group(1)) if m else None
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ANCHOR_PATH.unlink(missing_ok=True)
    maybe_build()

    files = sorted(IN_DIR.glob("*.txt"))
    rels, per_case = [], {}
    for f in files:
        name = f.name
        r1 = RANK1_LEN.get(name)
        score = run_case(f)
        if score is None or score <= 0 or r1 is None:
            rel = 0.0
        else:
            rel = r1 / score
        rels.append(rel)
        per_case[name] = {"rank1_len": r1, "your_score": score, "rel": round(rel, 4)}

    metric = sum(rels) / len(rels) if rels else 0.0
    reward = round(piecewise_log(metric), 6)

    flat = {"reward": reward}
    (LOG_DIR / "reward.json").write_text(json.dumps(flat))
    (LOG_DIR / "score_details.json").write_text(json.dumps({
        "per_case": per_case,
        "anchors": {"baseline": A_BASELINE,
                    "rank1": A_RANK1, "super": A_SUPER},
        "aggregate": flat}))
    print(f"metric={metric:.4f} reward={reward:.6f} "
          f"valid={flat['n_valid']}/{flat['n_cases']}")


if __name__ == "__main__":
    main()
