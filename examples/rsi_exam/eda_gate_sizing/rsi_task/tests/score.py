"""Per-case SCORING for the held-out grader — the metric lives HERE (swappable).

`score_case(case_dir, size_path) -> float` returns the per-case score, and
`score_case_detailed(...) -> (float, dict)` additionally returns how that score
splits across the four terms. The grader records the split in
`verifier/score_details.json`: without it a sealed score is a single opaque
number and questions like "did the method that cleared cap violations on the
visible designs also clear them here?" cannot be answered without a re-run.
It is a verifier-side artifact and never reaches the agent.

`score_case(case_dir, size_path) -> float` returns the per-case score,
**lower-is-better**, matching the contest objective:

    score = leakage_power_uW
          + tns_penalty  * |TNS|
          + slew_penalty * (#slew violations)
          + cap_penalty  * (#cap  violations)

Computing real power + post-sizing timing requires a timing engine, so this
shells out to OpenROAD's Python API running `eval/evaluate_case.py` (adapted
from the public benchmark's evaluation script). The verifier image provides an
`openroad` binary with Python bindings, built from the contest OpenROAD fork
(see tests/Dockerfile).

=== WHY OPENROAD (metric rationale) ===
An OpenROAD-free leakage-only proxy was evaluated and REJECTED: within-func
leakage spread in the IR tables yields ~0.00% headroom (a min-leakage greedy
barely moves total leakage). The measured do-nothing score is instead dominated
by timing/DRV violations (10*|TNS| + 20*slew + 20*cap ~= 4.3e5 on held-out),
which only the timing engine exposes — so the score is real and OpenROAD is
required. Anchors live in tests/anchors.json (per case, sealed). If you change the metric,
this is the only file to touch — but every anchor must then be re-measured.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

# hard-coded. `EVAL_SCRIPT` in particular must not be settable from the
# host — pointing it elsewhere swaps out the entire scorer.
OPENROAD = "openroad"
EVAL_SCRIPT = Path("/tests/eval/evaluate_case.py")
SCORE_TIMEOUT = 3600.0


def _openroad_available() -> bool:
    return shutil.which(OPENROAD) is not None


def score_case(case_dir: Path, size_path: Path) -> float:
    """Score only. See score_case_detailed for the per-term split."""
    return score_case_detailed(case_dir, size_path)[0]


def score_case_detailed(case_dir: Path, size_path: Path) -> tuple[float, dict]:
    """Run the OpenROAD timing/power evaluation for one case. Raises on any
    failure or if OpenROAD is unavailable (the grader maps that to reward 0).

    Returns (score, breakdown) where breakdown carries the raw quantities the
    evaluator reports and the penalty each contributes, so the four terms can be
    compared across designs after the fact."""
    if not _openroad_available():
        raise RuntimeError(
            f"OpenROAD binary '{OPENROAD}' not found — the verifier image must "
            f"ship OpenROAD+CircuitOps (see tests/Dockerfile)."
        )
    if not size_path.exists() or size_path.stat().st_size == 0:
        raise ValueError("solver produced no non-empty output")

    cmd = [OPENROAD, "-python", str(EVAL_SCRIPT),
           "--case_dir", str(case_dir), "--size_file", str(size_path)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=SCORE_TIMEOUT)
    if p.returncode != 0:
        raise ValueError(f"OpenROAD evaluation failed: {p.stdout[-400:].strip()}")
    m = re.search(r"SCORE:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", p.stdout)
    if not m:
        raise ValueError(f"could not parse SCORE from OpenROAD output: {p.stdout[-400:]}")

    # The evaluator prints each term; a term it reports as clean ("No slew
    # violation") is absent from the output and contributes exactly 0.
    num = r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    raw = {}
    for pat, key in ((r"TNS:" + num, "tns_ns"),
                     (r"slew violation difference:" + num, "slew_ns"),
                     (r"capacitance violation difference:" + num, "cap_ff"),
                     (r"Leakage power difference:" + num, "leakage_uw")):
        mm = re.search(pat, p.stdout)
        raw[key] = float(mm.group(1)) if mm else 0.0
    breakdown = dict(raw)
    breakdown["penalty_tns"] = 10.0 * abs(raw["tns_ns"]) if raw["tns_ns"] < 0 else 0.0
    breakdown["penalty_slew"] = 20.0 * abs(raw["slew_ns"])
    breakdown["penalty_cap"] = 20.0 * abs(raw["cap_ff"])
    return float(m.group(1)), breakdown
