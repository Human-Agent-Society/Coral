#!/usr/bin/env python3
"""Free, unlimited self-check: run the current solver on the visible cases and
report per-case legality + the real graded score (via OpenROAD).

    python /app/selfcheck.py            # all visible cases
    python /app/selfcheck.py <case>     # just that case

Legality here is the SAME structural gate the grader applies. The score is the
SAME OpenROAD evaluation the sealed grader uses (leakageΔ + 10·|TNS| + 20·slew +
20·cap), just computed here on the VISIBLE cases so you can see whether a change
helped — lower is better. The OpenROAD pass takes minutes on the
large cases. An illegal sizing scores nothing.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "tools"))
sys.path.insert(0, str(HERE / "methods" / "main"))

import legality  # noqa: E402
import solver  # noqa: E402

DATA = Path("/app/data")
EVAL_SCRIPT = HERE / "eval" / "evaluate_case.py"
SCORE_TIMEOUT = float(os.environ.get("SELFCHECK_SCORE_TIMEOUT_SEC", "3000"))


def _score(case_dir: Path, size_path: Path):
    """Real OpenROAD score + its breakdown, or (None, None) if unavailable."""
    if shutil.which("openroad") is None or not EVAL_SCRIPT.exists():
        return None, None
    with tempfile.TemporaryDirectory() as wk:
        env = {**os.environ, "PYTHONPATH": str(EVAL_SCRIPT.parent), "EVAL_WORKDIR": wk}
        p = subprocess.run(
            ["openroad", "-python", str(EVAL_SCRIPT),
             "--case_dir", str(case_dir), "--size_file", str(size_path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=SCORE_TIMEOUT, env=env,
        )
    m = re.search(r"SCORE:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", p.stdout)
    if not m:
        return None, None
    # Where the score came from: the evaluator reports each term separately.
    parts = {}
    for pat, key in ((r"TNS:\s*([-\d.eE+]+)", "TNS_ns"),
                     (r"slew violation difference:\s*([-\d.eE+]+)", "slew"),
                     (r"capacitance violation difference:\s*([-\d.eE+]+)", "cap"),
                     (r"Leakage power difference:\s*([-\d.eE+]+)", "leakage_uW")):
        mm = re.search(pat, p.stdout)
        parts[key] = float(mm.group(1)) if mm else 0.0
    parts["penalty_TNS"] = 10.0 * abs(parts["TNS_ns"]) if parts["TNS_ns"] < 0 else 0.0
    parts["penalty_slew"] = 20.0 * abs(parts["slew"])
    parts["penalty_cap"] = 20.0 * abs(parts["cap"])
    return float(m.group(1)), parts


def run_case(case: str) -> dict:
    case_dir = DATA / case
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / f"{case}.size"
        solver.solve(str(case_dir), str(out))
        rep = legality.check_case(case_dir / "IR_Tables", out)
        rep["score"], rep["breakdown"] = _score(case_dir, out) if rep["legal"] else (None, None)
    rep["case"] = case
    return rep


def main() -> int:
    if len(sys.argv) > 1:
        cases = sys.argv[1:]
    else:
        cases = json.loads((DATA / "manifest.json").read_text())["cases"]
    all_legal = True
    for case in cases:
        rep = run_case(case)
        all_legal &= rep["legal"]
        status = "LEGAL" if rep["legal"] else "ILLEGAL"
        score = "n/a (openroad unavailable)" if rep.get("score") is None and rep["legal"] \
            else ("-" if not rep["legal"] else f"{rep['score']:.3f}")
        print(f"[{case}] {status}  instances={rep['submitted_instances']}/"
              f"{rep['expected_instances']}  score(lower=better)={score}")
        b = rep.get("breakdown")
        if b:
            print(f"    where it comes from:  10*|TNS|={b['penalty_TNS']:.3f}"
                  f"  20*slew={b['penalty_slew']:.3f}  20*cap={b['penalty_cap']:.3f}"
                  f"  leakage={b['leakage_uW']:.3f}")
            print(f"    raw:  TNS={b['TNS_ns']:.3f} ns  slew={b['slew']:.3f} ns"
                  f"  cap={b['cap']:.3f} fF   (a term is switched OFF once it hits 0)")
        if not rep["legal"]:
            for e in rep["errors"]:
                print(f"    - {e}")
            for e in rep["illegal_examples"]:
                print(f"    · {e}")
    print("OK" if all_legal else "FAIL")

    # Budget reminder (docs/budget-hookup.md). Silent unless the overlay mounted it,
    # so a standalone run of this task is unaffected.
    if Path("/app/budget.py").exists():
        import subprocess
        subprocess.run([sys.executable, "/app/budget.py"], check=False)
    return 0 if all_legal else 1


if __name__ == "__main__":
    raise SystemExit(main())
