#!/usr/bin/env python3
"""Free, faithful local self-check for the MBFF banking & placement task.

Runs your CURRENT methods/main/solver.py on the visible cases through the SAME legality +
scoring gate the sealed grader uses (sanity, placement_checker, preliminary-evaluator,
bundled here under tools/), so a candidate that passes here will pass the same gate at
grading. The held-out cases stay sealed and are scored under a different cost-weight regime —
but the validity verdict and score formula are identical, so validate here before relying on a
change.

    python /app/selfcheck.py                       # all visible cases (slowest, ~3.4 min)
    python /app/selfcheck.py sampleCase            # one tiny case (seconds) — fast smoke test
    python /app/selfcheck.py testcase1_0812.txt    # one large case (~80 s) — fast iteration

The checker pipeline (esp. placement_checker) is slow on the large cases, so iterate against
a single case and run the full set only before relying on a change.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TOOLS = HERE / "tools"
TIMEOUT = 1800.0

FAIL_MARKERS = ("Fail !!!", "Fail, reason", "is overlap with", "ERROR[ONSITE]")


def _has_failure(text: str) -> bool:
    return any(m in text for m in FAIL_MARKERS)


def _run(cmd, timeout=TIMEOUT):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=timeout)
    return p.returncode, p.stdout


def _parse_score(text: str) -> float:
    for pat in (r"Final score:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
                r"Score:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"):
        m = re.search(pat, text)
        if m:
            return float(m.group(1))
    raise ValueError("could not parse final score")


def _load_solve():
    sys.path.insert(0, str(HERE / "methods" / "main"))
    spec = importlib.util.spec_from_file_location(
        "main_solver", HERE / "methods" / "main" / "solver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.solve


def _select_cases() -> list[Path]:
    """All visible cases by default; or only the names/globs passed on the command line."""
    if len(sys.argv) > 1:
        picked: list[Path] = []
        for arg in sys.argv[1:]:
            p = DATA / arg
            if p.is_file():
                picked.append(p)
            else:
                picked.extend(sorted(DATA.glob(arg)))
        cases = sorted(set(picked))
        if not cases:
            raise SystemExit(f"no visible case matched {sys.argv[1:]} in data/")
        return cases
    return sorted(p for p in DATA.iterdir() if p.is_file())


def main() -> None:
    solve = _load_solve()
    cases = _select_cases()
    if not cases:
        raise SystemExit("no visible cases in data/")
    scores, any_fail = [], False
    with tempfile.TemporaryDirectory() as tmp:
        for case in cases:
            out = Path(tmp) / f"{case.name}.out"
            try:
                solve(str(case), str(out))
            except Exception as exc:  # noqa: BLE001
                print(f"[FAIL] {case.name}: solver crashed: {type(exc).__name__}: {exc}")
                any_fail = True
                continue
            if not out.exists() or out.stat().st_size == 0:
                print(f"[FAIL] {case.name}: empty/no output")
                any_fail = True
                continue
            bad = False
            for tool in ("sanity", "placement_checker"):
                rc, so = _run([str(TOOLS / tool), str(case), str(out)])
                if rc != 0 or _has_failure(so):
                    print(f"[ILLEGAL] {case.name}: {tool} rejected: {so[-200:].strip()}")
                    any_fail = bad = True
                    break
            if bad:
                continue
            rc, so = _run([str(TOOLS / "preliminary-evaluator"), str(case), str(out)])
            if rc != 0 or _has_failure(so):
                print(f"[ILLEGAL] {case.name}: evaluator rejected: {so[-200:].strip()}")
                any_fail = True
                continue
            try:
                s = _parse_score(so)
                scores.append(s)
                print(f"[OK] {case.name}: legal, score={s:.6g}  (lower is better)")
            except ValueError as exc:
                print(f"[FAIL] {case.name}: {exc}")
                any_fail = True

    print("-" * 60)
    if any_fail:
        print("RESULT: at least one visible case is INVALID — fix legality before relying "
              "on this change (an illegal placement earns no credit at grading).")
    else:
        print(f"RESULT: all visible cases LEGAL. mean score = {sum(scores)/len(scores):.6g} "
              f"(proxy only — held-out cases use a different cost-weight regime).")

    # Budget reminder (docs/budget-hookup.md). Silent unless the overlay mounted it,
    # so a standalone run of this task is unaffected.
    if Path("/app/budget.py").exists():
        import subprocess
        subprocess.run([sys.executable, "/app/budget.py"], check=False)


if __name__ == "__main__":
    main()
