"""Free local dry-run on the visible sample (hidden scores differ; do not overfit this).

Runs your environment/methods/main/triage on environment/data/grade.jsonl and reports the
same TriageUtility the sealed verifier uses, plus the per-site breakdown so you can see
whether your risks generalize across sites. Pure stdlib.

    python environment/selfcheck.py
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "methods" / "main"))
from solver import triage  # noqa: E402

CRITICAL = ["Effusion", "Cardiomegaly", "Edema", "Pneumonia",
            "Consolidation", "Atelectasis", "Pneumothorax", "Mass"]
BUDGET, N_BINS = 0.20, 15


def read_jsonl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def brier(pairs):
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs) if pairs else 0.0


def main():
    data = ROOT / "data"
    grade = read_jsonl(data / "grade.jsonl")
    calib = read_jsonl(data / "calibration.jsonl")
    cases = [{k: v for k, v in c.items() if not k.startswith("gold_")} for c in grade]
    sites = sorted({c["site_id"] for c in grade})
    preds = triage(cases, {"budget": BUDGET, "labels": CRITICAL, "calibration": calib,
                           "sites": sites, "n_bins": N_BINS}, 0)
    risk = {p["study_uid"]: min(1.0, max(0.0, float(p["risk"]))) for p in preds}
    k = math.ceil(BUDGET * len(grade))
    referred = {c["study_uid"] for c in sorted(grade, key=lambda c: (-risk[c["study_uid"]], c["study_uid"]))[:k]}
    sens, briers = [], []
    print(f"{'site':>6} {'n':>6} {'crit':>6} {'sens':>7} {'brier':>7}")
    for s in sites:
        sc = [c for c in grade if c["site_id"] == s]
        crit = [c for c in sc if int(c["gold_critical"]) == 1]
        e = brier([(risk[c["study_uid"]], int(c["gold_critical"])) for c in sc])
        briers.append(e)
        if crit:
            sv = sum(1 for c in crit if c["study_uid"] in referred) / len(crit)
            sens.append(sv)
            print(f"{s:>6} {len(sc):>6} {len(crit):>6} {sv:>7.3f} {e:>7.3f}")
    ms, mn, me = sum(sens) / len(sens), min(sens), sum(briers) / len(briers)
    u = ms - 1.0 * me - 0.5 * (ms - mn)
    print(f"\nmean_sens={ms:.3f}  min_sens={mn:.3f}  mean_brier={me:.3f}")
    print(f"triage_utility_pct (visible) = {100 * max(0.0, min(1.0, u)):.3f}")
    print("(the hidden grade uses different studies; aim for a policy that generalizes across sites)")

    # Budget reminder. /app/budget.py is mounted by the harness; when it is not mounted the
    # whole block is skipped, so running this task standalone is unaffected. check=False so a
    # failing budget.py can never take selfcheck down with it. Placed at the end of main() so
    # the reminder prints after this run's score.
    import os as _os, subprocess as _sp, sys as _sys
    if _os.path.exists("/app/budget.py"):
        _sys.stdout.flush()   # without this the reminder prints before the score in a pipe
        _sp.run([_sys.executable, "/app/budget.py"], check=False)


if __name__ == "__main__":
    main()
