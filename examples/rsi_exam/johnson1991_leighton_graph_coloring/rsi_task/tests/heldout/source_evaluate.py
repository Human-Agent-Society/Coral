#!/usr/bin/env python3
"""
Evaluator for johnson1991_leighton_graph_coloring — realcolor_v5.

Fixed-budget graph coloring (min-conflicts) on real DIMACS/COLOR02 graphs. Each
instance is an undirected graph plus a hard color budget target_k = the PUBLISHED
best-known chromatic number. Assign every vertex a color in [0, target_k); you may
NOT use a color >= target_k. A proper (0-conflict) target_k-coloring provably EXISTS
(target_k is a known achievable color count) but sits at the coloring frontier and is
very hard; the objective is to MINIMIZE the number of monochromatic (conflicting) edges.

Per-instance KPI = conflicts = number of monochromatic edges (lower better).
Per-instance normalized score:
    s_j = clip((floor_conf_j - conflicts_j) / (floor_conf_j - best_conf_j), 0, 1)
floor_conf = the shipped reference TabuCol (fixed iterations); best_conf = 0 (a proper
coloring provably exists at target_k -- EXTERNAL, unbeatable target). So this reduces to
s_j = clip(1 - conflicts_j / floor_conf_j, 0, 1). Aggregate score = mean over instances.

Fail closed PER INSTANCE: a wrong-length, non-integer/negative-color, out-of-range
(>= target_k), or empty (timed-out) coloring scores 0 for THAT instance only; the
rest of the batch is unaffected. Structural errors (no instances, wrong decision
shape, missing anchor) still zero the whole submission. Never crashes.

Run: python evaluate.py DATA_JSON DECISION_JSON
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK_ROOT = HERE.parent


def load_json(path):
    with Path(path).open() as f:
        return json.load(f)


def zero(reason):
    return {"feasible": False, "score": 0.0, "kpi": None,
            "direction": "min", "reason": reason, "details": {}}


def anchor_table(data):
    split = data.get("split")
    if split == "hidden":
        anchors = load_json(HERE / "hidden_anchors.json")
    elif split == "public":
        anchors = load_json(HERE / "public_anchors.json")
    else:
        raise ValueError(f"unknown split {split!r}")
    floor = {r["name"]: int(r["conflicts"]) for r in anchors["floor_anchor"]["per_instance"]}
    best = {r["name"]: int(r["conflicts"]) for r in anchors["sota_anchor"]["per_instance"]}
    return floor, best


def coloring_from_entry(entry, n, target_k):
    if isinstance(entry, dict):
        if "colors" not in entry:
            return None, "per-instance dict has no 'colors' field"
        entry = entry["colors"]
    if not isinstance(entry, list):
        return None, "coloring must be a list of integer colors"
    if len(entry) != n:
        return None, f"coloring length {len(entry)} != n_vertices {n}"
    for v, c in enumerate(entry):
        if isinstance(c, bool) or not isinstance(c, int):
            return None, f"color for vertex {v} is not an integer"
        if c < 0 or c >= target_k:
            return None, f"color for vertex {v} = {c} outside allowed range [0,{target_k})"
    return entry, ""


def score_instance(inst, dec_entry):
    n = int(inst["n_vertices"])
    edges = inst["edges"]
    target_k = int(inst["target_k"])
    if int(inst.get("n_edges", len(edges))) != len(edges):
        return False, None, "data instance malformed (n_edges mismatch)"
    coloring, err = coloring_from_entry(dec_entry, n, target_k)
    if coloring is None:
        return False, None, err
    conf = 0
    for e in edges:
        u, v = int(e[0]), int(e[1])
        if u < 0 or u >= n or v < 0 or v >= n:
            return False, None, f"edge endpoint out of range: {e}"
        if coloring[u] == coloring[v]:
            conf += 1
    return True, conf, ""


def extract_decision_list(decision, n):
    if not isinstance(decision, dict):
        return None
    seq = decision.get("colorings")
    if seq is None:
        seq = decision.get("assignments")
    if not isinstance(seq, list) or len(seq) != n:
        return None
    return seq


def evaluate(data_path, decision_path):
    try:
        data = load_json(data_path)
        decision = load_json(decision_path)
        instances = data.get("instances")
        if not isinstance(instances, list) or not instances:
            return zero("data has no instances")
        seq = extract_decision_list(decision, len(instances))
        if seq is None:
            return zero(f"decision must have 'colorings' as a list of length {len(instances)}")
        floor_tbl, best_tbl = anchor_table(data)

        per, scores, total_conf, n_feasible = [], [], 0, 0
        for inst, dec_entry in zip(instances, seq):
            name = inst["name"]
            floor_c = floor_tbl.get(name)
            best_c = best_tbl.get(name)
            if floor_c is None or best_c is None:
                return zero(f"missing anchor for instance {name}")  # build error -> whole 0
            ok, conf, err = score_instance(inst, dec_entry)
            if not ok:
                # PER-INSTANCE fail-closed: a timed-out / invalid coloring on THIS instance scores
                # 0 for this instance only; the rest of the batch is unaffected.
                scores.append(0.0)
                per.append({"name": name, "target_k": int(inst["target_k"]),
                            "conflicts": None, "floor_conflicts": floor_c,
                            "best_conflicts": best_c, "instance_score": 0.0,
                            "status": err or "invalid_or_timeout"})
                continue
            n_feasible += 1
            if floor_c <= best_c:
                s = 1.0 if conf <= best_c else 0.0
            else:
                s = max(0.0, min(1.0, (floor_c - conf) / (floor_c - best_c)))
            scores.append(s)
            total_conf += conf
            per.append({"name": name, "target_k": int(inst["target_k"]),
                        "conflicts": conf, "floor_conflicts": floor_c,
                        "best_conflicts": best_c, "instance_score": s, "status": "ok"})

        agg = sum(scores) / len(scores)  # mean over ALL instances (failed ones count as 0)
        frac_at_best = sum(1 for p in per if p["conflicts"] is not None
                           and p["conflicts"] <= p["best_conflicts"]) / len(per)
        return {"feasible": True, "score": agg, "kpi": float(total_conf), "direction": "min",
                "reason": "ok",
                "details": {"aggregate_score_mean_normalized": agg,
                            "total_conflicts": total_conf,
                            "n_feasible_instances": n_feasible,
                            "n_invalid_or_timed_out": len(per) - n_feasible,
                            "fraction_instances_at_or_below_best": frac_at_best,
                            "n_instances": len(per), "per_instance": per}}
    except Exception as exc:
        return zero(f"evaluation_error: {exc}")


def main(argv):
    if len(argv) != 3:
        print("Usage: python evaluate.py DATA_JSON DECISION_JSON", file=sys.stderr)
        return 2
    print(json.dumps(evaluate(argv[1], argv[2]), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
