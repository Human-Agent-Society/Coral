"""Baseline solver for the fixed-budget conflict-minimization coloring task.

For each graph you are given a hard color budget `target_k` set to the graph's BEST-KNOWN
chromatic number: a proper (0-conflict) target_k-coloring provably EXISTS, but finding it
is at the coloring frontier and is very hard. Assign every vertex a color in [0, target_k)
so as to MINIMIZE the number of monochromatic (conflicting) edges — 0 is the target.

This inherited baseline is a single local-search run (the score-0 floor). Improving the
score means driving conflicts BELOW what it reaches within the per-instance time cap.
How you do that is up to you.

Contract (keep the name and signature):
    solve(data) -> {"colorings": [[color_for_vertex_0, ...], ...]}   # colors in [0, target_k)
    one coloring per instance, in the SAME order as data["instances"].
"""
from __future__ import annotations

import gc_lib as L

# The reference floor is measured by a FIXED number of TabuCol iterations (data field
# "ref_max_iters"), so the score-0 baseline is machine-independent — the same iteration count
# gives the same conflicts on any hardware. A wall backstop (safely under the 45s per-instance
# cap) guards against a slow machine; the iteration budget is calibrated to finish well within it.
SEED = 20240617
WALL_BACKSTOP = 44.0


def solve(data):
    colorings = []
    for inst in data["instances"]:
        n = int(inst["n_vertices"])
        k = int(inst["target_k"])
        edges = [(int(e[0]), int(e[1])) for e in inst["edges"]]
        adj = L.build_adj(n, edges)
        max_iters = int(inst.get("ref_max_iters", 10 ** 9))
        color, _ = L.reference_tabucol(n, adj, k, SEED, WALL_BACKSTOP, max_iters=max_iters)
        colorings.append(color)
    return {"colorings": colorings}
