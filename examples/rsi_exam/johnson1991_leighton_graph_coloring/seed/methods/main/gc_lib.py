"""Minimal helpers for the fixed-budget conflict-minimization coloring task.

Deliberately small: the agent gets the reference local search (the score-0 floor) and
basic utilities, not the hidden anchors or best-known certificates. `reference_tabucol`
is EXACTLY the method the floor anchor is measured with, so matching it earns ~0; the
task is to drive conflicts further down toward a proper coloring (0 conflicts, which is
provably achievable at each instance's target_k).
"""
from __future__ import annotations

import random
import time


def build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        u, v = int(u), int(v)
        adj[u].add(v)
        adj[v].add(u)
    return adj


def conflicts(color, edges):
    """Number of monochromatic edges under `color`."""
    return sum(1 for u, v in edges if color[int(u)] == color[int(v)])


def greedy_k(n, adj, k):
    """Least-conflict greedy k-coloring: vertices in descending-degree order, each
    takes the color in [0, k) shared by the fewest already-colored neighbours. A cheap
    starting point (usually far from a proper coloring at a tight target_k)."""
    order = sorted(range(n), key=lambda v: (-len(adj[v]), v))
    color = [-1] * n
    for v in order:
        cnt = [0] * k
        for w in adj[v]:
            if color[w] >= 0:
                cnt[color[w]] += 1
        best_c, best_cnt = 0, cnt[0]
        for c in range(1, k):
            if cnt[c] < best_cnt:
                best_c, best_cnt = c, cnt[c]
        color[v] = best_c
    return color


def reference_tabucol(n, adj, k, seed, seconds, max_iters=10 ** 9):
    """TabuCol (Hertz & de Werra 1987) from a greedy_k start, tracking the BEST (fewest
    conflicts) coloring seen. Returns (best_color, best_conflicts). This is the shipped
    baseline AND the exact method the floor anchor is measured with; the tabu plateaus
    early, so its best value is stable across machines. Improve on it in solver.py."""
    rng = random.Random(seed)
    end = time.monotonic() + seconds
    color = greedy_k(n, adj, k)
    gamma = [[0] * k for _ in range(n)]
    for v in range(n):
        for w in adj[v]:
            gamma[v][color[w]] += 1
    cur = sum(gamma[v][color[v]] for v in range(n)) // 2
    tabu = [[0] * k for _ in range(n)]
    best = cur
    best_color = color[:]
    it = 0
    while cur > 0 and it < max_iters and time.monotonic() < end:
        it += 1
        best_delta = 10 ** 9
        best_moves = []
        confv = [v for v in range(n) if gamma[v][color[v]] > 0]
        if not confv:
            break
        sample = confv if len(confv) <= 60 else rng.sample(confv, 60)
        for v in sample:
            gcur = gamma[v][color[v]]
            for c in range(k):
                if c == color[v]:
                    continue
                delta = gamma[v][c] - gcur
                if tabu[v][c] > it and (cur + delta) >= best:
                    continue
                if delta < best_delta:
                    best_delta = delta
                    best_moves = [(v, c)]
                elif delta == best_delta:
                    best_moves.append((v, c))
        if not best_moves:
            v = sample[rng.randrange(len(sample))]
            c = rng.randrange(k)
            best_delta = gamma[v][c] - gamma[v][color[v]]
            best_moves = [(v, c)]
        v, c = best_moves[rng.randrange(len(best_moves))]
        old = color[v]
        color[v] = c
        cur += gamma[v][c] - gamma[v][old]
        for w in adj[v]:
            gamma[w][old] -= 1
            gamma[w][c] += 1
        tabu[v][old] = it + rng.randrange(10) + int(0.6 * cur)
        if cur < best:
            best = cur
            best_color = color[:]
    return best_color, best
