#!/usr/bin/env python3
"""Baseline heuristic solver for the macro-controller task.

Reads one instance from stdin, writes an F/R/L/S button sequence to stdout.

Strategy (middling greedy, no macros): carry balls one at a time in index
order 0..M-1. For each ball k, walk a BFS shortest path to the ball, pick it
up (S), walk a BFS shortest path to basket k, drop it (S). Directions are
tracked so the correct turns (R/L) are emitted before each forward step (F).

This completes every instance within T (route length ~ 2X + O(M), and the
generator guarantees T >= 2X + 4M), but leaves a lot on the table: it uses no
macros and no route optimisation, so its button count is far above the strong
human anchors.
"""
import sys
from collections import deque

# directions: 0=Up, 1=Right, 2=Down, 3=Left  (matches official DIJ)
DIJ = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def main():
    data = sys.stdin.read().split()
    idx = 0

    def tok():
        nonlocal idx
        v = data[idx]
        idx += 1
        return v

    N = int(tok()); M = int(tok()); T = int(tok())
    # vertical walls: N rows, each a string of length N-1 (between (i,j) and (i,j+1))
    wall_v = [tok() for _ in range(N)]
    # horizontal walls: N-1 rows, each length N (between (i,j) and (i+1,j))
    wall_h = [tok() for _ in range(N - 1)]
    balls = []
    baskets = []
    for _ in range(M):
        b = int(tok()); c = int(tok()); d = int(tok()); e = int(tok())
        balls.append((b, c))
        baskets.append((d, e))

    def can_move(i, j, dr):
        di, dj = DIJ[dr]
        i2, j2 = i + di, j + dj
        if i2 < 0 or i2 >= N or j2 < 0 or j2 >= N:
            return False
        if di == 0:  # horizontal move -> vertical wall
            return wall_v[i][min(j, j2)] == '0'
        else:        # vertical move -> horizontal wall
            return wall_h[min(i, i2)][j] == '0'

    def bfs_dirs(src, dst):
        """Return list of step-directions along a shortest path src->dst."""
        if src == dst:
            return []
        prev = [[None] * N for _ in range(N)]
        prevd = [[-1] * N for _ in range(N)]
        seen = [[False] * N for _ in range(N)]
        seen[src[0]][src[1]] = True
        q = deque([src])
        while q:
            i, j = q.popleft()
            if (i, j) == dst:
                break
            for dr in range(4):
                if not can_move(i, j, dr):
                    continue
                di, dj = DIJ[dr]
                i2, j2 = i + di, j + dj
                if not seen[i2][j2]:
                    seen[i2][j2] = True
                    prev[i2][j2] = (i, j)
                    prevd[i2][j2] = dr
                    q.append((i2, j2))
        # reconstruct
        dirs = []
        cur = dst
        while cur != src:
            dr = prevd[cur[0]][cur[1]]
            dirs.append(dr)
            cur = prev[cur[0]][cur[1]]
        dirs.reverse()
        return dirs

    ops = []
    pos = (0, 0)
    facing = 1  # Right

    def turn_to(target):
        nonlocal facing
        diff = (target - facing) % 4
        if diff == 1:
            ops.append('R')
        elif diff == 3:
            ops.append('L')
        elif diff == 2:
            ops.append('R'); ops.append('R')
        facing = target

    def walk_to(dst):
        nonlocal pos
        for dr in bfs_dirs(pos, dst):
            turn_to(dr)
            ops.append('F')
            di, dj = DIJ[dr]
            pos = (pos[0] + di, pos[1] + dj)

    for k in range(M):
        walk_to(balls[k])
        ops.append('S')      # pick up ball k
        walk_to(baskets[k])
        ops.append('S')      # drop ball k into basket k

    sys.stdout.write(''.join(ops) + '\n')


if __name__ == '__main__':
    main()
