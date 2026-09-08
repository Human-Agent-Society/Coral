#!/usr/bin/env python3
"""Baseline heuristic for 2D guillotine glass cutting.

Usage:  python3 baseline.py <batch.csv> <defects.csv> <global_param.csv> <out.csv>

Deliberately simple "one item per full-height 1-cut column" floor. Items are
produced in a valid sequence order (stack-major, sequence-minor). Each item gets
its own column, laid left to right; within the column the item is slid to a
defect-free (x, y) found from defect-edge candidate positions, keeping every
waste band either 0 or >= 20 mm and every column >= 100 mm wide. When the next
column will not fit, a new plate starts; the right strip of the last plate is the
reusable residual. Valid but wasteful -> a low reward floor a real packer beats.
"""
import csv
import sys

PLATE_W = 6000
PLATE_H = 3210
MIN_WASTE = 20
MIN_1CUT = 100
MIN_2CUT = 100
MAX_1CUT = 3500


def read_csv(path):
    rows = []
    with open(path) as f:
        r = csv.reader(f, delimiter=';')
        next(r, None)
        for row in r:
            if row and any(c.strip() for c in row):
                rows.append(row)
    return rows


def main():
    batch_p, defects_p, global_p, out_p = sys.argv[1:5]

    items = []
    for row in read_csv(batch_p):
        items.append({"id": int(row[0]), "L": int(row[1]), "W": int(row[2]),
                      "stack": int(row[3]), "seq": int(row[4])})

    defects = {}
    for row in read_csv(defects_p):
        pid = int(row[1])
        x = float(row[2]); y = float(row[3]); w = float(row[4]); h = float(row[5])
        defects.setdefault(pid, []).append((x, y, x + w, y + h))

    gp = {row[0]: row[1] for row in read_csv(global_p)}
    global PLATE_W, PLATE_H
    PLATE_W = int(gp.get("widthPlates", PLATE_W))
    PLATE_H = int(gp.get("heightPlates", PLATE_H))

    items.sort(key=lambda it: (it["stack"], it["seq"]))

    def orient(it):
        cands = [(it["L"], it["W"]), (it["W"], it["L"])]
        legal = [(w, h) for (w, h) in cands
                 if h <= PLATE_H and w <= PLATE_W and w <= MAX_1CUT]
        if not legal:
            legal = cands
        legal.sort()
        return legal[0]

    def col_width(w):
        return w if w >= MIN_1CUT else max(MIN_1CUT, w + MIN_WASTE)

    def band_ok(top, bot):
        for b in (top, bot):
            if 0 < b < MIN_WASTE or 0 < b < MIN_2CUT:
                return False
        return True

    def defect_free(pid, x0, y0, x1, y1):
        for (dx0, dy0, dx1, dy1) in defects.get(pid, []):
            if x0 < dx1 and dx0 < x1 and y0 < dy1 and dy0 < y1:
                return False
        return True

    def place_y(pid, x, w, h):
        max_y = PLATE_H - h
        if max_y < 0:
            return None
        cands = {0, max_y}
        for (dx0, dy0, dx1, dy1) in defects.get(pid, []):
            if dx0 < x + w and x < dx1:      # defect in this column x-range
                cands.add(int(dy1))          # item just above defect
                cands.add(int(dy0) - h)      # item just below defect
        for y in sorted(c for c in cands if 0 <= c <= max_y):
            top = y
            bot = PLATE_H - (y + h)
            if not band_ok(top, bot):
                continue
            if defect_free(pid, x, y, x + w, y + h):
                return y
        return None

    def find_x(pid, x_min, w, h, colw):
        """Smallest legal column-start x >= x_min with a defect-free y."""
        cands = {x_min}
        for (dx0, dy0, dx1, dy1) in defects.get(pid, []):
            cands.add(int(dx1))              # start column just past a defect
        for x in sorted(c for c in cands if c >= x_min):
            if x != x_min and x < x_min + MIN_WASTE:
                x = x_min + MIN_WASTE
            if x + colw > PLATE_W:
                continue
            rem = PLATE_W - (x + colw)
            if not (rem == 0 or rem >= MIN_WASTE):
                continue
            y = place_y(pid, x, w, h)
            if y is not None:
                return x, y
        return None

    nodes = []
    node_counter = [0]

    def new_node(plate, x, y, w, h, typ, cut, parent):
        nid = node_counter[0]; node_counter[0] += 1
        nodes.append([plate, nid, x, y, w, h, typ, cut, parent])
        return nid

    plate = 0
    x_cursor = 0
    plate_root = {}
    plate_root[0] = new_node(0, 0, 0, PLATE_W, PLATE_H, -2, 0, "")

    def open_plate():
        nonlocal plate, x_cursor
        plate += 1
        x_cursor = 0
        plate_root[plate] = new_node(plate, 0, 0, PLATE_W, PLATE_H, -2, 0, "")

    def emit_column(pid, x, y, w, h, colw, item_id):
        col = new_node(pid, x, 0, colw, PLATE_H, -2, 1, plate_root[pid])
        if y > 0:
            new_node(pid, x, 0, colw, y, -1, 2, col)
        if colw > w:
            band = new_node(pid, x, y, colw, h, -2, 2, col)
            new_node(pid, x, y, w, h, item_id, 3, band)
            new_node(pid, x + w, y, colw - w, h, -1, 3, band)
        else:
            new_node(pid, x, y, w, h, item_id, 2, col)
        bot = PLATE_H - (y + h)
        if bot > 0:
            new_node(pid, x, y + h, colw, bot, -1, 2, col)

    def close_plate_leftover(as_residual):
        leftover = PLATE_W - x_cursor
        if leftover > 0:
            typ = -3 if as_residual else -1
            new_node(plate, x_cursor, 0, leftover, PLATE_H, typ, 1,
                     plate_root[plate])

    for it in items:
        w, h = orient(it)
        colw = col_width(w)
        placed = False
        for _attempt in range(3):
            spot = find_x(plate, x_cursor, w, h, colw)
            if spot is not None:
                x, y = spot
                if x > x_cursor:
                    new_node(plate, x_cursor, 0, x - x_cursor, PLATE_H, -1, 1,
                             plate_root[plate])
                emit_column(plate, x, y, w, h, colw, it["id"])
                x_cursor = x + colw
                placed = True
                break
            close_plate_leftover(as_residual=False)
            open_plate()
        if not placed:
            # last-resort: fresh plate, x=0, y=0 (defects extremely dense)
            close_plate_leftover(as_residual=False)
            open_plate()
            y = place_y(plate, 0, w, h)
            y = 0 if y is None else y
            emit_column(plate, 0, y, w, h, colw, it["id"])
            x_cursor = colw

    close_plate_leftover(as_residual=True)

    with open(out_p, "w", newline="") as f:
        wr = csv.writer(f, delimiter=';')
        wr.writerow(["PLATE_ID", "NODE_ID", "X", "Y", "WIDTH", "HEIGHT",
                     "TYPE", "CUT", "PARENT"])
        for n in nodes:
            wr.writerow(n)


if __name__ == "__main__":
    main()
