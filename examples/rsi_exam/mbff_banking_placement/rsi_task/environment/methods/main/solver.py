"""Minimal template baseline for the multibit flip-flop banking and placement task.

It preserves every input flip-flop as the same library cell at the same coordinate
(no banking, no optimization), renames output instances, and emits one-to-one pin
mappings. The only non-trivial step is a minimal legalization pass: any flip-flop
whose input coordinate is off the placement-row site grid is snapped to the nearest
legal site, so the output is always a valid placement. This is a readable starting
template, not a competitive method.
"""

from __future__ import annotations

import bisect
from pathlib import Path


def _parse_testcase(path: Path):
    ff_lib_pins: dict[str, list[str]] = {}
    ff_instances: list[tuple[str, str, int, int]] = []
    rows: dict[int, tuple[int, int, int]] = {}  # row_y -> (start_x, site_w, num_sites)

    current_ff = None
    in_instances = False
    remaining_instances = 0

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0]

        if key == "FlipFlop":
            current_ff = parts[2]
            ff_lib_pins[current_ff] = []
            in_instances = False
            continue
        if key == "Gate":
            current_ff = None
            in_instances = False
            continue
        if key == "Pin" and current_ff:
            ff_lib_pins[current_ff].append(parts[1])
            continue
        if key == "PlacementRows":
            _, sx, sy, sw, _sh, n = parts
            rows[int(sy)] = (int(sx), int(sw), int(n))
            continue
        if key == "NumInstances":
            in_instances = True
            remaining_instances = int(parts[1])
            current_ff = None
            continue
        if in_instances and key == "Inst" and remaining_instances > 0:
            inst_name, lib_name, x, y = parts[1], parts[2], int(parts[3]), int(parts[4])
            if lib_name in ff_lib_pins:
                ff_instances.append((inst_name, lib_name, x, y))
            remaining_instances -= 1
            if remaining_instances == 0:
                in_instances = False

    return ff_lib_pins, ff_instances, rows


def _legalize(x: int, y: int, rows: dict[int, tuple[int, int, int]], row_ys: list[int]) -> tuple[int, int]:
    """Snap (x, y) to the nearest placement-row site. On-site coordinates are unchanged."""
    if not row_ys:
        return x, y
    # nearest row in y
    i = bisect.bisect_left(row_ys, y)
    cands = []
    if i < len(row_ys):
        cands.append(row_ys[i])
    if i > 0:
        cands.append(row_ys[i - 1])
    ny = min(cands, key=lambda ry: abs(ry - y))
    start_x, site_w, num_sites = rows[ny]
    k = round((x - start_x) / site_w)
    k = max(0, min(num_sites - 1, k))
    nx = start_x + k * site_w
    return nx, ny


def solve(input_path: str, output_path: str) -> None:
    ff_lib_pins, ff_instances, rows = _parse_testcase(Path(input_path))
    row_ys = sorted(rows)
    legal_ys = set(row_ys)

    placed: list[tuple[str, str, int, int]] = []
    for old_name, lib_name, x, y in ff_instances:
        on_site = y in legal_ys and rows and ((x - rows[y][0]) % rows[y][1] == 0)
        if not on_site:
            x, y = _legalize(x, y, rows, row_ys)
        placed.append((old_name, lib_name, x, y))

    lines: list[str] = [f"CellInst {len(placed)}"]
    rename = {}
    for idx, (old_name, lib_name, x, y) in enumerate(placed):
        new_name = f"ARA_FF_{idx}"
        rename[old_name] = new_name
        lines.append(f"Inst {new_name} {lib_name} {x} {y}")

    for old_name, lib_name, _, _ in placed:
        new_name = rename[old_name]
        for pin in ff_lib_pins[lib_name]:
            lines.append(f"{old_name}/{pin} map {new_name}/{pin}")

    Path(output_path).write_text("\n".join(lines) + "\n")
