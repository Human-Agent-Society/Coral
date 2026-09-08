"""Split the source benchmark into the visible dev set and the sealed holdout.

Visible = conversations {3, 5, 6, 9} (812 QA, 40.9% of the total; per-category
proportions within 1% of the global distribution). Holdout = the other six
conversations (1,174 QA). Run offline by the task author; both outputs are
committed (visible into environment/data/, holdout into tests/heldout/).

Usage:
    python3 prepare_split.py <locomo10.json> <visible_out.json> <holdout_out.json>
"""

import json
import sys

VISIBLE = [3, 5, 6, 9]
HOLDOUT = [0, 1, 2, 4, 7, 8]


def main(src, visible_out, holdout_out):
    with open(src) as f:
        raw = json.load(f)
    with open(visible_out, "w") as f:
        json.dump([raw[i] for i in VISIBLE], f)
    with open(holdout_out, "w") as f:
        json.dump([raw[i] for i in HOLDOUT], f)
    for name, idxs in (("visible", VISIBLE), ("holdout", HOLDOUT)):
        n = sum(len(raw[i].get("qa", [])) for i in idxs)
        print(f"{name}: convs {idxs} -> {n} QA")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
