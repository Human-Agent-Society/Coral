#!/usr/bin/env python3
"""Sealed oracle-normalized scorer for the noisy continuous task."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from oracle_scoring import score_minimization_traces  # noqa: E402


def main(argv: list[str]) -> int:
    try:
        decision = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        anchors = json.loads((HERE / "frozen_anchors.json").read_text(encoding="utf-8"))
        result = score_minimization_traces(decision, anchors, HERE)
    except Exception as exc:  # noqa: BLE001
        result = {"feasible": False, "score": 0.0, "score_anytime": 0.0, "score_final": 0.0,
                  "reason": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
