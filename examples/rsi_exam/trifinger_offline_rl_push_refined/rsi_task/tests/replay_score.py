"""TRUSTED scorer (sealed): replay each recorded action trace on a FRESH env and compute the mean
return over the held-out episodes. No submission code is imported here — only the child's plain
action-trace data is replayed.

argv: replay_score.py <traces_json>

This file no longer sees the reward anchors at all. It used to take them on
argv and call the RETIRED trifinger_score.reward_of (landing 0/0.5/0.9/1.0, no clamp above `upper`),
printing a reward that grade.py deliberately discarded. The mapping now exists once, in
grade.py::reward_of, and the anchors never leave the trusted parent. This file is 0400 root in the
verifier image; only the trusted root parent runs it. Prints a JSON result to stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trifinger_score as ts
from heldout_seeds import HELDOUT_SEEDS


# An invalid trace is a SUBMISSION-side failure and must stay distinguishable from "the scorer
# itself crashed". replay_episode raises ValueError on non-finite or wrongly shaped actions,
# and those actions come from an untrusted child. A plain raise would exit rc=1, which grade.py
# reads as a trusted-scorer crash -> infra failure, letting a submission that trained to NaN
# launder its zero into "infrastructure fault, please re-run". A dedicated exit code
# (grade.py::INVALID_TRACE_RC) marks it as submission-side instead.
INVALID_TRACE_RC = 3


def main() -> int:
    traces_json = Path(sys.argv[1])
    traces = json.loads(traces_json.read_text())

    env = ts.make_env(data_dir=None)
    returns = []
    for seed_k in HELDOUT_SEEDS:
        tr = traces.get(str(seed_k))
        if not tr:
            returns.append(0.0)
            continue
        try:
            rep = ts.replay_episode(env, seed_k, tr)
        except ValueError as exc:      # trace validation failed: submission-side, not a scorer fault
            print(f"invalid action trace: {exc}", file=sys.stderr)
            return INVALID_TRACE_RC
        returns.append(rep["return"])

    metric = sum(returns) / len(returns) if returns else 0.0
    # This file used to also call the RETIRED trifinger_score.reward_of
    # (landing 0/0.5/0.9/1.0, no clamp above `upper`) and print its value, which grade.py then threw
    # away. Two mappings in the chain, one of them wrong, one of them ignored — the copy is gone.
    # grade.py::reward_of is the single source of truth; this file only reports the raw metric.
    print(json.dumps({"metric": metric,
                      "n_episodes": len(returns), "per_episode_return": returns}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
