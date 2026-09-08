"""TRUSTED simulator runner: run ONE hidden (scale, seed) simulation.

Invoked by grade.py as a subprocess so that a SUMO crash or a case timeout cannot take
the grader down with it.  Since the trust-boundary merge it imports NO
submitted code: `eval_core.run_sim()` starts the submitted controller as a further,
unprivileged child (uid `ctrl_uid`) that reaches SUMO only through the filtered proxy in
`sim_rpc.py`.

Consequently everything Cost is computed from is authored on this side of the boundary:
SUMO writes the tripinfo into a root-owned 0700 directory the controller uid cannot
write, and the teleport / unfinished counters are read from the live simulation.  The
controller never learns the tripinfo path and never touches the counters.  (Before the
merge this process imported controller.py, so `sys.argv[4]` handed the submission the
path of the very file it was graded from.)

The 5th argument is the tls.add.xml snapshot the PARENT already froze and
whitelist-checked (`eval_core.freeze_and_validate_additional`), or "" when the parent
found no additional file.  Either way it is passed straight through to `run_sim`, which
uses it verbatim: NO validation runs in this process.

Usage: run_eval.py <solution_dir> <scale> <seed> <tripinfo_out> [add_frozen_or_empty]
                   [<ctrl_uid> <ctrl_gid> <ctrl_scratch>]   ("-" = no privilege drop)
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from eval_core import run_sim  # trusted copy in tests/  # noqa: E402


def _opt(i):
    return sys.argv[i] if len(sys.argv) > i and sys.argv[i] not in ("", "-") else None


def main():
    solution_dir, scale, seed, tripinfo_out = (
        sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    # "" (parent says: no additional file) is NOT None — None would mean "freeze and
    # validate it yourself", which must never happen on this side of the boundary.
    add_frozen = sys.argv[5] if len(sys.argv) > 5 else None
    uid, gid, scratch = _opt(6), _opt(7), _opt(8)
    try:
        counters = run_sim(solution_dir, scale, seed, tripinfo_out, add_frozen,
                           controller_uid=int(uid) if uid else None,
                           controller_gid=int(gid) if gid else None,
                           controller_scratch=scratch)
        print(json.dumps({"ok": True, **counters}))
    except BaseException as exc:  # noqa: BLE001
        print(json.dumps({"ok": False,
                          "error": f"{type(exc).__name__}: {str(exc)[:300]}"}))


if __name__ == "__main__":
    main()
