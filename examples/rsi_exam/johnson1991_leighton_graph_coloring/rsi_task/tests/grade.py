#!/usr/bin/env python3
"""TWO-PHASE Harbor verifier for johnson1991_leighton_graph_coloring.

The trusted parent never imports the submission. It hides hidden_anchors.json, spawns a fresh
child process that imports /app/methods/main/solver.py and writes only decision.json, restores the
anchors, and scores that decision with the sealed source_evaluate.py. Any crash, timeout,
malformed output, or improper coloring writes reward 0.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

HELDOUT = Path(os.environ.get("HELDOUT_DIR", "/tests/heldout"))
SUBMISSION_DIR = Path(os.environ.get("SUBMISSION_DIR", "/app/methods/main"))
REWARD_DIR = Path(os.environ.get("VERIFIER_LOG_DIR", "/logs/verifier"))
CHILD = Path(os.environ.get("CHILD_SCRIPT", "/tests/child_solve.py"))

HIDDEN = HELDOUT / "hidden_data.json"
SCORER = HELDOUT / "source_evaluate.py"
ANCHORS = HELDOUT / "hidden_anchors.json"
# child runs the 7 hidden graphs one at a time, each capped at PER_INSTANCE_TIMEOUT_SEC (45s)
# inside child_solve.py; 7*45=315s worst case. This is the whole-batch safety ceiling.
CHILD_TIMEOUT = float(os.environ.get("CHILD_TIMEOUT_SEC", "600"))
SCORE_TIMEOUT = float(os.environ.get("SCORE_TIMEOUT_SEC", "300"))


def _run_child(out_json: Path) -> None:
    proc = subprocess.Popen(
        [sys.executable, str(CHILD), str(SUBMISSION_DIR), str(HIDDEN), str(out_json)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, _ = proc.communicate(timeout=CHILD_TIMEOUT)
        if stdout:
            print(stdout, file=sys.stderr)
    except subprocess.TimeoutExpired:
        print(f"child timed out after {CHILD_TIMEOUT}s", file=sys.stderr)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def main() -> None:
    REWARD_DIR.mkdir(parents=True, exist_ok=True)
    out = {"metric": None, "reward": 0.0, "correctness": False, "errors": []}
    stashed = None
    try:
        with tempfile.TemporaryDirectory(prefix="gc_out_") as tmp:
            os.chmod(tmp, 0o700)
            decision = Path(tmp) / "decision.json"
            if ANCHORS.exists():
                stashed = ANCHORS.with_name("hidden_anchors.json.stashed")
                ANCHORS.rename(stashed)
            _run_child(decision)
            if stashed is not None:
                stashed.rename(ANCHORS)
                stashed = None
            if not decision.exists() or decision.stat().st_size == 0:
                raise ValueError("solver produced no decision")
            proc = subprocess.run(
                [sys.executable, str(SCORER), str(HIDDEN), str(decision)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=SCORE_TIMEOUT,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"scorer failed: {proc.stderr[-500:].strip()} {proc.stdout[-500:].strip()}")
            raw = json.loads(proc.stdout)
        if raw.get("feasible") is True and raw.get("score") is not None:
            out = {
                "metric": float(raw.get("kpi", 0.0)),
                "reward": round(float(raw["score"]), 6),
                "correctness": True,
                "errors": [],
                "details": raw,
            }
        else:
            out = {"metric": None, "reward": 0.0, "correctness": False,
                   "errors": [str(raw.get("reason", "infeasible"))], "details": raw}
    except Exception as exc:  # fail closed
        out = {"metric": None, "reward": 0.0, "correctness": False,
               "errors": [f"{type(exc).__name__}: {exc}"]}
    finally:
        if stashed is not None and stashed.exists() and not ANCHORS.exists():
            stashed.rename(ANCHORS)

    rewards = {"reward": float(out["reward"])}   # kpi lives in grade_debug.json
    (REWARD_DIR / "reward.txt").write_text(f"{out['reward']}\n", encoding="utf-8")
    (REWARD_DIR / "reward.json").write_text(json.dumps(rewards), encoding="utf-8")
    (REWARD_DIR / "grade_debug.json").write_text(json.dumps(out), encoding="utf-8")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
