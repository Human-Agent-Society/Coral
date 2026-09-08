"""Sealed grader: correctness gate + per-case speedup vs frozen anchors.

The untrusted submission runs in a deprivileged, isolated child (restricted_runner);
this trusted parent only maps returned timings onto reward via sealed anchors the
child never sees.
"""
import json
import math
import sys
from pathlib import Path

TESTS = Path("/tests")
SUBMISSION = Path("/app/methods/main")
LOGS = Path("/logs/verifier")
CASE_TIMEOUT = 900


def write_reward(x):
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "reward.json").write_text(json.dumps({"reward": float(x)}))


def reward_of(speedup, sota, upper, rewards):
    """Log-linear from baseline(1.0)->0 to the roofline upper->1; no sota rung."""
    if not math.isfinite(speedup) or speedup <= 1.0:
        return 0.0
    if upper <= 1.0:
        return 1.0
    return min(1.0, math.log(speedup) / math.log(upper))


def main():
    sys.path.insert(0, str(TESTS))
    from eval import attn_eval
    from restricted_runner import run_case

    anchors = json.loads((TESTS / "anchors.json").read_text())
    manifest = json.loads((TESTS / "baseline_manifest.json").read_text())
    cases = attn_eval.load_strata(TESTS / "heldout_strata.json")

    details, rewards = [], []
    for i, cfg in enumerate(cases):
        anc = manifest[cfg["name"]]
        r = run_case(cfg, SUBMISSION, i, CASE_TIMEOUT)  # deprivileged child
        if r.get("ok") and r.get("t", 0) > 0:
            r["speedup"] = anc["baseline_t"] / r["t"]
            r["reward"] = reward_of(r["speedup"], anc["sota_speedup"],
                                    anc["upper_speedup"], anchors["rewards"])
        else:
            r["reward"] = 0.0
        details.append(r)
        rewards.append(r["reward"])

    (LOGS / "score_details.json").write_text(json.dumps(details, indent=2))
    write_reward(math.fsum(rewards) / len(rewards))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 - fail closed, never skip the reward file
        try:
            LOGS.mkdir(parents=True, exist_ok=True)
            (LOGS / "grader_error.txt").write_text(repr(e))
        finally:
            write_reward(0.0)
        raise
