#!/usr/bin/env python3
"""Sealed grader: judge the agent's blind answers to the 5 held-out tasks.

The agent worked the visible tasks with rubric feedback, then answered the
held-out tasks blind (it never saw their rubrics) and submitted
methods/main/answers.json. Here the pinned judge scores those answers against
the sealed rubrics; the metric is the fraction of binary criteria satisfied,
scored with pass@1 -- a held-out task counts only if EVERY one of its binary
criteria is met, which is APEX-Agents' own headline metric (arXiv:2601.14242
§4.3). The criteria fraction is still computed and logged as a diagnostic, and
it is what the grader shows the agent during the visible phase, since pass/fail
alone gives nothing to improve against.

The gold ceiling is measured, not assumed: it is what the reference answers
themselves score under this same judge, so judge strictness cancels out.
"""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import apex_engine as E

SUBMISSION_DIR = Path(os.environ.get("SUBMISSION_DIR", "/app/methods/main"))
HELDOUT_DIR = Path(os.environ.get("HELDOUT_DIR", "/heldout"))
LOG_DIR = Path(os.environ.get("VERIFIER_LOG_DIR", "/logs/verifier"))
TASKS = json.loads((HELDOUT_DIR / "sealed_rubrics.json").read_text())["tasks"]

# Anchors live in a sealed root-owned file, never in [verifier.env]:
# anything injected there is readable from /proc/1/environ by submitted code.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "anchors.json")) as _f:
    _ANCHORS = json.load(_f)
A_BASELINE = float(_ANCHORS["BASELINE"])  # unanswered floor -> 0
A_SOTA = float(_ANCHORS["SOTA"])          # gold answers     -> 1


def reward_of(metric):
    if metric <= A_BASELINE:
        return 0.0
    return float(min(1.0, (metric - A_BASELINE) / max(1e-9, A_SOTA - A_BASELINE)))


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    answers = E.load_answers(SUBMISSION_DIR / "answers.json")
    res = E.score_answers(answers, TASKS)
    reward = round(reward_of(res["metric"]), 6)

    # reward.json keeps only the reward: the hub sorts the keys and displays
    # the first numeric one, so `metric` would be shown instead. Everything
    # else is already in score_details.json below.
    (LOG_DIR / "reward.json").write_text(json.dumps({"reward": reward}))
    (LOG_DIR / "score_details.json").write_text(json.dumps(
        {"anchors": {"baseline": A_BASELINE, "gold_ceiling": A_SOTA},
         "judge_model": E.JUDGE_MODEL, "judge_samples": E.JUDGE_SAMPLES, **res}, indent=1))

    missing = [t["task_name"] for t in res["tasks"] if not t["answered"]]
    if missing:
        print(f"WARNING: no answer submitted for {len(missing)} task(s): {', '.join(missing)}")
    print(f"pass@1={res['pass_at_1']}/{res['n_tasks']} = {res['metric']:.4f} "
          f"reward={reward:.6f}  |  criteria {res['criteria_passed']}/"
          f"{res['criteria_total']} = {res['criteria_fraction']:.1%} (diagnostic)")


if __name__ == "__main__":
    main()
