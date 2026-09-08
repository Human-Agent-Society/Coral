#!/usr/bin/env python3
"""Rubric judge for the APEX case-work task.

The agent under test does the legal work itself: it reads the world, answers the
visible tasks, sees which binary rubric criteria its own answers satisfied, and
iterates. It then answers the held-out tasks blind and submits those answers.
This module is only the *scorer* — a pinned judge LLM that decides, per binary
rubric criterion, whether an answer satisfies it.

There is deliberately no solver here. An earlier version of this task had a
pinned gpt-4o "solver" answer the tasks from a memory the agent wrote; that made
the agent optimise for a single-shot completion's quirks rather than do the legal
work. The judge is a scoring instrument, not a stand-in for the agent.

The judge is pinned to the same instrument APEX-Agents itself grades with:
Gemini 3 Flash with thinking set to low (arXiv:2601.14242 §4.2), which they
validated against human labels at 98.5% accuracy (n=747 labels / 249 criteria,
1.3% false-positive rate). Reached over Google's OpenAI-compatible endpoint, so
this stays a plain /chat/completions call.

Config via env (pin these in task.toml so runs are reproducible):
    APEX_LLM_API_KEY, APEX_LLM_API_BASE, APEX_JUDGE_MODEL, APEX_JUDGE_EFFORT
Set APEX_LLM_MOCK=1 for a deterministic offline stub (keyword coverage) so the
plumbing is exercisable without a key.
"""
import json
import os
import re
import urllib.request
from pathlib import Path

MOCK = os.environ.get("APEX_LLM_MOCK") == "1"
API_KEY = os.environ.get("APEX_LLM_API_KEY", "")
API_BASE = os.environ.get("APEX_LLM_API_BASE", "https://api.openai.com/v1")
JUDGE_MODEL = os.environ.get("APEX_JUDGE_MODEL", "gemini-3-flash-preview")
# APEX-Agents grades with Gemini 3 Flash, thinking set to low (arXiv:2601.14242
# §4.2). Empty string omits the field, for endpoints that reject it (e.g. gpt-4o-mini).
JUDGE_EFFORT = os.environ.get("APEX_JUDGE_EFFORT", "low")
# Thinking tokens count against max_tokens on reasoning models; leave headroom.
MAX_TOKENS = int(os.environ.get("APEX_JUDGE_MAX_TOKENS", "512"))
JUDGE_SAMPLES = int(os.environ.get("APEX_JUDGE_SAMPLES", "1"))
MAX_ANSWER_CHARS = int(os.environ.get("APEX_MAX_ANSWER_CHARS", "24000"))


# --------------------------------------------------------------------------- #
# OpenAI-compatible chat call
# --------------------------------------------------------------------------- #
def chat(model, messages, temperature=0.0, max_tokens=MAX_TOKENS):
    """One judge call. Returns the assistant text.

    max_tokens must leave room for THINKING tokens, which count against it on
    reasoning models: at effort=low this judge spends ~24 tokens thinking before
    emitting its 1-token verdict, so a tight cap (we shipped 8) returns
    finish_reason='length' with completion_tokens=0 and a message carrying only
    'role' — no 'content' key at all. Raising the cap costs nothing, since only
    the tokens actually emitted are billed.

    A response without usable content is retried, then raised. It is never
    silently treated as a verdict: an empty string would read as "not YES" and
    quietly score the criterion as failed.
    """
    if MOCK:
        return _mock_chat(messages)
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    if JUDGE_EFFORT:
        payload["reasoning_effort"] = JUDGE_EFFORT
    body = json.dumps(payload).encode()
    last = None
    for _attempt in range(3):
        try:
            req = urllib.request.Request(
                API_BASE.rstrip("/") + "/chat/completions", data=body,
                headers={"Authorization": f"Bearer {API_KEY}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.loads(r.read())
            choice = out["choices"][0]
            text = (choice.get("message") or {}).get("content")
            if text and text.strip():
                return text
            last = RuntimeError(
                f"judge returned no content (finish_reason="
                f"{choice.get('finish_reason')!r}, usage={out.get('usage')}); "
                f"raise APEX_JUDGE_MAX_TOKENS above {max_tokens} if this is "
                f"finish_reason='length'")
        except Exception as e:  # transient network / endpoint flakiness -> retry
            last = e
    raise last


def _mock_chat(messages):
    """Deterministic stub: pass if >=60% of the criterion's salient tokens appear."""
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    crit = re.search(r"CRITERION:\s*(.*?)\s*ANSWER:", user, re.S)
    ans = re.search(r"ANSWER:\s*(.*)$", user, re.S)
    if crit and ans:
        toks = {w for w in re.findall(r"[A-Za-z0-9$§.,]+", crit.group(1).lower()) if len(w) > 3}
        hit = sum(1 for w in toks if w in ans.group(1).lower())
        return "YES" if toks and hit / len(toks) >= 0.6 else "NO"
    return "NO"


# --------------------------------------------------------------------------- #
# Judge: one answer vs each binary rubric criterion
# --------------------------------------------------------------------------- #
def judge(answer, rubric):
    """Score one answer against its binary rubric. Returns per-criterion detail.

    With APEX_JUDGE_SAMPLES > 1 each criterion is judged N times and decided by
    majority vote, which trims the judge's run-to-run variance at N x the cost.
    """
    answer = (answer or "")[:MAX_ANSWER_CHARS]
    results = []
    for crit in rubric:
        sys = ("You are a strict grader. Decide whether the ANSWER satisfies the "
               "CRITERION. The answer need not use the same wording, but it must "
               "actually assert what the criterion requires. Respond with exactly "
               "YES or NO, nothing else.")
        user = f"CRITERION: {crit['criteria']}\nANSWER: {answer}\n\nRespond with exactly YES or NO."
        votes = []
        for _ in range(max(1, JUDGE_SAMPLES)):
            verdict = chat(JUDGE_MODEL, [{"role": "system", "content": sys},
                                         {"role": "user", "content": user}])
            votes.append(verdict.strip().upper().startswith("YES"))
        passed = sum(votes) * 2 > len(votes)
        results.append({"id": crit.get("id"), "criteria": crit["criteria"],
                        "passed": passed, "votes": sum(votes), "samples": len(votes)})
    return results


def load_answers(path):
    """Read the agent's answers file: {task_name: answer_text}. Tolerant of a
    list-of-objects form and of task_id keys."""
    try:
        raw = json.loads(Path(path).read_text())
    except Exception:
        return {}
    if isinstance(raw, list):
        out = {}
        for item in raw:
            if isinstance(item, dict):
                k = item.get("task_name") or item.get("task_id")
                if k:
                    out[str(k)] = item.get("answer", "")
        return out
    if isinstance(raw, dict):
        return {str(k): (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
                for k, v in raw.items()}
    return {}


def score_answers(answers, tasks):
    """Judge the agent's answers for a list of tasks; return per-task detail plus
    the overall fraction of binary rubric criteria satisfied.

    `answers` may be keyed by task_name or task_id. A task with no answer scores
    0 on every criterion without burning a judge call.
    """
    details, total, passed, full = [], 0, 0, 0
    for t in tasks:
        ans = answers.get(t["task_name"]) or answers.get(t["task_id"]) or ""
        if ans.strip():
            jr = judge(ans, t["rubric"])
        else:
            jr = [{"id": c.get("id"), "criteria": c["criteria"], "passed": False,
                   "votes": 0, "samples": 0} for c in t["rubric"]]
        p = sum(1 for r in jr if r["passed"])
        is_full = p == len(jr) and len(jr) > 0
        full += int(is_full)
        details.append({"task_name": t["task_name"], "passed": p, "total": len(jr),
                        "pass_at_1": is_full,
                        "answered": bool(ans.strip()), "answer_chars": len(ans),
                        "criteria": jr})
        total += len(jr)
        passed += p
    n = len(tasks)
    return {
        # pass@1 is the SCORE: a task counts only if every one of its criteria is met.
        # This is APEX-Agents' own headline metric (arXiv:2601.14242 §4.3): "the criteria
        # are all must-haves - if any are not met, the task is incomplete."
        "metric": round(full / n, 4) if n else 0.0,
        "pass_at_1": full, "n_tasks": n,
        # criteria fraction is kept as a DIAGNOSTIC (APEX reports it as "mean score").
        # It is what the grader shows the agent per submission, because pass/fail alone
        # gives no gradient to improve against.
        "criteria_passed": passed, "criteria_total": total,
        "criteria_fraction": round(passed / total, 4) if total else 0.0,
        "tasks": details}
