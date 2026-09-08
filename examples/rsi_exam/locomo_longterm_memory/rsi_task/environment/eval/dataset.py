"""Visible dev-set loader: 4 multi-session conversations, 812 free-text QA.

Categories: 1 single-hop, 2 temporal, 3 multi-hop/inferential, 4 open-domain,
5 adversarial (gold = the adversarial answer). Scoring is token-level F1.
The sealed holdout is more conversations of the same kind.
"""

from __future__ import annotations

import json


def load_conversations(path, sample_indices=None, max_qa_per_sample=None):
    """Return a list of samples: {sample_id, sessions, qa_pairs}.

    sessions: list of (session_id, date_str, turns); turn = {speaker, text}.
    qa_pairs: list of {question, answer, category}.
    """
    with open(path) as f:
        raw = json.load(f)
    if sample_indices is None:
        sample_indices = list(range(len(raw)))

    samples = []
    for si in sample_indices:
        s = raw[si]
        conv = s["conversation"]
        session_keys = sorted(
            [k for k in conv
             if k.startswith("session_") and not k.endswith("_date_time")],
            key=lambda x: int(x.split("_")[1]),
        )
        sessions = []
        for sk in session_keys:
            date_str = conv.get(f"{sk}_date_time", "")
            turns_raw = conv[sk]
            if isinstance(turns_raw, str):
                try:
                    turns_raw = json.loads(turns_raw)
                except json.JSONDecodeError:
                    turns_raw = []
            turns = [
                {"speaker": t.get("speaker", "?"), "text": t.get("text", "")}
                for t in (turns_raw or [])
            ]
            sessions.append((sk, date_str, turns))

        qa_pairs = []
        for qa in s.get("qa", []):
            ref = qa.get("answer") or qa.get("adversarial_answer", "")
            qa_pairs.append({
                "question": qa["question"],
                "answer": str(ref),
                "category": int(qa.get("category", 0)),
            })
        if max_qa_per_sample is not None:
            qa_pairs = qa_pairs[:max_qa_per_sample]

        samples.append({
            "sample_id": str(s.get("sample_id", si)),
            "sessions": sessions,
            "qa_pairs": qa_pairs,
        })
    return samples


def merge_samples(samples):
    """Merge many samples into one (sessions, qa_pairs) pair for a single
    ingest — session ids get a per-sample prefix so they never collide."""
    all_sessions, all_qa = [], []
    for s in samples:
        for (sid, date, turns) in s["sessions"]:
            all_sessions.append((f"{s['sample_id']}::{sid}", date, turns))
        all_qa.extend(s["qa_pairs"])
    return all_sessions, all_qa
