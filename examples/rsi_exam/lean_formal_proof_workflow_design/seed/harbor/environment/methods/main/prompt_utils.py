"""Prompt and proof-body extraction helpers for the editable Lean workflow."""
from __future__ import annotations

import re


def proposal_messages(example):
    statement = example["formal_statement"].rstrip()
    problem = example.get("problem", "")
    return [
        {"role": "user", "content": (
            "Return only a valid Lean 4 proof body for this fixed theorem. Do not explain the proof, "
            "change the theorem, or use sorry/admit/axioms. A full theorem in a lean4 code block "
            "is accepted, but its proof must begin with `by`.\n\nNatural-language source:\n" +
            problem + "\n\nimport Mathlib\nimport Aesop\n\nset_option maxHeartbeats 400000\n\n"
            "open BigOperators\n\n" + statement + "\n  sorry"
        )},
    ]


def repair_messages(example, proof, diagnostics):
    statement = example["formal_statement"].rstrip()
    problem = example.get("problem", "")
    return [
        {"role": "user", "content": (
            "Repair the Lean 4 proof below using the compiler diagnostics. Return only one "
            "complete replacement proof body beginning with `by`. Do not change the theorem, "
            "add declarations, or use sorry/admit/axioms.\n\nNatural-language source:\n" +
            problem + "\n\nFixed theorem:\nimport Mathlib\nimport Aesop\n\n"
            "set_option maxHeartbeats 400000\n\nopen BigOperators\n\n" + statement +
            "\n\nRejected proof:\n" + proof +
            "\n\nLean diagnostics:\n" + diagnostics[-6000:]
        )},
    ]


def extract_proof(text):
    """Extract a proof term beginning with ``by`` from common model response formats."""
    text = text or ""
    blocks = re.findall(r"```(?:lean4?|Lean4?)?\s*\n?(.*?)```", text, flags=re.S)
    candidates = list(reversed(blocks)) + [text]
    for candidate in candidates:
        candidate = candidate.strip()
        marker = re.search(r":=\s*(by\b.*)\Z", candidate, flags=re.S)
        if marker:
            return marker.group(1).strip()
        direct = re.search(r"(?:\A|\n)\s*(by\b.*)\Z", candidate, flags=re.S)
        if direct:
            return direct.group(1).strip()
    return "by\n  aesop"


def safe_call(llm, messages, max_tokens):
    try:
        return llm(messages, max_tokens=max_tokens)
    except Exception:
        return ""
