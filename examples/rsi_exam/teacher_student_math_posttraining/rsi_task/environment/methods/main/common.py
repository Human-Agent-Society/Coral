from __future__ import annotations

import re

PROMPT_PREFIX = (
    "Solve the following mathematics problem. Reason step by step and put only the final "
    "answer inside \\boxed{...}.\n\nProblem:\n"
)


def extract_integer(text: str) -> int | None:
    # A boxed answer is authoritative even if later prose repeats intermediate numbers.
    candidates = re.findall(r"\\boxed\s*\{\s*([+-]?\d[\d,]*)\s*\}", text)
    if not candidates:
        candidates = re.findall(
            r"(?:final\s+answer|answer)\s*(?:is|:|=)?\s*\\?\(?\s*([+-]?\d[\d,]*)",
            text, re.I,
        )
    try:
        return int(candidates[-1].replace(",", "")) if candidates else None
    except ValueError:
        return None


def extract_explicit_integer(text: str) -> int | None:
    candidates = re.findall(r"\\boxed\s*\{\s*([+-]?\d[\d,]*)\s*\}", text)
    if not candidates:
        candidates = re.findall(
            r"(?:final\s+answer|answer)\s*(?:is|:|=)?\s*\\?\(?\s*([+-]?\d[\d,]*)",
            text, re.I,
        )
    try:
        return int(candidates[-1].replace(",", "")) if candidates else None
    except ValueError:
        return None
