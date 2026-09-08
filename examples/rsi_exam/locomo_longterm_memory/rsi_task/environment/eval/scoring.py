"""Token-level F1 for the visible dev-set QA (number-word normalization included)."""

from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
    "once": "1", "twice": "2", "thrice": "3", "single": "1", "double": "2",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
}


def tokenize(s: str) -> list[str]:
    toks = _PUNCT_RE.sub(" ", str(s).lower()).split()
    return [_NUMBER_WORDS.get(t, t) for t in toks]


def token_f1(prediction: str, reference: str) -> float:
    p, r = tokenize(prediction), tokenize(reference)
    if not p or not r:
        return 0.0
    rc = list(r)
    c = 0
    for t in p:
        if t in rc:
            c += 1
            rc.remove(t)
    if c == 0:
        return 0.0
    pr, rec = c / len(p), c / len(r)
    return 2 * pr * rec / (pr + rec)
