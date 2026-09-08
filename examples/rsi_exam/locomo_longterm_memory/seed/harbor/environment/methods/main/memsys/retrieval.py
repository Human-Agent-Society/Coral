"""Keyword (BM25) retrieval over extracted memories — the inherited retrieval layer.

One view, small k, tight context. This is deliberately minimal: it indexes each
memory's content text with BM25 and returns the top-k lexical matches.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[^a-z0-9\s]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.sub(" ", str(text).lower()).split()


class MemoryIndex:
    """BM25 index over a list of memory dicts (as produced by MemoryExtractor)."""

    def __init__(self, memories: list[dict]):
        self.memories = list(memories)
        self._bm25 = None
        corpus = [tokenize(m.get("content", "")) for m in self.memories]
        if corpus:
            try:
                from rank_bm25 import BM25Okapi
                self._bm25 = BM25Okapi(corpus)
            except ImportError:
                logger.warning("rank_bm25 not installed — keyword search disabled")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top_k memories ranked by BM25 score (descending)."""
        if self._bm25 is None or not self.memories:
            return []
        toks = tokenize(query)
        if not toks:
            return []
        scores = self._bm25.get_scores(toks)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self.memories[i] for i in order[:top_k] if scores[i] > 0]


def format_context(memories: list[dict], max_items: int = 8) -> str:
    """Render retrieved memories as a numbered context block."""
    lines = []
    for i, m in enumerate(memories[:max_items], 1):
        ts = m.get("timestamp") or ""
        loc = m.get("location") or ""
        tag = " ".join(x for x in (str(ts), str(loc)) if x and x != "None")
        prefix = f"[{tag}] " if tag else ""
        lines.append(f"{i}. {prefix}{m.get('content', '')}")
    return "\n".join(lines)
