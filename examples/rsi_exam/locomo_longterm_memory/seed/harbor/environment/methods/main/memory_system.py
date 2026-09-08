"""Inherited baseline memory system — THIS FILE'S CONTRACT IS WHAT GETS GRADED.

The grader (and selfcheck.py) drives your submission exclusively through:

    build_memory_system(llm_call) -> object with
        .ingest(sessions)  sessions: list of (session_id, date_str, turns);
                           each turn is {"speaker": str, "text": str}.
                           May be called once with several independent
                           conversations merged together (session ids are
                           namespaced and never collide).
        .answer(question, question_time="") -> str
                           Free-text answer; short and specific scores best
                           (token-level F1 against a concise gold answer).

    llm_call(messages, max_tokens=512, temperature=0.0) -> str
        The ONLY LLM access. At grade time it is routed through a proxy that
        pins the base model; requests for other models are rewritten.

Everything else in this directory is yours to change, replace, or delete —
redesign the memory representation, indexing, retrieval, and answering
however you like, as long as the contract above still holds.

Baseline behavior: sliding-window LLM extraction into atomic memory entries,
a single BM25 keyword index (k=5), a tight 8-entry context, and one concise
answer call per question.
"""

from __future__ import annotations

import os

from memsys.answering import answer_freetext
from memsys.extractor import MemoryExtractor
from memsys.retrieval import MemoryIndex, format_context

KEYWORD_TOP_K = 5
MAX_CONTEXT = 8


class MemorySystem:
    def __init__(self, llm_call):
        self.llm_call = llm_call
        self.extractor = MemoryExtractor(llm_call)
        self.memories: list[dict] = []
        self.index: MemoryIndex | None = None

    def ingest(self, sessions):
        cache_dir = os.environ.get("MEMSYS_CACHE_DIR") or None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        mems, _ = self.extractor.extract_sessions(
            [tuple(s) for s in sessions], cache_dir=cache_dir,
        )
        self.memories.extend(mems)
        self.index = MemoryIndex(self.memories)

    def answer(self, question, question_time=""):
        hits = self.index.search(question, top_k=KEYWORD_TOP_K) if self.index else []
        if not hits:
            return "unknown"
        context = format_context(hits, max_items=MAX_CONTEXT)
        return answer_freetext(self.llm_call, question, context)


def build_memory_system(llm_call):
    return MemorySystem(llm_call)
