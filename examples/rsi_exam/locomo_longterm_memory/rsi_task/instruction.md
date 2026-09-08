# Redesign a conversational memory system that generalizes to unseen conversations

You inherit a minimal long-term-memory system for LLM agents: it
extracts atomic memory entries from multi-session conversations with an LLM,
indexes them with BM25 only (k=5, 8-entry context), and answers questions with
one concise LLM call. Your goal is to redesign this memory architecture —
extraction, storage, retrieval, and answering are all yours to change — using
the visible conversations for development. Your submission is re-run by a
sealed verifier on unseen conversations of the same kind and scored by
token-level F1, so improvements must generalize beyond the conversations you
can see.

## Hard Constraints

- Keep the grading contract of `/app/methods/main/memory_system.py` intact:
  `build_memory_system(llm_call)` returning an object with
  `.ingest(sessions)` and `.answer(question, question_time="")`.
  The docstring in that file is the authoritative contract.
- Everything you submit must live under `/app/methods/` — only that directory
  is collected for grading.
- All LLM access must go through the `llm_call` the harness passes in. At
  grade time it is routed through a proxy that pins the base model
  (gpt-4o-mini); requests for any other model are rewritten to it, and your
  code never receives a real API key.
- Only preinstalled packages are available at grade time (openai, rank_bm25,
  numpy, sentence-transformers with locally cached `BAAI/bge-base-en-v1.5` and
  `all-MiniLM-L6-v2`, scikit-learn, pandas, networkx, pyyaml). The verifier
  cannot install packages for your code.
- Answers must be derived from the ingested sessions via your memory system —
  no hardcoded question-answer mappings.
- `.answer()` must return a string; a question whose call raises scores 0 —
  prefer a best-effort answer over an exception.
- `.ingest()` may be called with several independent conversations merged into
  one batch (namespaced session ids); do not assume a single continuous
  conversation.

## What You Have

- `/app/methods/main/`: the inherited baseline (extraction -> BM25 index ->
  concise answer). **This is what gets graded**; improve it in place or
  rewrite it.
- `/app/data/conversations_visible.json`: the visible dev set — 4
  multi-session conversations (19-31 sessions each), 812 free-text QA pairs
  in 5 categories (single-hop, temporal, multi-hop/inferential, open-domain,
  adversarial).
- `/app/selfcheck.py`: free local scoring on the visible set (token F1), with
  per-category breakdown, conversation/QA subsetting for cheap runs, and an
  extraction cache flag. A full cold visible run with the baseline takes
  ~70 min (extraction dominates); cached re-runs ~13 min.
- `MEMORY_LLM_API_KEY` / `MEMORY_LLM_API_BASE` / `MEMORY_LLM_MODEL` in your
  environment for development runs (selfcheck.py reads them).

## What You Submit

Leave your best memory system under `/app/methods/` with the entry point
`/app/methods/main/memory_system.py` honoring the contract. There is no
submit step; it is graded once at the end.

## How It Is Judged

The sealed verifier ingests unseen conversations of the same kind
into one instance of your system, asks every question, and scores
token-level F1 against concise gold answers. Higher mean F1 is better.
Grading re-runs your full pipeline (ingest + answer) within a fixed
wall-clock budget — the inherited baseline uses about half of it, and a run
that exceeds it scores 0.
