"""Answer generation from retrieved memory context — concise free-text QA
(token-F1 friendly)."""

from __future__ import annotations

import json
import re

ANSWER_STANDARD = """Answer the question based on the provided context.

Question: {question}

Context:
{context}

Rules:
1. Answer in 1-10 words when possible. Shorter and more specific is ALWAYS better.
   - "how many" -> just the number
   - "where" -> the SPECIFIC place name (e.g., "Sweden", NOT "her home country")
   - "when" -> the date/time (e.g., "15 March 2023" or "Since 2016")
   - "who" -> the person's name
   - "what" -> the specific thing
   - "yes/no" or "would" -> "Yes"/"No"/"Likely yes/no" with brief reason
2. Use EXACT words from context. Be SPECIFIC, not vague.
3. Do NOT paraphrase into vaguer language.
4. For inference questions ("Would X likely..."), answer "Likely yes/no" with brief reason.

Return JSON: {{"reasoning": "brief thought", "answer": "concise answer"}}
Return ONLY JSON."""


def parse_answer(response: str) -> str:
    """Extract the answer field from an LLM JSON response."""
    if not response:
        return "unknown"
    try:
        m = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            ans = str(obj.get("answer", "")).strip().strip("\"'")
            if ans and ans.lower() not in (
                "i don't know", "not mentioned", "not specified", "unknown", "n/a"
            ):
                return clean_answer(ans)
    except (json.JSONDecodeError, AttributeError):
        pass
    m = re.search(r'"answer"\s*:\s*"([^"]*)"', response)
    if m:
        return clean_answer(m.group(1))
    return response.split("\n")[0].strip()[:60] if response else "unknown"


def clean_answer(ans: str) -> str:
    """Strip common verbose prefixes/suffixes that hurt token F1."""
    for prefix in (
        "the answer is ", "based on the context, ",
        "according to the context, ", "the answer would be ",
    ):
        if ans.lower().startswith(prefix):
            ans = ans[len(prefix):]
    return ans.strip(" *_`\"'")


def answer_freetext(llm_call, question: str, context: str) -> str:
    msgs = [
        {"role": "system",
         "content": "Professional Q&A assistant. Concise answers. JSON output only."},
        {"role": "user",
         "content": ANSWER_STANDARD.format(question=question, context=context)},
    ]
    return parse_answer(llm_call(msgs, 512, 0.0))
