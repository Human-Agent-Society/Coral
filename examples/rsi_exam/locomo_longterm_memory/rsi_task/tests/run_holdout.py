"""Untrusted child: drive the submitted memory system over the stripped holdout.

Receives a questions file with sessions + questions (no answers), imports the
agent's memory_system.py, ingests everything once, answers every question, and
writes predictions JSON. All LLM traffic goes through the parent's
pinned-model proxy (OPENAI_API_BASE points at it; no real key here).

Usage: python3 run_holdout.py <questions.json> <predictions.json>
"""

import importlib.util
import json
import os
import sys
import time

METHODS_DIR = "/app/methods/main"   # hardcoded: see grade.py


def make_llm_call():
    from openai import OpenAI
    client = OpenAI(
        base_url=os.environ["OPENAI_API_BASE"],
        api_key=os.environ.get("OPENAI_API_KEY", "proxy"),
    )
    model = os.environ["LLM_MODEL"]   # set by grade.py from the pinned constant

    def llm_call(messages, max_tokens=512, temperature=0.0):
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=model, messages=messages,
                    max_completion_tokens=max_tokens, temperature=temperature,
                )
                return (r.choices[0].message.content or "").strip()
            except Exception:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
        return ""

    return llm_call


def main():
    questions_path, predictions_path = sys.argv[1], sys.argv[2]
    with open(questions_path) as f:
        data = json.load(f)

    spec = importlib.util.spec_from_file_location(
        "memory_system", os.path.join(METHODS_DIR, "memory_system.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, METHODS_DIR)
    spec.loader.exec_module(mod)

    ms = mod.build_memory_system(make_llm_call())

    t0 = time.time()
    ms.ingest([tuple(s) for s in data["sessions"]])
    print(f"[child] ingest done in {time.time() - t0:.0f}s", flush=True)

    preds = {}
    for qi, q in enumerate(data["questions"]):
        qid = str(q["qid"])
        try:
            preds[qid] = str(ms.answer(q["question"],
                                       question_time=q.get("question_time", "")))
        except Exception as e:  # one bad question must not zero the run
            preds[qid] = f"__error__: {e}"
        if (qi + 1) % 100 == 0:
            with open(predictions_path, "w") as f:
                json.dump(preds, f)
            print(f"[child] answered {qi + 1}/{len(data['questions'])}", flush=True)
    with open(predictions_path, "w") as f:
        json.dump(preds, f)
    print(f"[child] answered {len(preds)} questions", flush=True)


if __name__ == "__main__":
    main()
