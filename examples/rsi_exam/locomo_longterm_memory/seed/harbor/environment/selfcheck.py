"""Visible self-check: score the current submission on the dev set (token F1).

Free to run as often as you like. Typical loop:

    # cheap smoke run (1 conversation, 60 questions)
    python3 selfcheck.py --samples 0 --max-qa 60

    # full visible set (4 conversations, 812 QA; ~70 min cold with the
    # baseline, ~13 min when extraction is cached)
    python3 selfcheck.py --samples 0,1,2,3 --cache-dir /app/extract_cache

Extraction is the expensive part; pass --cache-dir to reuse extracted
memories across runs (invalidate it yourself if you change extraction).
Results land in /app/selfcheck_results/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path

# A-only: the backend has no AAAA record and the dual-stack lookup fails behind the
# egress sidecar, silently degrading every call into the caller's fallback.
_getaddrinfo = socket.getaddrinfo


def _getaddrinfo_v4(host, port, family=0, *args, **kwargs):
    if family in (0, socket.AF_UNSPEC):
        family = socket.AF_INET
    return _getaddrinfo(host, port, family, *args, **kwargs)


if os.environ.get("MEMORY_LLM_DUAL_STACK", "") != "1":
    socket.getaddrinfo = _getaddrinfo_v4

sys.path.insert(0, "/app")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval.dataset import load_conversations, merge_samples
from eval.scoring import token_f1

METHODS_DIR = os.environ.get("METHODS_DIR", "/app/methods/main")
DATA = os.environ.get("VISIBLE_DATA_PATH", "/app/data/conversations_visible.json")
OUT_DIR = os.environ.get("SELFCHECK_OUT", "/app/selfcheck_results")


def make_llm_call():
    from openai import OpenAI
    api_key = os.environ.get("MEMORY_LLM_API_KEY",
                             os.environ.get("OPENAI_API_KEY", ""))
    if not api_key:
        raise SystemExit("MEMORY_LLM_API_KEY is not set")
    base_url = os.environ.get("MEMORY_LLM_API_BASE",
                              os.environ.get("OPENAI_API_BASE",
                                             "https://api.openai.com/v1"))
    model = os.environ.get("MEMORY_LLM_MODEL", "gpt-4o-mini")
    client = OpenAI(base_url=base_url, api_key=api_key)

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


def load_submission(llm_call):
    spec = importlib.util.spec_from_file_location(
        "memory_system", os.path.join(METHODS_DIR, "memory_system.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, METHODS_DIR)
    spec.loader.exec_module(mod)
    return mod.build_memory_system(llm_call)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="0",
                   help="comma-separated visible conversation indices (0-3)")
    p.add_argument("--max-qa", type=int, default=None,
                   help="cap QA per conversation for cheap runs")
    p.add_argument("--cache-dir", default=None,
                   help="reuse extracted memories across runs (via MEMSYS_CACHE_DIR)")
    args = p.parse_args()

    llm_call = make_llm_call()
    if args.cache_dir:
        os.environ["MEMSYS_CACHE_DIR"] = args.cache_dir

    indices = [int(x) for x in args.samples.split(",")]
    samples = load_conversations(DATA, sample_indices=indices,
                                 max_qa_per_sample=args.max_qa)
    sessions, qa_pairs = merge_samples(samples)
    print(f"visible dev set: {len(samples)} conversations | {len(sessions)} "
          f"sessions | {len(qa_pairs)} QA")

    ms = load_submission(llm_call)
    t0 = time.time()
    ms.ingest(sessions)
    print(f"ingest done in {time.time() - t0:.0f}s")

    os.makedirs(OUT_DIR, exist_ok=True)
    raw_path = os.path.join(OUT_DIR, f"visible_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    per_cat = defaultdict(list)
    scores = []
    with open(raw_path, "w") as f:
        for qi, qa in enumerate(qa_pairs):
            pred = ms.answer(qa["question"])
            s = token_f1(pred, qa["answer"])
            scores.append(s)
            per_cat[qa["category"]].append(s)
            f.write(json.dumps({"q": qa["question"], "pred": pred,
                                "gold": qa["answer"], "cat": qa["category"],
                                "f1": round(s, 4)}) + "\n")
            if (qi + 1) % 50 == 0:
                print(f"  [{qi + 1}/{len(qa_pairs)}] "
                      f"F1={sum(scores) / len(scores):.4f}")

    overall = sum(scores) / len(scores) if scores else 0.0
    print(f"\nOverall token-F1: {overall:.4f}  ({len(scores)} QA)")
    for cat in sorted(per_cat):
        v = per_cat[cat]
        print(f"  cat {cat}: {sum(v) / len(v):.4f}  (n={len(v)})")
    print(f"raw results: {raw_path}")
    print("\nReminder: grading runs your system on UNSEEN conversations of "
          "the same kind; this score is on the visible ones.")

    if Path("/app/budget.py").exists():
        import subprocess
        subprocess.run([sys.executable, "/app/budget.py"], check=False)


if __name__ == "__main__":
    main()
