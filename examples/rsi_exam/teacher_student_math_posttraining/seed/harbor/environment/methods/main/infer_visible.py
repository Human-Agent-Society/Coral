#!/opt/venvs/vllm/bin/python
"""Static-batch Qwen-recommended sampling for the visible score-only evaluator."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from common import extract_integer  # noqa: E402

EVAL_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}.\n\nProblem:\n"
SEED_BASE = 20260804
MAX_OUTPUT_TOKENS = 30720


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/app/results/dev_predictions.json"))
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path("/app/data/dev.jsonl").open()]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": EVAL_PROMPT + str(row["problem"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        for row in rows
    ]
    sampling = [
        SamplingParams(
            temperature=0.6, top_p=0.95, top_k=20, min_p=0,
            max_tokens=MAX_OUTPUT_TOKENS, seed=SEED_BASE + index,
        )
        for index in range(len(rows))
    ]
    engine = LLM(
        model=str(args.model), tokenizer=str(args.model), trust_remote_code=False,
        dtype="bfloat16", tensor_parallel_size=1, max_model_len=32768,
        gpu_memory_utilization=0.90, max_num_seqs=8, seed=0,
    )
    outputs = engine.generate(prompts, sampling)
    predictions: dict[str, int] = {}
    for row, output in zip(rows, outputs):
        parsed = extract_integer(output.outputs[0].text)
        predictions[str(row["id"])] = parsed if parsed is not None and 0 <= parsed <= 999 else -1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(predictions, sort_keys=True) + "\n")
    print(json.dumps({"predictions": len(predictions), "output": str(args.output)}))


if __name__ == "__main__":
    main()
