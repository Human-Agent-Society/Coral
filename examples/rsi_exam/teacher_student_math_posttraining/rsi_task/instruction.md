# Mathematical Post-Training

Improve the mathematical-reasoning accuracy of an inherited 1.7B LLM, with potentially useful a sanitized training corpus, using a fixed local 8B teacher LLM and a weak answer-only starter. Optimize exact integer accuracy; your submitted merged model will re-run on sealed hidden problems for scoring.

## Hard Constraints

- Hard agent budget: 8 CPUs, 2 × 48 GB GPUs, and 8 hours wall time.
- Training is offline. All model and corpus assets are local and immutable; do not attempt to recover benchmark identities or evaluation answers.
- Training phase must use PEFT LoRA adapters.
- The provided 1.7B and 8B models, corpus answers, and model generated text are permitted supervision sources. Do not introduce new hand-authored labels or data.
- The final artifact must be a standalone merged 1.7B LLM model. Preserve the frozen student architecture and tokenizer.
- Manage only processes you launch. Do not use broad `pkill`, `killall`, GPU-index killing, or `fuser` on a shared host.

## What You Have

- 1.7B model you need to improve: `/app/models/student` (`Qwen/Qwen3-1.7B-Base`, 1.7B parameters, pretraining checkpoint).
- 8B teacher model: `/app/models/teacher` (8B parameters).
- Training corpus: `/app/data/train.jsonl`, one JSON object per line with exactly `problem` and task-provided `answer` fields.
- Development questions: `/app/data/dev.jsonl`, with opaque IDs and no answers.
- Isolated vLLM environment: `/opt/venvs/vllm`.
- Isolated LLaMA-Factory/training environment: `/opt/venvs/train`.
- Editable starter methods and utilities: `/app/methods/main`.
- Weak answer-only LoRA baseline: `/app/methods/main/train.sh`.
- Method documentation and measured throughput guidance: `/app/methods/main/README.md`.
- You're free and limited to these provided offline resources with any methods you can implement to improve the 1.7B model's mathematical reasoning accuracy.
The original training-corpus solutions, source metadata, and download caches are not present.

## What You Submit

Create `/app/submission` containing complete `safetensors` weights, config, and the unchanged student tokenizer, loadable by `transformers` and vLLM while offline.

- Merge the selected LoRA adapter into the original student before submission; an adapter-only checkpoint is invalid.
- Do not include Python, custom model code, shell scripts, symlinks, external references, or teacher weights.
- Keep editable research code under `/app/methods` and write a concise experiment record to `/app/experiment_log.md`.
- Before finishing, run:

```bash
python /app/selfcheck.py /app/submission
```

## How It Is Judged

The final model is sampled on a disjoint sealed set of integer-answer mathematics problems using the fixed Qwen3 sampling protocol: temperature 0.6, top-p 0.95, top-k 20, min-p 0, and a 30,720-token output cap within the Base model's immutable 32,768-token context. Static batching and fixed per-question seeds make the sampled protocol reproducible. The metric is exact integer accuracy, and higher is better. The sealed verifier reveals no problem text, answers, predictions, or per-example feedback and never executes submitted code.

You may iterate on visible answer-free development questions with:

```bash
python /app/methods/main/eval_visible.py /app/submission
```

This score-only service has a small query budget and returns aggregate accuracy only.
