# Local research surface

Everything in this directory is editable. Fixed models and sanitized corpus assets are read-only.

## Isolated stacks

```text
/opt/venvs/vllm  : vLLM 0.26.0, torch 2.11.0+cu130, transformers 5.14.1
/opt/venvs/train : LLaMA-Factory 0.9.6.dev0 at commit 713b5a3f..., torch 2.11.0+cu130,
                   transformers 5.8.0
```

The model you're wroking to improve is `Qwen/Qwen3-1.7B-Base`. The Base model has an immutable 32,768-token context.

## Starter baseline

`./train.sh` trains answer-only LoRA on all 40,315 task-provided answers and merges complete weights into `/app/submission`. This is the starting point; the graded metric is exact integer accuracy on sealed problems.

## Tools

- `merge.py` — merge a LoRA adapter into the frozen student and write standalone weights.
- `eval_visible.py` — submit a prediction vector to the aggregate-only development scorer.
- `teacher_control.sh` — start/stop the local teacher 8B LLM inference server in a dedicated process group. You are free to use it or not.
- `common.py` — shared integer-extraction and prompt utilities.

## Evaluation

```bash
python /app/methods/main/eval_visible.py /app/submission
```

The fixed evaluator uses static sampled decoding:

- temperature 0.6;
- top-p 0.95;
- top-k 20;
- min-p 0;
- maximum output 30,720 tokens;
- immutable Base-model context 32,768 tokens;
- per-question seed `20260804 + index`.

## Safe process handling

`teacher_control.sh` starts the fixed 8B LLM in a dedicated process group, records and validates its exact PID, and stops only that group. Do not use `pkill`, `killall`, GPU-index killing, or `fuser`.
