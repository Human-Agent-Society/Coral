# methods/main — the inherited, editable GRPO project

This is the half-finished project you improve: a GRPO trainer for Qwen2-VL, vendored here at
image-build time so you train on a known-good stack.

```
methods/main/
├── train.sh          # the launcher you run + tune (hyper-parameters). EDIT THIS.
├── open_r1/          # trainer source (grpo.py + trainer/). EDIT grpo.py's reward functions.
│   ├── grpo.py       #   reward_funcs_registry + the training QUESTION_TEMPLATE
│   └── trainer/      #   Qwen2VLGRPOTrainer (do not need to touch)
└── zero3.json        # DeepSpeed ZeRO-3 config
```

## The two surfaces (this is the task)

1. **Reward design** — `open_r1/grpo.py`. The baseline `train.sh` selects a single, deliberately
   weak reward term. What the policy gets paid for is yours to design; you may add new reward
   functions to `reward_funcs_registry` and reference them by name in `--reward_funcs`.
2. **Hyper-parameters** — `train.sh`: optimization, GRPO group size, KL, step count, vision-token
   budget, batch/grad-accum, and optionally LoRA (`--use_peft --lora_r ...`, but then merge before
   saving — see train.sh).

Which changes actually help is not specified anywhere. That is the research.

## Run

```bash
bash /app/methods/main/train.sh        # trains, then writes the merged full model to /app/submission
python /app/selfcheck.py               # score /app/submission with the exact grader protocol
```

## Notes / gotchas

- **Training vs eval prompt differ**: training uses `"... Output the thinking process ..."`; the
  grader/self-check use `"... First output the thinking process ..."`. Train so the model is robust
  to the eval phrasing.
- The grader extracts the **first integer** in `<answer>…</answer>` and requires **exact integer
  equality** with the gold count.
- Keep `per_device_train_batch_size 1` (batched training in this trainer is buggy); scale via
  `gradient_accumulation_steps` and `num_generations`.
- Reduce `num_generations` / `max_pixels` if you OOM.
