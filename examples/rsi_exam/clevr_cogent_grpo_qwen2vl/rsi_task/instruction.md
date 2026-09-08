# Train Qwen2-VL-2B with GRPO so it generalizes off the training distribution

You inherit a deliberately weak reinforcement-learning (GRPO) training project for the
vision-language model `Qwen2-VL-2B-Instruct`: the starting point in `/app/methods/main/` trains on
the provided counting-VQA split with a trivial reward and conservative hyper-parameters, and barely
beats the untrained model. You submit a **trained model** (full merged weights in
`/app/submission/`), *not* code, and a sealed verifier scores it on two hidden test sets that are
**not** in-distribution samples of your train split — so only real generalization moves the score.

## Hard Constraints

1. The submitted model **must be `Qwen2-VL-2B-Instruct`** (same architecture/tokenizer/processor),
   fine-tuned. Write the **full merged** model (config + weights + tokenizer + processor) to
   `/app/submission/` with `save_pretrained(...)` so the verifier can load it with
   `Qwen2VLForConditionalGeneration.from_pretrained("/app/submission")`. The trainer does full
   fine-tuning by default, so the saved checkpoint already is the full model; `train.sh` copies it
   to `/app/submission`. If you switch to a LoRA adapter, **merge it** before saving.
2. **No internet at training time.** The base model and the train data are already on disk / in the
   HF cache. You may not download anything, and in particular you may not fetch any test data.
3. **Train only on the provided train split.** Do not hand-write or memorize answer tables, do not
   key behavior on the visible val ids.
4. The verifier uses a **fixed evaluation protocol** (prompt template + greedy decoding, below).
   Train your model to that protocol — there is no separate prompt you get to submit.
5. Keep `per_device_train_batch_size 1` — batched training in this trainer is buggy.
6. **Everything must finish inside the wall-clock budget with the model saved.** GRPO is slow
   (multiple generations per prompt); a run that never writes `/app/submission/` scores nothing.
7. There is no submit step and no per-attempt feedback. Iterate against your local self-check, then
   leave your best model in `/app/submission/`; it is graded once at the end on the hidden sets.

## What You Have

**Budget: 2 GPUs (48 GB each), 8h wall-clock** (`accelerate` / `torchrun` / `deepspeed` are
installed). That is meant for **several rounds**, not one long run — a LoRA GRPO round is
~40–60 min.

- **Base model** (`/app/models/Qwen2-VL-2B-Instruct/`): the untrained instruct model to start from.
  This exact model, with no fine-tuning, is the floor — you must beat it.
- **Train data** (`/app/data/train/`): ~68k counting-VQA examples (columns `image` (PIL),
  `problem`, `solution`), baked to disk, with the validation set removed. The baseline `train.sh`
  points `--dataset_name /app/data/train` at it (loaded offline). Train only on this.
- **In-distribution val** (`/app/data/val/`): 2k examples **held out from training**, same
  distribution as train. A training-health check — is the model actually learning to count, is it
  under-/over-training. Do **not** train on it.
- **OOD val** (`/app/data/ood_val/`): ~1k examples that are **out-of-distribution** relative to
  train, and **visible to you**. It is disjoint from the sealed test data and tracks the sealed
  out-of-distribution test. Do **not** train on it.
- **Editable baseline** (`/app/methods/main/`): a GRPO trainer (`open_r1/grpo.py` +
  `Qwen2VLGRPOTrainer`) launched by `train.sh`, plus `zero3.json`. Two surfaces are yours: the
  **reward functions and their registry** in `open_r1/grpo.py` (new ones can be registered and
  referenced by name in `--reward_funcs`), and the **training hyper-parameters** in `train.sh`
  (optimization, GRPO group size, batching, step count, KL, vision-token budget, sequence lengths,
  and optionally LoRA via `--use_peft --lora_r …`). See `methods/main/README.md`. Running
  `bash /app/methods/main/train.sh` trains and writes the model to `/app/submission/`.
  Nothing here prescribes *which* changes work. Finding that out is the task.
- **Self-check** (`/app/selfcheck.py`): runs the **exact same eval as the verifier** (same prompt,
  extraction, matching) on **both** val sets against whatever model is in `/app/submission/`. Free
  and unlimited.

## What You Submit

A directory `/app/submission/` that `Qwen2VLForConditionalGeneration.from_pretrained` can load
(full merged Qwen2-VL-2B weights + tokenizer + processor). That directory is the graded artifact.

## How It Is Judged

The verifier loads `/app/submission/` in a clean, network-isolated box and runs the protocol below
on two hidden test sets (5k examples each). One is a **compositional-generalization** split — the
same visual domain, but attribute combinations that never co-occur in training; the other comes
from a **different distribution** entirely (different renderer/scene statistics).

For every test example the model is prompted with the image and:

```text
{question} First output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.
```

> Note: the trainer's *training* template says "Output …" while the eval template says
> "First output …". The grader uses the EVAL phrasing above. Train so your model is robust to it.

The model is generated **greedily** (`do_sample=False`, `max_new_tokens=256`). The grader extracts
the **first integer** inside `<answer>…</answer>` (regex `<answer>\s*(\d+)\s*</answer>`) and counts
the example **correct** only on **exact integer equality** with the gold count — no fuzzy matching
and no fallback, so a model that stops emitting the `<answer>` tag is parsed wrong everywhere.

```text
accuracy_pct = 100 * (correct / total)            # computed per test set
score = mean(accuracy_pct over the two hidden test sets)
```

Higher mean accuracy is better. A model that fails to load, is the wrong architecture, or produces
unparseable output everywhere scores 0.
