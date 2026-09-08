#!/usr/bin/env bash
# Weak-baseline GRPO launcher for Qwen2-VL-2B on the provided counting-VQA train split, built on the
# vendored trainer (open_r1/grpo.py + Qwen2VLGRPOTrainer) in this directory.
#
# THIS IS THE EDITABLE RESEARCH SURFACE — two surfaces, both yours to design:
#   1. REWARD: the reward functions / registry in   open_r1/grpo.py
#   2. HYPER-PARAMS: the flags below (lr, num_generations, beta/KL, steps, max_pixels, LoRA, ...).
#
# The baseline is deliberately weak: a single trivial reward term + very few steps -> barely beats
# the untrained model (the reward-0 anchor). Run:  bash /app/methods/main/train.sh
#
# Flag notes (tune these):
#   --reward_funcs format   WEAK: one trivial term. Redesign what the policy is paid for.
#   --max_steps 50          WEAK: few steps. Increase to actually train.
#   --learning_rate 1e-6    tune.
#   --beta 0.04             KL coefficient — tune.
#   --num_generations 4     GRPO group size. MUST divide (NPROC x per_device_batch). Lower if OOM.
#   --per_device_train_batch_size 1   keep 1 (batched training in this trainer has a bug).
set -euo pipefail
cd /app/methods/main
export PYTHONPATH=/app/methods/main:${PYTHONPATH:-}
export WANDB_MODE=offline
export DEBUG_MODE=true
export LOG_PATH=/app/methods/main/debug_log.txt

MODEL=/app/models/Qwen2-VL-2B-Instruct
DATASET=/app/data/train                     # task-fixed train split (val held out); loaded offline
OUT=/app/_train                             # trainer scratch OUTSIDE /app/submission (the collect step rm -rf's /app/submission, so its source must not live under it)
NPROC=2                                      # number of GPUs to train on (match NVIDIA_VISIBLE_DEVICES; budget = 2 GPUs)

torchrun --nproc_per_node="${NPROC}" --nnodes=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=12345 \
    open_r1/grpo.py \
    --output_dir "${OUT}" \
    --model_name_or_path "${MODEL}" \
    --dataset_name "${DATASET}" \
    --deepspeed zero3.json \
    --reward_funcs format \
    --max_steps 50 \
    --learning_rate 1e-6 \
    --beta 0.04 \
    --num_generations 4 \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --max_pixels 401408 \
    --bf16 \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --logging_steps 1 \
    --save_steps 1000000 \
    --save_only_model true \
    --report_to none

# --- Collect the final model into /app/submission as a FULL, mergeable model (graded artifact). --
# Handles BOTH paths automatically: full fine-tune (copy the weights) and LoRA (merge the adapter
# onto the base, so the verifier can load a plain Qwen2-VL with from_pretrained).
LAST="$(ls -d "${OUT}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
[ -z "${LAST}" ] && LAST="${OUT}"
rm -rf /app/submission && mkdir -p /app/submission

if [ -f "${LAST}/adapter_config.json" ]; then
  echo "[train] LoRA adapter detected -> merging onto base"
  MODEL="${MODEL}" LAST="${LAST}" python - <<'PY'
import os, torch
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
base = Qwen2VLForConditionalGeneration.from_pretrained(os.environ["MODEL"], torch_dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(base, os.environ["LAST"]).merge_and_unload()
merged.save_pretrained("/app/submission", safe_serialization=True)
AutoProcessor.from_pretrained(os.environ["MODEL"]).save_pretrained("/app/submission")
print("[train] merged LoRA -> /app/submission")
PY
else
  echo "[train] full fine-tune -> copying weights"
  cp -r "${LAST}/." /app/submission/
  python -c "from transformers import AutoProcessor; AutoProcessor.from_pretrained('${MODEL}').save_pretrained('/app/submission')"
fi
echo "[train] final model -> /app/submission"
