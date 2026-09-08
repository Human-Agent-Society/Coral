#!/usr/bin/env bash
# Weak reproducible baseline: answer-only LoRA SFT, then merge complete weights.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
ROOT=/app/methods/main
mkdir -p /app/results
/opt/venvs/train/bin/python "$ROOT/build_answer_sft.py"
/opt/venvs/train/bin/llamafactory-cli train "$ROOT/sft_answer.yaml" 2>&1 | tee /app/results/answer-sft.log
/opt/venvs/train/bin/python "$ROOT/merge.py" --adapter /app/results/answer-sft-adapter --output /app/submission
/opt/venvs/train/bin/python /app/selfcheck.py /app/submission
cat >>/app/experiment_log.md <<'EOF'

## Answer-only baseline
Ran the fixed starter answer-only LoRA SFT configuration and merged the adapter into the student.
EOF
