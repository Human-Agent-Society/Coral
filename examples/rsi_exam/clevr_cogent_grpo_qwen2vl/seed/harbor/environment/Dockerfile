# Agent image: a faithful reproduction of R1-V's training environment (the project this task is
# derived from), so GRPO on Qwen2-VL-2B runs on the known-good stack. Versions are pinned to
# R1-V's authoritative lockfile: github.com/StarsfieldAI/R1-V  src/requirements.txt
#   torch 2.5.1 / cu124, transformers @ git commit, trl 0.14.0, accelerate 1.3.0,
#   deepspeed 0.15.4, datasets 3.2.0, qwen-vl-utils 0.0.10, flash-attn 2.7.4.post1, vllm 0.7.2.
# devel (not runtime) image: nvcc is needed to compile flash-attn.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/.hf \
    PIP_NO_CACHE_DIR=1 \
    MAX_JOBS=8

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential ninja-build \
        libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# --- R1-V's exact training stack (pinned to src/requirements.txt) -----------------------------
# transformers from the same git commit R1-V locks (Qwen2-VL GRPO support depends on it).
RUN pip install "git+https://github.com/huggingface/transformers.git@92c5ca9dd70de3ade2af2eb835c96215cc50e815"
RUN pip install \
        "trl==0.14.0" \
        "accelerate==1.3.0" \
        "deepspeed==0.15.4" \
        "datasets==3.2.0" \
        "qwen-vl-utils==0.0.10" \
        "vllm==0.7.2" \
        "math-verify==0.5.2" \
        "latex2sympy2_extended==1.0.6" \
        "numpy==1.26.4" \
        "pillow==11.1.0" \
        "PyYAML==6.0.2" \
        "wandb==0.18.3" \
        "tensorboardX" \
        "tqdm"
# flash-attn needs torch already present + nvcc; build isolation off so it links this torch.
RUN pip install "flash-attn==2.7.4.post1" --no-build-isolation

# --- Vendor R1-V's GRPO trainer + source as the inherited, editable project -------------------
# Pinned commit. The agent edits the reward functions in methods/main/open_r1/grpo.py and the
# launch flags in methods/main/train.sh (this is the research surface).
RUN git clone https://github.com/StarsfieldAI/R1-V /opt/R1-V \
 && git -C /opt/R1-V checkout e35f97e566357631a2602accdd79610b752d8f0b
RUN mkdir -p /app/methods/main \
 && cp -r /opt/R1-V/src/r1-v/src/open_r1 /app/methods/main/open_r1 \
 && cp /opt/R1-V/src/r1-v/local_scripts/zero3.json /app/methods/main/zero3.json
# Task-fixed train/val split: make the trainer load a LOCAL save_to_disk dir (/app/data/train, val
# excluded) when --dataset_name is a directory, instead of the full HF dataset. (grpo.py already
# imports load_from_disk.) The train/val partition is set by the task, not the agent.
RUN python - <<'PY'
p = "/app/methods/main/open_r1/grpo.py"
s = open(p).read()
old = "    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)"
new = (
    "    import os as _os\n"
    "    from datasets import DatasetDict as _DatasetDict\n"
    "    if _os.path.isdir(script_args.dataset_name):\n"
    "        dataset = _DatasetDict({script_args.dataset_train_split: load_from_disk(script_args.dataset_name)})\n"
    "    else:\n"
    "        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)"
)
assert old in s, "grpo.py load_dataset line not found — R1-V pin changed?"
open(p, "w").write(s.replace(old, new))
print("[patch] grpo.py: load local train split when --dataset_name is a dir")
PY

# --- Bake the base model + CoGenT-A train split into the HF cache (build has network). ---
# We pre-download via the SAME identifiers R1-V uses (model path + dataset_name) so the trainer
# loads them offline at run time (HF_HUB_OFFLINE=1 in task.toml).
COPY prepare_data.py /tmp/prepare_data.py
RUN python /tmp/prepare_data.py --role environment && rm /tmp/prepare_data.py

# PEFT/LoRA support — the task offers `--use_peft --lora_r ...` as a tunable; without this the
# agent's LoRA route errors ("You need PEFT installed"). --no-deps so it can't perturb the pinned
# transformers/trl/accelerate stack (peft's deps are all already present).
RUN pip install --no-deps "peft==0.14.0"

# --- Editable launcher + docs + local self-check (same eval as the verifier). ---
COPY methods/main/train.sh   /app/methods/main/train.sh
COPY methods/main/README.md  /app/methods/main/README.md
COPY selfcheck.py            /app/selfcheck.py
COPY eval_common.py          /app/eval_common.py
RUN chmod +x /app/methods/main/train.sh && mkdir -p /app/submission

# --- Anonymization: drop the upstream checkout from the agent-visible filesystem. --------------
# Only src/open_r1 + zero3.json are needed at run time (already copied into /app/methods/main).
# The rest of the clone (README with the upstream's reported accuracies = our REFERENCE anchor,
# and src/eval/*.py which names the sealed test sets) must not be readable by the agent.
RUN rm -rf /opt/R1-V
# Same reason, three smaller traces:
#   * the upstream project name left in an unused trainer script,
#   * the source dataset id in save_to_disk's metadata (names the benchmark family, from which the
#     sealed compositional split is guessable) — only `features`/`splits` are load-bearing here,
#   * the hub-cache copy of that dataset: unused at run time (the trainer loads /app/data/* from
#     disk), so it is pure identity leak.
RUN sed -i 's/"tags": \["R1-V"\]/"tags": []/' /app/methods/main/open_r1/sft.py \
 && python - <<'PY' \
 && rm -rf /app/.hf/hub/datasets--leonardPKU--clevr_cogen_a_train /app/.hf/xet/logs
import json, pathlib
DROP = ("dataset_name", "config_name", "download_checksums", "homepage",
        "citation", "description", "license")
def strip(o):                      # the id also hides under splits.<name>.dataset_name
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in DROP}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o
for p in pathlib.Path("/app/data").rglob("dataset_info.json"):
    p.write_text(json.dumps(strip(json.loads(p.read_text()))))
    print("[anon]", p)
PY
