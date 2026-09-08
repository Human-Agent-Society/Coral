# Verifier image: inference-only subset of R1-V's stack, version-matched to the agent image so a
# model trained there loads here identically (same torch 2.5.1/cu124 + transformers git commit +
# qwen-vl-utils). No trl/deepspeed/flash-attn needed for grading (we load with attn=sdpa), so the
# smaller runtime image (no CUDA toolkit) is enough — torch 2.5.1/cu124 still matches the agent.
# Bakes the sealed held-out test sets (CoGenT-B + SuperCLEVR), which never enter the agent image.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/tests/.hf \
    PIP_NO_CACHE_DIR=1

WORKDIR /tests

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Same transformers commit as the agent (so the submitted Qwen2-VL weights load cleanly).
RUN pip install "git+https://github.com/huggingface/transformers.git@92c5ca9dd70de3ade2af2eb835c96215cc50e815"
RUN pip install \
        "accelerate==1.3.0" \
        "datasets==3.2.0" \
        "qwen-vl-utils==0.0.10" \
        "numpy==1.26.4" \
        "pillow==11.1.0" \
        "tqdm"

# Bake the base model + the two sealed held-out test sets (build has network).
COPY prepare_heldout.py /tmp/prepare_heldout.py
RUN python /tmp/prepare_heldout.py && rm /tmp/prepare_heldout.py

# Shared eval protocol (identical to the agent's) + grader + entrypoint.
COPY eval_common.py /tests/eval_common.py
COPY grade.py       /tests/grade.py
COPY test.sh        /tests/test.sh
COPY anchors.json /tests/anchors.json
RUN chown root:root /tests/anchors.json && chmod 400 /tests/anchors.json
