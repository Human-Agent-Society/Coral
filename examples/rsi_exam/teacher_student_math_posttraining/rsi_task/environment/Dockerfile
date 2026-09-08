# vLLM stable v0.26.0, immutable linux/amd64 image manifest.
FROM vllm/vllm-openai:v0.26.0@sha256:770fe65b2c73ee74a5c42165cf3433de4048cc2cd9c57a937ca4e35aba5aa87b

USER root
ENTRYPOINT []
ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/tmp/build-hf \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    VLLM_WORKER_MULTIPROC_METHOD=spawn
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates curl build-essential procps psmisc jq \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m pip install --no-cache-dir "uv==0.11.2"

# Deliberately separate environments. Training can neither import nor perturb vLLM's stack.
COPY requirements.vllm.lock /tmp/requirements.vllm.lock
COPY requirements.train.lock /tmp/requirements.train.lock
RUN uv venv --python python3 /opt/venvs/vllm \
 && uv pip sync --python /opt/venvs/vllm/bin/python --torch-backend cu130 /tmp/requirements.vllm.lock \
 && uv venv --python python3 /opt/venvs/train \
 && uv pip sync --python /opt/venvs/train/bin/python --torch-backend cu130 /tmp/requirements.train.lock \
 && /opt/venvs/vllm/bin/python -c 'import torch,vllm,transformers; assert vllm.__version__=="0.26.0"; assert torch.__version__.startswith("2.11.0"); assert transformers.__version__=="5.14.1"' \
 && /opt/venvs/train/bin/python -c 'import torch,transformers; assert torch.__version__.startswith("2.11.0"); assert transformers.__version__=="5.8.0"'

# Current LLaMA-Factory main at task freeze time. Remove VCS metadata after proving the commit.
RUN git clone https://github.com/hiyouga/LLaMA-Factory.git /opt/LLaMA-Factory \
 && git -C /opt/LLaMA-Factory checkout 713b5a3f959d0c647d46f7ab76cbb8fde8155586 \
 && test "$(git -C /opt/LLaMA-Factory rev-parse HEAD)" = "713b5a3f959d0c647d46f7ab76cbb8fde8155586" \
 && uv pip install --python /opt/venvs/train/bin/python --no-deps -e /opt/LLaMA-Factory \
 && rm -rf /opt/LLaMA-Factory/.git

# Build-time network only: immutable model/data revisions are downloaded, sanitized, then all
# source dataset snapshots and HF caches are deleted.
COPY prepare_assets.py /tmp/prepare_assets.py
RUN /opt/venvs/train/bin/python /tmp/prepare_assets.py --role environment \
 && rm -rf /tmp/prepare_assets.py /tmp/build-hf /root/.cache/huggingface /root/.cache/uv /tmp/* \
 && test "$(find /app/data -maxdepth 1 -type f | wc -l)" = 3 \
 && ! grep -R '"solution"' /app/data

COPY methods/main/ /app/methods/main/
COPY visible_client.py /app/visible_client.py
COPY selfcheck.py /app/selfcheck.py
RUN mkdir -p /app/submission /app/results /app/run \
 && touch /app/experiment_log.md \
 && chmod -R a+rX /app/models /app/data /opt/LLaMA-Factory \
 && chmod -R a+rwX /app/methods /app/submission /app/results /app/run \
 && chmod a+rw /app/experiment_log.md \
 && chmod +x /app/methods/main/*.sh

ENV PATH=/opt/venvs/train/bin:/opt/venvs/vllm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HF_HOME=/app/.hf-disabled \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    WANDB_MODE=offline \
    TOKENIZERS_PARALLELISM=false
CMD ["bash", "-lc", "sleep infinity"]
