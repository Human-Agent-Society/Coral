FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
LABEL ai.autoresearch.task_version="flashattention-noncausal-global-d160-extreme-gqa-full-vjp-v21" \
      ai.autoresearch.verifier_lifecycle_revision="cpu-exact-snapshot-r2" \
      ai.autoresearch.selfcheck_sha256="108a738152e6eac58051a2da6d6848bca6c9e31e715c675f1a433dc2001425e6"
WORKDIR /app
RUN pip install --no-cache-dir "numpy<2"
COPY protocol.py selfcheck.py submission_guard.py /app/
COPY problems/ /app/problems/
COPY methods/ /app/methods/
RUN ln -sf /app/methods/experiment_log.md /app/experiment_log.md
