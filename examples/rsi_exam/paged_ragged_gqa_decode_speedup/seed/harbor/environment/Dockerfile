FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel
LABEL ai.autoresearch.task-contract="paged-ragged-gqa-decode-d256-extreme-gqa-v25" \
      ai.autoresearch.timing-protocol="paged_ragged_gqa_decode_v25_d256_extreme_gqa_streamed_exact_snapshot"
RUN useradd --create-home --uid 10001 arena
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
WORKDIR /app
COPY protocol.py submission_guard.py restricted_runner.py selfcheck.py /app/
COPY protocol.py child_eval.py /runner/
COPY problems /app/problems
COPY methods /app/methods
RUN ln -s /app/methods/experiment_log.md /app/experiment_log.md
CMD ["python", "/app/selfcheck.py"]
