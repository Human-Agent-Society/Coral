FROM ghcr.io/leanprover-community/mathlib4/lean@sha256:e1f1eca5fdb9bd74ef4c05998becec18eccb7db042d14bce75d1d396f0d710d5

USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git libseccomp2 python3 python3-pip python3-venv zstd \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir \
      "annotated-types==0.7.0" "anyio==4.14.2" "certifi==2026.7.22" \
      "distro==1.9.0" "h11==0.16.0" "httpcore==1.0.9" "httpx==0.28.1" \
      "idna==3.18" "jiter==0.16.0" "openai==1.109.1" \
      "pydantic==2.13.4" "pydantic-core==2.46.4" "pyseccomp==0.1.2" \
      "sniffio==1.3.1" "tqdm==4.69.0" "typing-extensions==4.16.0" \
      "typing-inspection==0.4.2"
ENV ELAN_HOME=/home/lean/.elan

# Frozen official mathlib v4.9.0. Its cache is available and its Lean version
# matches the proof model's documented Lean 4.9 target.
RUN mkdir -p /opt/mathlib \
    && git -C /opt/mathlib init \
    && git -C /opt/mathlib remote add origin https://github.com/leanprover-community/mathlib4.git \
    && git -C /opt/mathlib fetch --depth 1 origin f0957a7575317490107578ebaee9efaf8e62a4ab \
    && git -C /opt/mathlib checkout --detach FETCH_HEAD \
    && cd /opt/mathlib \
    && /home/lean/.elan/bin/lake exe cache get \
    && /home/lean/.elan/bin/lake env printenv LEAN_PATH > /opt/lean.path \
    && git -C /opt/mathlib rev-parse HEAD > /opt/mathlib.commit

ENV PATH="/opt/venv/bin:/home/lean/.elan/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV MATHLIB_DIR=/opt/mathlib
ENV LEAN_REQUIRE_SECCOMP=1
ENV OMP_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8
WORKDIR /app
COPY data/ /app/data/
COPY methods/ /app/methods/
COPY child_runner.py /app/child_runner.py
COPY evaluator.py /app/evaluator.py
COPY lean_compiler.py /app/lean_compiler.py
COPY model_proxy.py /app/model_proxy.py
COPY selfcheck.py /app/selfcheck.py
RUN chmod -R a+rX /app /opt/mathlib \
    && chmod -R u+w /app/methods

# The pinned mathlib image inherits `ENTRYPOINT ["/bin/bash", "-l"]`.
# Harbor supplies its own keepalive command, so clear that inherited entrypoint.
ENTRYPOINT []
CMD ["sh", "-c", "sleep infinity"]
