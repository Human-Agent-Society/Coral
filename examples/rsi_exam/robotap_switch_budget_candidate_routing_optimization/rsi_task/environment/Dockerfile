FROM node:22-bookworm-slim@sha256:53ada149d435c38b14476cb57e4a7da73c15595aba79bd6971b547ceb6d018bf AS codex-tooling

RUN npm install --global @openai/codex@0.142.5 && codex --version | grep -F '0.142.5'

FROM python:3.12-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b AS bwrap-guard-builder

COPY bwrap_guard.c /tmp/bwrap_guard.c
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && gcc -O2 -Wall -Wextra -Werror /tmp/bwrap_guard.c -o /tmp/bwrap-guard

FROM python:3.12-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bubblewrap \
        ca-certificates \
        git \
        passwd \
        ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir \
        numpy==1.26.4 \
        scipy==1.16.2 \
        joblib==1.5.3 \
        scikit-learn==1.7.2

COPY --from=codex-tooling /usr/local/bin/node /usr/local/bin/node
COPY --from=codex-tooling /usr/local/lib/node_modules/ /usr/local/lib/node_modules/
RUN ln -s ../lib/node_modules/@openai/codex/bin/codex.js /usr/local/bin/codex \
    && node --version \
    && codex --version | grep -F '0.142.5' \
    && rg --version

WORKDIR /app

COPY data/visible/ /app/data/visible/
COPY methods/ /app/methods/
COPY selfcheck.py /app/selfcheck.py
COPY experiment_log.md /app/experiment_log.md

RUN mkdir -p /usr/libexec/ara-routing \
    && mv /usr/bin/bwrap /usr/libexec/ara-routing/bwrap.real \
    && chown root:root /usr/libexec/ara-routing/bwrap.real \
    && chmod 0755 /usr/libexec/ara-routing/bwrap.real
COPY --from=bwrap-guard-builder /tmp/bwrap-guard /usr/bin/bwrap
COPY bwrap_guard_selfcheck.sh /tmp/bwrap_guard_selfcheck.sh
RUN chown root:root /usr/bin/bwrap /usr/libexec/ara-routing/bwrap.real \
    && chmod 04755 /usr/bin/bwrap \
    && chmod 0755 /usr/libexec/ara-routing/bwrap.real \
    && /bin/sh /tmp/bwrap_guard_selfcheck.sh \
    && rm -f /tmp/bwrap_guard_selfcheck.sh

RUN test -f /app/data/visible/index.json \
    && test -d /app/data/visible/inputs \
    && test -d /app/data/visible/labels \
    && python -m py_compile /app/selfcheck.py /app/methods/main/predict.py \
    && groupadd --gid 61124 ara-codex \
    && useradd --uid 61124 --gid 61124 --home-dir /home/ara-codex \
        --no-create-home --shell /usr/sbin/nologin ara-codex \
    && mkdir -p /home/ara-codex \
    && chown -R root:root /home/ara-codex \
    && chmod -R a-w /home/ara-codex \
    && chmod 0555 /home/ara-codex \
    && test -z "$(find /home/ara-codex -mindepth 1 -print -quit)" \
    && find / -xdev -type f -perm /6000 ! -path /usr/bin/bwrap -exec chmod a-s {} + \
    && chmod 04755 /usr/bin/bwrap \
    && chmod 0711 /root \
    && chown root:root /app \
    && chmod 0755 /app \
    && mkdir -p /app/.git /app/.agents /app/.codex \
    && chown root:root /app/.git /app/.agents /app/.codex \
    && chmod 0555 /app/.git /app/.agents /app/.codex \
    && test -z "$(find /app/.git /app/.agents /app/.codex -mindepth 1 -print -quit)" \
    && chown -R root:root /app/data /app/selfcheck.py \
    && chmod -R a-w /app/data /app/selfcheck.py \
    && chmod -R a+rX /app/data \
    && chmod 0555 /app/selfcheck.py \
    && chown -R 61124:61124 /app/methods \
    && chmod -R u+rwX,go+rX /app/methods \
    && chown 61124:61124 /app/experiment_log.md \
    && chmod 0644 /app/experiment_log.md

CMD ["bash"]
