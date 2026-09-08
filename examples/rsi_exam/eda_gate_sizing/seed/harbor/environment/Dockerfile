# Agent image: bakes in the VISIBLE cases only (held-out lives solely in the
# verifier image). FROMs the prebuilt OpenROAD base (see openroad-base/Dockerfile)
# so this image builds in SECONDS — just data download + copies. The base carries
# OpenROAD (Python bindings) for the visible self-check score + python deps.
#
#   base image: pulled from ghcr.io/xinyuyee/arena-gate-sizing-openroad:cde1403 (public GHCR).
#   To rebuild locally instead: docker build -t ghcr.io/xinyuyee/arena-gate-sizing-openroad:cde1403 openroad-base/
FROM ghcr.io/xinyuyee/arena-gate-sizing-openroad:cde1403
ENV DEBIAN_FRONTEND=noninteractive
# same thread constant as the verifier, so the agent's self-check score and
# the graded score are computed under identical conditions.
ENV OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
WORKDIR /app

# --- fetch the public benchmark data at a pinned commit ---
# Only the data tree is fetched; the OpenROAD / CircuitOps git submodules are
# gitlinks and are NOT pulled, so this stays light.
ARG SRC_REPO=https://github.com/ASU-VDA-Lab/2024_ICCAD_Contest_Gate_Sizing_Benchmark.git
ARG SRC_COMMIT=2037507693b56fc182de81cdc821706e0551971f
COPY prepare_data.py /tmp/prepare_data.py
RUN git init -q /tmp/src && cd /tmp/src \
    && git remote add origin "$SRC_REPO" \
    && git fetch -q --depth 1 origin "$SRC_COMMIT" \
    && git checkout -q FETCH_HEAD \
    && python3 /tmp/prepare_data.py --src /tmp/src --out /app/data --split visible \
    && rm -rf /tmp/src /tmp/prepare_data.py

# --- editable baseline + self-check surface (legality + real OpenROAD score) ---
# the dropped solver uid in the verifier must be able to read this tree.
COPY methods /app/methods
RUN chmod -R a+rX /app/methods
COPY tools /app/tools
COPY eval /app/eval
COPY selfcheck.py /app/selfcheck.py
