# Verifier image: FROMs the prebuilt OpenROAD base (see openroad-base/Dockerfile)
# so it builds in SECONDS. Ships the SEALED held-out cases + the trusted two-phase
# grader + the OpenROAD scorer. Held-out lands ONLY here, never in the agent image.
#
#   base image: pulled from ghcr.io/xinyuyee/arena-gate-sizing-openroad:cde1403 (public GHCR).
#   To rebuild locally instead: docker build -t ghcr.io/xinyuyee/arena-gate-sizing-openroad:cde1403 openroad-base/
FROM ghcr.io/xinyuyee/arena-gate-sizing-openroad:cde1403
ENV DEBIAN_FRONTEND=noninteractive
# thread count is a constant equal to [verifier] cpus, so the score does not
# depend on how many cores the host has. Keep in step with environment/Dockerfile
# and with grade.py's CHILD_ENV.
ENV OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
WORKDIR /tests

# --- sealed held-out cases: fetch public source @ pinned commit + heldout split
#     via the SAME anonymize/split script the agent image uses ---
ARG SRC_REPO=https://github.com/ASU-VDA-Lab/2024_ICCAD_Contest_Gate_Sizing_Benchmark.git
ARG SRC_COMMIT=2037507693b56fc182de81cdc821706e0551971f
COPY prepare_data.py /tmp/prepare_data.py
RUN git init -q /tmp/src && cd /tmp/src \
    && git remote add origin "$SRC_REPO" \
    && git fetch -q --depth 1 origin "$SRC_COMMIT" \
    && git checkout -q FETCH_HEAD \
    && python3 /tmp/prepare_data.py --src /tmp/src --out /tests/heldout --split heldout \
    && rm -rf /tmp/src /tmp/prepare_data.py

# Trusted grader stack (parent never imports the submission).
COPY grade.py       /tests/grade.py
# sealed anchor table, root-owned and unreadable by the dropped solver uid.
COPY anchors.json   /tests/anchors.json
RUN chown root:root /tests/anchors.json && chmod 400 /tests/anchors.json
# Defence in depth for the same hole the escape test found: the repo ships these
# 0664, so a child that still held group 0 could rewrite the grader itself. Even with
# setgroups([]) in grade.py, nothing here should be writable after the image is built.
RUN chmod -R go-w /tests && chmod 0400 /tests/anchors.json
COPY child_solve.py /tests/child_solve.py
COPY score.py       /tests/score.py
COPY test.sh        /tests/test.sh
COPY tools/         /tests/tools/
COPY eval/          /tests/eval/
