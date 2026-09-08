# Agent image: CPU-only TriFinger offline-RL sim/dataset stack + the editable methods/main policy +
# the visible mixed-dataset download + the free self-check. Pins match the verifier image exactly (the
# submitted solver.py is re-run inside the verifier).
# The base image is pinned by DIGEST, not by the floating `python:3.12-slim`
# tag. Both Dockerfiles of this task carry the SAME digest. Resolved 2026-08-01 from the
# image these anchors were re-measured against; a tag can be re-pointed upstream at any
# time, and this task's anchors are replayed action traces from a pinned wheel stack sitting
# on top of that base -- a silent base bump is exactly the kind of thing that moves them.
FROM python@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
WORKDIR /app

# pybullet (a trifinger_simulation dependency) ships no cp312 manylinux wheel and builds from sdist;
# opencv-python (pulled in for image_obs, unused here) needs libGL/X11 shared libs at import time.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# every version is pinned EXACTLY and BOTH Dockerfiles (environment/ + tests/)
# carry byte-identical pip lines. The anchors are measured against these wheels; `torch` unpinned and
# `zarr<3` as a range meant the image resolved to whatever was current on build day, so the anchors
# were not reproducible from the repo. Versions read off the built images with
# `docker run --rm <img> pip list --format=freeze` (2026-07-28).
RUN pip install --no-cache-dir "torch==2.13.0+cpu" --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
      "trifinger_rl_datasets==1.0.3" \
      "trifinger_simulation==2.0.0" \
      "gymnasium==0.29.1" \
      "zarr==2.18.7" \
      "numcodecs==0.15.1" \
      "numpy==2.5.1" \
      "scipy==1.18.0" \
      "pybullet==3.2.7" \
      "opencv-python==5.0.0.93" \
      "numpy-quaternion==2024.0.13"

# pin the BLAS/OpenMP pools to 1 core. NOT a speed knob — it changes the SCORE:
# same checkpoint + same sealed batch gives mean return 460.9641695477112 at threads in {1,4,8} and
# 447.4884114018423 at threads = host cpu_count 96 (= 0.0246 reward). torch reads the HOST core count,
# not the cgroup quota, so an unpinned image grades differently on different machines. Identical ENV
# block in the other Dockerfile so the agent's selfcheck and the verifier are isomorphic.
ENV OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4

# Shared seed/env/run/replay/reward helpers (identical copy in tests/).
COPY trifinger_score.py  /app/trifinger_score.py
# The editable ARTIFACT-EVAL policy (train + Policy). This dir is what gets graded.
COPY methods/             /app/methods/
# Free, unlimited local proxy the agent iterates against: python /app/selfcheck.py
COPY selfcheck.py         /app/selfcheck.py
# Download the visible offline-RL training dataset (trifinger-cube-push-sim-mixed-v0, ~850MB) into
# /app/data/trifinger_dataset. Needs network; the verifier build below never downloads anything.
COPY prepare_visible.py   /app/prepare_visible.py
RUN python /app/prepare_visible.py

# Where the agent's trained checkpoint lands (the verifier loads it via artifacts=["/app/submission"]).
RUN mkdir -p /app/submission/model
