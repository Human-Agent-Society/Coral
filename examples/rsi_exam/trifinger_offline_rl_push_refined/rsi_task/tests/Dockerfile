# Verifier image: same CPU-only TriFinger sim/eval stack as the agent image (pinned so the replayed
# traces reproduce exactly) plus the two-phase grader and the sealed held-out seeds. No dataset is
# downloaded here — a policy rollout only needs the simulator, whose assets ship inside the
# trifinger_simulation / robot_properties_fingers wheels, so this image (and the verifier container at
# run time) needs no network at all.
# The base image is pinned by DIGEST, not by the floating `python:3.12-slim`
# tag. Both Dockerfiles of this task carry the SAME digest. Resolved 2026-08-01 from the
# image these anchors were re-measured against; a tag can be re-pointed upstream at any
# time, and this task's anchors are replayed action traces from a pinned wheel stack sitting
# on top of that base -- a silent base bump is exactly the kind of thing that moves them.
FROM python@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
WORKDIR /tests

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

# Dedicated unprivileged account the untrusted solver child is dropped to (grade.py SOLVER_UID/GID).
# No home, no shell: it exists only as a uid that cannot write /tests or /logs and cannot signal root.
RUN groupadd -g 4242 solver && useradd -u 4242 -g 4242 -M -s /usr/sbin/nologin solver

COPY trifinger_score.py  /tests/trifinger_score.py
COPY heldout_seeds.py    /tests/heldout_seeds.py
COPY anchors.json        /tests/anchors.json
COPY grade.py             /tests/grade.py
COPY child_solve.py       /tests/child_solve.py
COPY replay_score.py      /tests/replay_score.py
COPY test.sh              /tests/test.sh

# /tests is root-owned and unwritable by uid 4242, so the child can no longer
# overwrite the scorer the trusted parent executes afterwards; the sealed files go 0400, which the
# root parent reads regardless of mode and the child cannot open at all.
#   sealed: anchors.json (the 4 reward anchors), heldout_seeds.py (which episodes are graded),
#           replay_score.py (the trusted scorer itself)
#   readable: trifinger_score.py + child_solve.py — the child has to import/run these, and an
#           identical copy of trifinger_score.py already ships in the agent image anyway.
RUN chown -R root:root /tests \
 && chmod 0755 /tests \
 && chmod 0644 /tests/*.py \
 && chmod 0755 /tests/test.sh \
 && chmod 0400 /tests/anchors.json /tests/heldout_seeds.py /tests/replay_score.py \
 && rm -rf /tests/__pycache__
