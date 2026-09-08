# Verifier image for finscope_dcf_valuation: the observable mirror the submission runs on, the
# sealed held-out truth + sealed fin-compute engine, and the grader.
# The base image is pinned by sha256 DIGEST rather than by the floating tag; both Dockerfiles
# of this task carry the same digest. A tag can be re-pointed upstream at any time, which would
# leave the anchors impossible to rebuild from this repo.
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
WORKDIR /tests

# EXACT pins, byte-identical in environment/Dockerfile and
# tests/Dockerfile. Versions read off the built image (pip list --format=freeze), not
# invented, so today's anchors stay valid. Ranges made the anchors a function of the day
# the image happened to be built, i.e. unreproducible from the repo.
RUN pip install --no-cache-dir "numpy==2.5.1"
# single-thread BLAS/OpenMP. Threaded reductions change the summation order, so
# a syrk / covariance / SVD / ridge solve is bit-different at 1 vs 16 threads (measured on the
# 96-core host). Pinned in BOTH images so the agent's own self-check is bit-identical to grading.
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2


# G1 hardening: dedicated unprivileged account the UNTRUSTED solve subprocess is dropped to
# (grade.py: SOLVER_UID / SOLVER_GID). Running the submission as root in the grader's own
# filesystem is what makes "seed a reward then SIGKILL the grader", "overwrite the scorer",
# "read /proc/1/environ" and "read the sealed engine" all possible at once.
RUN groupadd -g 4242 solver && useradd -u 4242 -g 4242 -M -s /usr/sbin/nologin solver

# Observable panel ONLY (no held-out truth) — the submission's solve() runs against this.
COPY features/ /tests/features/
# Public DCF bridge (same valuation.py the agent had under /app/lib) — put on the solve subprocess's
# PYTHONPATH so a forecaster that imports `valuation` for its own value checks still runs at grade time.
COPY libpub/ /tests/libpub/
# Sealed answer key (realized drivers + perfect-foresight value, dev/test tag) + sealed engine.
COPY heldout/  /tests/heldout/
# Isolated solve harness (runs the submission) + grader + entrypoint (run AFTER the agent finishes).
COPY solve.py /tests/solve.py
COPY grade.py /tests/grade.py
COPY test.sh  /tests/test.sh
# Sealed anchors (moved out of task.toml [verifier.env]; see anchors.json + task.toml comment).
COPY anchors.json /tests/anchors.json

# Hardening: the trusted side is root-owned and NOT writable by uid 4242, and the
# sealed material is not even readable by it (holes D/E). root ignores 0400/0700, so the grader
# itself still reads everything it needs before it drops privileges and spawns the submission.
RUN chown -R root:root /tests \
 && chmod 0755 /tests /tests/features /tests/libpub \
 && chmod 0644 /tests/grade.py /tests/solve.py /tests/features/* /tests/libpub/* \
 && chmod 0755 /tests/test.sh \
 && chmod 0700 /tests/heldout \
 && chmod 0400 /tests/anchors.json /tests/heldout/holdout_truth.json \
      /tests/heldout/_valuation.py /tests/heldout/_fin_compute_engine.py \
 && rm -rf /tests/heldout/__pycache__
