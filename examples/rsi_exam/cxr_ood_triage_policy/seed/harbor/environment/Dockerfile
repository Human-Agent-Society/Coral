# Agent image for cxr_ood_triage_policy: ships the visible labeled grade sample + labeled
# calibration sample + the score schema + the editable weak baseline + the free self-check.
# Standard-library only — the agent's solver.py is re-imported and re-run inside the stdlib-only
# verifier image. NO held-out study ever enters this image.
# the base image is pinned to a sha256 digest, the SAME digest as tests/
# Dockerfile. The digest was taken from `docker pull python:3.12-slim` on 2026-08-01. It used to be
# the floating tag `python:3.12-slim`; a floating tag resolves to different images on different
# days, which makes the anchors unreproducible from the repository.
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
WORKDIR /app

# single-thread BLAS/OpenMP, pinned in BOTH images so the agent's
# self-check is bit-identical to grading. DEFENSIVE here: both images are stdlib-only
# (no numpy), so nothing reads these today -- they cost nothing and make it impossible
# to add a numeric package later and silently make the score depend on the host's
# core count. Exact pip pins are N/A for the same reason: there is no pip line.
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2


# Visible grade sample (with gold for the local self-check), labeled calibration, and the schema.
COPY data/        /app/data/
# Free, unlimited local dry-run: python /app/selfcheck.py
COPY selfcheck.py /app/selfcheck.py
# The editable weak baseline (this dir is what the grader re-runs).
COPY methods/     /app/methods/
