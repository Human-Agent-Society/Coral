# Agent image for finscope_dcf_valuation: ship the messy observable fundamentals panel, the
# public DCF bridge, the editable baseline forecasters, and the free local self-check.
# The base image is pinned by sha256 DIGEST rather than by the floating tag; both Dockerfiles
# of this task carry the same digest. A tag can be re-pointed upstream at any time, which would
# leave the anchors impossible to rebuild from this repo.
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
WORKDIR /app

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


# Observable panel (coverage / fundamentals / calibration_truth) + the data dictionary.
COPY data/ /app/data/
# Public DCF bridge the agent may import for its own checks (the verifier uses a sealed copy).
COPY lib/  /app/lib/
# Editable baselines — methods/main is what the grader re-runs; this dir is collected as the artifact.
COPY methods/ /app/methods/
# Predecessor orientation + the free local dry-run (out-of-sample error on visible names).
COPY notes/ /app/notes/
COPY README.md          /app/README.md
COPY TODO.md            /app/TODO.md
COPY fit_calibration.py /app/fit_calibration.py
