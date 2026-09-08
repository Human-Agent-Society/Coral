# Verifier image for cxr_ood_triage_policy: ships the SEALED held-out grade studies (gold-bearing)
# + the labeled calibration sample + the two-phase grader. Held-out lands ONLY here, never in the
# agent image. Standard-library only, matching environment/Dockerfile (the agent's solver.py is
# re-imported and re-run here).
# The base image is pinned by sha256 DIGEST, the same one as environment/Dockerfile: different
# digests on the two sides would make the agent's self-check and the grader non-isomorphic.
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de
WORKDIR /tests

# single-thread BLAS/OpenMP, pinned in BOTH images so the agent's
# self-check is bit-identical to grading. DEFENSIVE here: both images are stdlib-only
# (no numpy), so nothing reads these today -- they cost nothing and make it impossible
# to add a numeric package later and silently make the score depend on the host's
# core count. Exact pip pins are N/A for the same reason: there is no pip line.
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2


# Dedicated unprivileged account the untrusted CHILD is dropped to (grade.py: SOLVER_UID/GID).
# Running the submission as root is what made the sealed gold readable, /tests writable,
# /logs/verifier writable and the trusted parent killable -- one uid closes all four.
RUN groupadd -g 4242 solver && useradd -u 4242 -g 4242 -M -s /usr/sbin/nologin solver

# Sealed held-out: hidden study bundles WITH gold_critical (stripped before the submission runs),
# plus the labeled calibration sample handed to triage() as a resource.
COPY heldout/        /tests/heldout/
# Two-phase grader: PARENT (trusted, holds gold + scores, never imports the submission) + CHILD
# runner (fresh python3 subprocess that imports + runs the submitted triage on stripped studies).
COPY grade.py        /tests/grade.py
COPY child_triage.py /tests/child_triage.py
COPY test.sh         /tests/test.sh
# Sealed reward anchors (moved out of task.toml [verifier.env], which leaks via /proc/1/environ).
COPY anchors.json    /tests/anchors.json

# Lock the scoring chain down against the uid-4242 child: nothing under /tests is writable by it,
# and the two sealed artifacts (gold labels, anchors) are not even readable. root is exempt from
# 0400, so the trusted parent still reads both. calibration.jsonl stays world-readable: it is the
# labeled sample the task deliberately hands to triage() as a resource, not protected gold.
RUN chown -R root:root /tests \
 && chmod 0755 /tests /tests/heldout \
 && chmod 0644 /tests/grade.py /tests/child_triage.py \
 && chmod 0755 /tests/test.sh \
 && chmod 0400 /tests/anchors.json /tests/heldout/grade.jsonl \
 && chmod 0444 /tests/heldout/calibration.jsonl
