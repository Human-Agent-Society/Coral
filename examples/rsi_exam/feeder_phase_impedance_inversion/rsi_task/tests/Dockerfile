# The base image is pinned by sha256 DIGEST rather than by the floating tag; both Dockerfiles
# of this task carry the same digest. A tag can be re-pointed upstream at any time, which would
# leave the anchors impossible to rebuild from this repo.
FROM ubuntu:22.04@sha256:0e0a0fc6d18feda9db1590da249ac93e8d5abfea8f4c3c0c849ce512b5ef8982

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir "numpy==2.0.2" "scipy==1.13.1" \
      "dss-python-backend==0.14.5" "dss-python==0.15.7" \
      "opendssdirect.py==0.9.4"

# single-thread BLAS: discrete searches amplify float-context differences;
# pinning is required for exact anchor reproducibility (spike-measured)
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2

# trusted-parent marker: grade.py deletes the sealed sources from disk
# before any submitted code runs.
# The marker is a root-only sentinel FILE (/tests/.verifier_image), never an ENV var: a
# host-side `-e VERIFIER_IMAGE=0` could switch the whole sealing step off, whereas the file is
# created at build time, /tests is root-owned, and the de-privileged uid 4242 cannot remove or
# alter it.

# Dedicated unprivileged account the submitted method's subprocess is dropped to
# (grade.py: SOLVER_UID/SOLVER_GID). One change closes holes A/B/D/E at the source:
# uid 4242 cannot write /logs/verifier, cannot write /tests, cannot signal the root
# parent, cannot read /proc/1/environ, and cannot read the 0400 sealed anchors.
RUN groupadd -g 4242 solver && useradd -u 4242 -g 4242 -M -s /usr/sbin/nologin solver

WORKDIR /tests
COPY grade.py test.sh anchors.json /tests/
COPY heldout/ /heldout/

# Lock the trusted scoring chain and the sealed anchors to root; world-readable code,
# but anchors.json 0400 (root-only). root reads 0400 fine, so the parent
# still loads the anchors; uid 4242 cannot.
# /heldout must NOT be left at COPY's defaults (0755 dir + 0644 files): that makes the sealed
# generator (which carries the secret seeds) and the ground truth world-readable to uid 4242.
# It is currently not exploitable only because grade.py calls seal_disk() to delete them before
# any child process starts -- but that is a TIMING guarantee, not a permission one, and it
# fails the moment the sentinel is missing, seal_disk raises, or a spawn is moved ahead of the
# seal. Hence 0700 dir + 0400 files. root ignores 0400, so the grader parent still reads the
# sealed data into memory and still rmtree()s it.
RUN touch /tests/.verifier_image \
 && chown -R root:root /tests /heldout \
 && chmod 0755 /tests \
 && chmod 0644 /tests/grade.py \
 && chmod 0755 /tests/test.sh \
 && chmod 0400 /tests/anchors.json /tests/.verifier_image \
 && chmod 0700 /heldout \
 && chmod 0400 /heldout/*
