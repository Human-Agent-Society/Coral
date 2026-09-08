FROM ubuntu:22.04@sha256:0e0a0fc6d18feda9db1590da249ac93e8d5abfea8f4c3c0c849ce512b5ef8982
# The base image is pinned by sha256 DIGEST, not by the floating `ubuntu:22.04` tag (22.04 is
# an LTS whose point releases keep moving). Without the pin the anchors would hang on whatever
# the tag pointed at on build day and could not be rebuilt from this repo. This task also
# depends on apt-installed libblas3/liblapack3, whose versions come entirely from the base
# image snapshot -- DEVSIM's sparse solve follows them bit for bit, so this matters even more
# than pinning pip versions.

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip libblas3 liblapack3 \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir "numpy==2.0.2" "scipy==1.13.1" "devsim==2.10.0"

# DEVSIM needs BLAS/LAPACK shared objects resolvable at import time
ENV DEVSIM_MATH_LIBS="liblapack.so.3:libblas.so.3"
RUN python3 -c "import devsim"

# Single-thread BLAS/OpenMP -- identical to environment/Dockerfile.
# Anchor values in anchors.json only reproduce under a fixed thread count;
# grade.py additionally forces the same four vars into the solver child's env as
# CONSTANTS, so a host that exports them cannot move the score.
ENV OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4

# trusted-parent marker: grade.py deletes the sealed sources from disk
# before any submitted code runs (only the image build can set this)
ENV VERIFIER_IMAGE=1

# Dedicated unprivileged account the grader drops to before
# it ever runs submitted code (grade.py: SOLVER_UID / SOLVER_GID = 4242).
RUN groupadd -g 4242 solver \
 && useradd -u 4242 -g 4242 -M -s /usr/sbin/nologin solver

WORKDIR /tests
COPY grade.py test.sh anchors.json /tests/
COPY heldout/ /heldout/

# Trusted side is root-owned and not writable by uid 4242; the sealed
# anchors and the sealed lot are 0400 (hole E) -- root reads them regardless,
# and grade.py does so before dropping privileges.
RUN chown -R root:root /tests /heldout \
 && chmod 0755 /tests /heldout \
 && chmod 0644 /tests/grade.py \
 && chmod 0755 /tests/test.sh \
 && chmod 0400 /tests/anchors.json \
 && chmod 0400 /heldout/instances_sealed.json /heldout/dd_core.py \
               /heldout/heldout_gen.py
