FROM python:3.12
WORKDIR /tests
ENV SCENARIO_DATA=/data/scenario

# Thread pinning. The constant is 1, not [verifier] cpus = 2, and that is
# deliberate: the grader is a single libsumo simulation stepped in one python
# loop, with no multiprocessing and no BLAS reduction anywhere on the path --
# the two declared cpus buy the container headroom, not parallelism. What the pin
# actually requires is that the count never be a function of the HOST, and a
# fixed 1 satisfies that. Set in BOTH images so the agent's selfcheck and the
# grader are isomorphic (an image that omits this runs the selfcheck at the
# host's default while grading runs pinned, and the selfcheck looks healthy the
# whole way). Mirrors the constants in tests/grade.py.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# libsumo 1.19.0 links against libGL/libX11 even for headless use; without these the
# very first `import libsumo` dies with "libGL.so.1: cannot open shared object file",
# i.e. neither selfcheck nor grade.py reaches a single line of real work.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libx11-6 \
 && rm -rf /var/lib/apt/lists/*
#
# sumolib/traci are pinned too: they used to be pulled in unpinned as transitive deps, so
# a fresh build could silently mix a 1.19.0 core with a newer python layer — and the whole
# point of the pin is that the scenario + bindings define every measured number.
RUN pip install --no-cache-dir "eclipse-sumo==1.19.0" "libsumo==1.19.0" \
                               "sumolib==1.19.0" "traci==1.19.0"

# Same public scenario the agent used (net + demand); the hidden part is the protocol.
COPY prepare_data.py /tmp/prepare_data.py
RUN python /tmp/prepare_data.py --out /data && rm /tmp/prepare_data.py

# Trusted scorer + sealed hidden protocol.
COPY eval_core.py /tests/eval_core.py
# Trust boundary: the filtered libsumo bridge.  It is re-exec'd as
# `python sim_rpc.py --controller ...` by ControllerHost, in the process that is dropped
# to uid 65534, so it must stay world-READABLE (it is world-readable but not writable,
# like every other public module in /tests) while /tests/heldout and anchors.json are not.
# Byte-identical to environment/sim/sim_rpc.py: the agent is told it reads the exact
# scoring core, and the selfcheck must enforce the same call policy as the grader.
COPY sim_rpc.py   /tests/sim_rpc.py
COPY run_eval.py  /tests/run_eval.py
COPY grade.py     /tests/grade.py
COPY test.sh      /tests/test.sh
COPY heldout/     /tests/heldout/

# Anchors: root-owned, 0400, and present only in the verifier image. A file rather than
# [verifier.env] because PID 1 in the verifier container is test.sh and submitted code can read
# /proc/1/environ -- scrubbing the env of the child alone does not seal it. The 0400 only truly
# binds once submitted code has been dropped to an unprivileged uid; before that it at least
# keeps the anchors out of every process environ.
COPY anchors.json /tests/anchors.json
RUN chown root:root /tests/anchors.json && chmod 0400 /tests/anchors.json

RUN chmod +x /tests/test.sh

# Make the seals real.  grade.py drops the solver child to uid/gid 65534
# BEFORE exec, so from 2026-08-05 these modes exclude somebody: /tests/heldout (the
# sealed protocol) and /tests/anchors.json (what the submission is scored against) are
# root-only, while the rest of /tests stays world-readable because the unprivileged
# child has to exec run_eval.py and import the trusted public modules.  The scenario
# data is explicitly made world-readable for the same reason.
RUN chown -R root:root /tests \
 && chmod 0555 /tests \
 && chmod 0500 /tests/heldout \
 && find /tests/heldout -type d -exec chmod 0500 {} + \
 && find /tests/heldout -type f -exec chmod 0400 {} + \
 && chmod 0400 /tests/anchors.json /tests/grade.py \
 && chmod 0500 /tests/test.sh \
 && chmod -R a+rX /data
