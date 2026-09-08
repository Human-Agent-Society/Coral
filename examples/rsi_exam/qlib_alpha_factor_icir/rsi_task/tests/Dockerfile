# Verifier image for qlib_alpha_factor_icir: rebuild the train/valid panels and the sealed
# test_features/test_labels from the public Qlib CN bundle, alongside the grader.

# ---- Stage 1: regenerate public panels + sealed held-out from the public Qlib CN source ------
FROM python:3.12 AS datagen
WORKDIR /gen
RUN pip install --no-cache-dir "pyqlib==0.9.7" "pyarrow>=14"
COPY prepare_data.py /gen/prepare_data.py
# build-time network: download the public CN bundle once, then emit both subsets.
RUN python prepare_data.py --role visible --out /out/data \
 && python prepare_data.py --role heldout --out /out/heldout \
 && ls -l /out/data /out/heldout

# ---- Stage 2: lean verifier runtime (no qlib, no raw bundle) ----------------------------------
FROM python:3.12-slim
WORKDIR /tests

# MUST stay identical to environment/Dockerfile: the grader re-runs the agent's code here, so
# anything importable there must be importable here too (and nothing more).
RUN pip install --no-cache-dir \
      "numpy>=1.24" "pandas>=2.0" "scikit-learn>=1.3" \
      "PyYAML>=6.0" "pyarrow>=14"
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Public panels the grader re-fits on (NOT secret) + sealed answer key.
COPY --from=datagen /out/data/    /tests/data/
COPY --from=datagen /out/heldout/ /tests/heldout/
# Grader + entrypoint (run AFTER the agent finishes; see test.sh).
COPY --chown=root:root --chmod=400 anchors.json /tests/anchors.json
COPY grade.py /tests/grade.py
COPY test.sh  /tests/test.sh
