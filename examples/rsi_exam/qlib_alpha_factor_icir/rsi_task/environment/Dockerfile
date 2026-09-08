# Agent image for qlib_alpha_factor_icir: rebuild the visible factor panels (train/valid +
# feature_catalog) from the public Qlib CN bundle, then ship them with the editable baseline.

# ---- Stage 1: regenerate visible data from the public Qlib CN source --------------------------
FROM python:3.12 AS datagen
WORKDIR /gen
RUN pip install --no-cache-dir "pyqlib==0.9.7" "pyarrow>=14"
COPY prepare_data.py /gen/prepare_data.py
# build-time network: downloads the public CN bundle to /opt/qlib_cn_data, then writes /app/data
RUN python prepare_data.py --role visible --out /app/data \
 && ls -l /app/data

# ---- Stage 2: lean agent runtime (no qlib, no raw bundle) -------------------------------------
FROM python:3.12-slim
WORKDIR /app

RUN pip install --no-cache-dir \
      "numpy>=1.24" "pandas>=2.0" "scikit-learn>=1.3" \
      "PyYAML>=6.0" "pyarrow>=14"
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Visible data (public train/valid panels + feature catalog), baked from the datagen stage.
COPY --from=datagen /app/data/ /app/data/
# Free, unlimited proxy the agent iterates against (NOT the hidden test period): python /app/selfcheck.py
COPY selfcheck.py /app/selfcheck.py
# The editable baseline the agent improves in place (this dir is what the grader re-runs).
COPY methods/ /app/methods/
