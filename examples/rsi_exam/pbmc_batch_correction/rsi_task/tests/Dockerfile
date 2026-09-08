# Verifier image for pbmc_batch_correction: scientific-python runtime + scanpy/igraph (the
# best-over-resolution Leiden NMI the paper uses) + the SEALED test split, plus the isolated harness.
FROM python:3.12-slim
WORKDIR /tests

RUN pip install --no-cache-dir \
      "numpy>=1.24" "scipy>=1.10" "scikit-learn>=1.3" "pandas>=2.0" \
      "anndata>=0.10" "scanpy>=1.10" "igraph>=0.11" "huggingface_hub>=0.23"

# Hidden split (the embedding + NMI are computed on the test cells). Public HF dataset, no token.
COPY download_data.py /tmp/download_data.py
RUN python /tmp/download_data.py --out /tests/heldout --files pbmc_atlas_atac_test.h5ad \
 && ls -lh /tests/heldout

COPY solve.py /tests/solve.py
COPY grade.py /tests/grade.py
COPY test.sh  /tests/test.sh
COPY anchors.json /tests/anchors.json
RUN chown root:root /tests/anchors.json && chmod 0400 /tests/anchors.json
