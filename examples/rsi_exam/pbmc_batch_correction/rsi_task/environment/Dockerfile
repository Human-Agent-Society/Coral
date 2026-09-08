# Agent image for pbmc_batch_correction: scientific-python runtime + the VISIBLE split (downloaded
# from the public HF dataset at build), the editable baseline, and the self-check. scanpy/igraph are
# installed so the self-check computes the same best-over-resolution Leiden NMI the judge uses.
FROM python:3.12-slim
WORKDIR /app

RUN pip install --no-cache-dir \
      "numpy>=1.24" "scipy>=1.10" "scikit-learn>=1.3" "pandas>=2.0" \
      "anndata>=0.10" "scanpy>=1.10" "igraph>=0.11" "huggingface_hub>=0.23"
RUN pip install --no-cache-dir "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu

# Visible split only (the ATAC file is all this scenario needs). Public HF dataset, no token.
COPY data/download_data.py /tmp/download_data.py
RUN python /tmp/download_data.py --out /app/data --files pbmc_atlas_atac.h5ad \
 && ls -lh /app/data \
 && rm -f /tmp/download_data.py

COPY data/README.md /app/data/README.md
COPY methods/       /app/methods/
COPY selfcheck.py   /app/selfcheck.py
