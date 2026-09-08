# Agent image: visible conversation data + inherited baseline memory system.
FROM python:3.11-slim
WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
      "openai>=1.0" "rank_bm25>=0.2" "numpy>=1.24" "sentence-transformers>=2.2" \
      "scikit-learn>=1.3" "pandas>=2.0" "networkx>=3.0" "pyyaml>=6.0"

# Harbor's agent setup runs `apt-get install curl bash nodejs npm ripgrep` inside the
# agent container to install codex -- and that happens under [agent]'s no-network/allowlist
# policy, not [environment]'s public one. Bake them here (build time has the network) or
# that setup step hangs until it times out.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ripgrep curl bash nodejs npm ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# embedding models cached at build time (semantic retrieval is optional to use)
ENV HF_HOME=/opt/hf
RUN python3 -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-base-en-v1.5'); SentenceTransformer('all-MiniLM-L6-v2')"

COPY data/        /app/data/
COPY eval/        /app/eval/
COPY methods/     /app/methods/
COPY selfcheck.py /app/selfcheck.py

ENV PYTHONPATH=/app
