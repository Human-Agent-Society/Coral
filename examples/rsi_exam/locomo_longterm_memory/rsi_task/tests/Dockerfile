# Verifier image: sealed holdout + grader. Package set must match the agent
# image — the agent's memory system is re-run here.
FROM python:3.11-slim
WORKDIR /tests

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
      "openai>=1.0" "rank_bm25>=0.2" "numpy>=1.24" "sentence-transformers>=2.2" \
      "scikit-learn>=1.3" "pandas>=2.0" "networkx>=3.0" "pyyaml>=6.0"

# embedding models cached at build time; grading runs without HF network access
ENV HF_HOME=/opt/hf
RUN python3 -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-base-en-v1.5'); SentenceTransformer('all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1

COPY grade.py       /tests/grade.py
COPY run_holdout.py /tests/run_holdout.py
COPY test.sh        /tests/test.sh
COPY heldout/       /tests/heldout/
COPY anchors.json   /tests/anchors.json
RUN chown root:root /tests/anchors.json && chmod 400 /tests/anchors.json

ENV PYTHONPATH=/tests
