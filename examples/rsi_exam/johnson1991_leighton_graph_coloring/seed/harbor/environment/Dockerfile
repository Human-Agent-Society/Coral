# Agent image for johnson1991_leighton_graph_coloring (flat_minconf_v3):
# visible generated graphs only, plus the editable tabu/greedy baseline solver and the
# stdlib+numpy self-check. Hidden graphs, hidden anchors, generator audit data and best-known
# colorings never enter this image.
FROM python:3.12-slim
WORKDIR /app
# Allowed third-party library: numpy only (kept identical in the verifier image).
RUN pip install --no-cache-dir numpy==2.2.5
# NOTE: scoring anchors (floor/sota) are NOT shipped to the agent — they live sealed under
# tests/heldout/. data/ now holds only visible.json, the problem data the agent must solve.
COPY data/        /app/data/
COPY selfcheck.py /app/selfcheck.py
COPY methods/     /app/methods/
