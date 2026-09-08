# Verifier image for johnson1991_leighton_graph_coloring (flat_minconf_v3):
# sealed hidden graphs, sealed per-instance anchors, trusted scorer, and two-phase runner with
# per-instance isolation + hard time cap. Ships the SAME numpy as the agent image; no other
# third-party (the agent's solver.py is re-imported here, so its numpy import must resolve).
FROM python:3.12-slim
WORKDIR /tests
# Allowed third-party library: numpy only (kept identical to environment/Dockerfile).
RUN pip install --no-cache-dir numpy==2.2.5
COPY heldout/       /tests/heldout/
COPY grade.py       /tests/grade.py
COPY child_solve.py /tests/child_solve.py
COPY test.sh        /tests/test.sh
