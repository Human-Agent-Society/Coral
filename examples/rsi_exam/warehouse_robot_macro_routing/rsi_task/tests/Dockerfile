FROM python:3.11-slim

# g++ so C++/other compiled submissions can build via solution/build.sh; the
# grader itself and the official visualiser (a static-ish Rust binary) need no
# extra deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tests
COPY grade.py test.sh /tests/
COPY anchors.json /tests/anchors.json
RUN chown root:root /tests/anchors.json && chmod 0400 /tests/anchors.json
COPY heldout/ /heldout/
RUN chmod +x /heldout/vis
