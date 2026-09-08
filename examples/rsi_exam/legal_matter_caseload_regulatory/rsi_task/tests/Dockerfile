FROM python:3.11-slim
WORKDIR /tests
COPY anchors.json /tests/anchors.json
RUN chown root:root /tests/anchors.json && chmod 0400 /tests/anchors.json
COPY grade.py test.sh apex_engine.py /tests/
COPY heldout/ /heldout/
