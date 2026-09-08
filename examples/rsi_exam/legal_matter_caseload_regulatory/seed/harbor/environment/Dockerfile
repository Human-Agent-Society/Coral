FROM python:3.11-slim
# The `matter` client and the judge engine are stdlib-only; the world documents
# are pre-extracted text. Nothing here holds rubric text — the grader container does.
RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY bin/ /usr/local/bin/
RUN chmod +x /usr/local/bin/matter
COPY tools/ /app/tools/
COPY methods/ /app/methods/
