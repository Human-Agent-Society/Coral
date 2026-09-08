FROM python:3.11-slim

RUN pip install --no-cache-dir "jax[tpu]==0.6.2"

WORKDIR /app
COPY methods/ /app/methods/
COPY eval/ /app/eval/
COPY data/ /app/data/
COPY selfcheck.py /app/selfcheck.py
