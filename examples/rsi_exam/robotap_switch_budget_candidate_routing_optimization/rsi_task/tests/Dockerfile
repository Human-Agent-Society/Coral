FROM python:3.12-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/tests \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir \
    numpy==1.26.4 scipy==1.16.2 scikit-learn==1.7.2 joblib==1.5.3

WORKDIR /tests
COPY . /tests/

RUN mkdir -p /tests/heldout /tests/trusted /run/candidate-routing-grade \
    && chmod 0711 /run/candidate-routing-grade \
    && chmod 0755 /tests/child_entry.py /tests/grade.py /tests/test.sh \
        /tests/test_reward_log_owner.sh \
    && chmod 0644 /tests/fail_closed.py /tests/secure_session.py /tests/test_verifier.py \
        /tests/test_trusted_equivalence.py /tests/reward_log_owner.sh \
    && find /tests/heldout -type d -exec chmod 0500 {} + \
    && find /tests/heldout -type f -exec chmod 0400 {} + \
    && find /tests/trusted -type d -exec chmod 0500 {} + \
    && find /tests/trusted -type f -exec chmod 0400 {} + \
    && chown root:root /tests/anchors.json \
    && chmod 0400 /tests/anchors.json \
    && python -m py_compile /tests/*.py \
    && /tests/test_reward_log_owner.sh \
    && python -m unittest -v /tests/test_verifier.py /tests/test_trusted_equivalence.py
