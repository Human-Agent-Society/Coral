FROM python:3.11-slim

RUN pip install --no-cache-dir "jax[tpu]==0.6.2"

WORKDIR /tests
COPY grade.py child_eval.py restricted_runner.py submission_guard.py test.sh \
     heldout_strata.json anchors.json baseline_manifest.json /tests/
COPY eval/ /tests/eval/

# grade.py runs as root only to drop to per-case uids for the untrusted child;
# anchors stay root-owned 0400 so the deprivileged child cannot read them.
RUN chmod 0400 /tests/anchors.json /tests/baseline_manifest.json
