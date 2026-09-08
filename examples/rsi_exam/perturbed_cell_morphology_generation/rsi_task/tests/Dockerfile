FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:2fcc4280646484290cc50dce5e65f388dd04352b07cbe89a635703bd1f9aedb6

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    TORCH_HOME=/opt/torch \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        libffi8 \
        libgcc-s1 \
        libgomp1 \
        libncurses6 \
        libreadline8 \
        libsqlite3-0 \
        libstdc++6 \
        libuuid1 \
        libx11-6 \
        libxau6 \
        libxcb1 \
        libxdmcp6 \
        openssl \
        python3.10 \
        python3-pip \
        tk8.6 \
        xz-utils \
        zlib1g \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python

WORKDIR /tests
COPY requirements.lock.txt /tmp/requirements.lock.txt
RUN python -m pip install pip==25.1 setuptools==78.1.1 wheel==0.45.1 \
    && python -m pip install -r /tmp/requirements.lock.txt \
    && rm /tmp/requirements.lock.txt

# The same public source is prepared independently for the offline verifier.
COPY prepare_data.py copy_check.py /tmp/
RUN PYTHONPATH=/tmp python /tmp/prepare_data.py --role verifier --output-root /tests/sealed \
    && rm -rf /tmp/cellflux-data /tmp/prepare_data.py /tmp/copy_check.py

COPY grade.py run_predict.py fid_metric.py copy_check.py test.sh anchors.json /tests/
RUN mkdir -p /app/methods /app/submission /logs/verifier \
    && chmod +x /tests/test.sh \
    && chmod -R a+rX /tests /app /opt/torch \
    && chown root:root /tests/anchors.json /tests/sealed/targets.npz /tests/sealed/train_reference.npz \
    && chmod 0400 /tests/anchors.json /tests/sealed/targets.npz /tests/sealed/train_reference.npz

CMD ["/tests/test.sh"]
