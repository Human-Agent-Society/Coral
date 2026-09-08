FROM python:3.12-slim-bookworm@sha256:72d3d75f2639ab82b34b29390ad3d6e0827c775befee94edda8e9976818f488d

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /opt/visible-build
RUN pip install "huggingface-hub==1.26.0" "pyarrow==25.0.0"
COPY prepare_assets.py /opt/visible-build/prepare_assets.py
RUN python /opt/visible-build/prepare_assets.py --role visible \
 && rm -rf /opt/visible-build /root/.cache /tmp/* \
 && chmod 0600 /opt/visible/answers.json
COPY evaluator/visible_server.py /opt/visible/visible_server.py
RUN chmod 0555 /opt/visible/visible_server.py
CMD ["python", "/opt/visible/visible_server.py"]
