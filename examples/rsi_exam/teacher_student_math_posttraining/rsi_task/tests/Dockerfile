FROM vllm/vllm-openai:v0.26.0@sha256:770fe65b2c73ee74a5c42165cf3433de4048cc2cd9c57a937ca4e35aba5aa87b

USER root
ENTRYPOINT []
ENV HF_HOME=/tmp/verifier-hf PIP_NO_CACHE_DIR=1 VLLM_NO_USAGE_STATS=1
WORKDIR /tests
RUN python3 -m pip install --no-cache-dir "pyarrow==25.0.0"
COPY prepare_hidden.py /tmp/prepare_hidden.py
RUN python3 /tmp/prepare_hidden.py \
 && rm -rf /tmp/prepare_hidden.py /tmp/verifier-hf /root/.cache/huggingface /tmp/*
COPY anchors.json reward_mapping.py grade.py test.sh /tests/
RUN chown -R root:root /tests \
 && chmod 0400 /tests/anchors.json /tests/heldout/sealed.json \
 && chmod 0444 /tests/ASSET_MANIFEST.json /tests/reward_mapping.py \
 && chmod 0500 /tests/grade.py /tests/test.sh \
 && chmod -R a+rX /tests/trusted-tokenizer
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
CMD ["/tests/test.sh"]
