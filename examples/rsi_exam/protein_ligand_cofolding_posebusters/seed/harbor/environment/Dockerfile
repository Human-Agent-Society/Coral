FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    BOLTZ_CACHE=/opt/boltz_cache \
    CHAI_DOWNLOADS_DIR=/opt/chai_cache

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates wget tar gzip libgomp1 gcc \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock \
 && python -m pip check

RUN mkdir -p /opt/boltz_cache /opt/chai_cache/models_v2 /opt/chai_cache/esm \
 && wget -q -O /opt/boltz_cache/ccd.pkl \
      https://huggingface.co/boltz-community/boltz-1/resolve/7c1d83b779e4c65ecc37dfdf0c6b2788076f31e1/ccd.pkl \
 && wget -q -O /opt/boltz_cache/boltz1_conf.ckpt \
      https://huggingface.co/boltz-community/boltz-1/resolve/7c1d83b779e4c65ecc37dfdf0c6b2788076f31e1/boltz1_conf.ckpt \
 && echo "2d3b2f03a3c5665944adba51e33263511e51b21c9cd05d902f9c4b7c1e58d2f4  /opt/boltz_cache/ccd.pkl" | sha256sum -c - \
 && echo "fea245d912c570ec117b2277c2719f312a6fc109c07b6f6ef741690ee775c2f5  /opt/boltz_cache/boltz1_conf.ckpt" | sha256sum -c -

RUN wget -q -O /opt/chai_cache/conformers_v1.apkl \
      https://chaiassets.com/chai1-inference-depencencies/conformers_v1.apkl \
 && wget -q -O /opt/chai_cache/models_v2/feature_embedding.pt \
      https://chaiassets.com/chai1-inference-depencencies/models_v2/feature_embedding.pt \
 && wget -q -O /opt/chai_cache/models_v2/bond_loss_input_proj.pt \
      https://chaiassets.com/chai1-inference-depencencies/models_v2/bond_loss_input_proj.pt \
 && wget -q -O /opt/chai_cache/models_v2/token_embedder.pt \
      https://chaiassets.com/chai1-inference-depencencies/models_v2/token_embedder.pt \
 && wget -q -O /opt/chai_cache/models_v2/trunk.pt \
      https://chaiassets.com/chai1-inference-depencencies/models_v2/trunk.pt \
 && wget -q -O /opt/chai_cache/models_v2/diffusion_module.pt \
      https://chaiassets.com/chai1-inference-depencencies/models_v2/diffusion_module.pt \
 && wget -q -O /opt/chai_cache/models_v2/confidence_head.pt \
      https://chaiassets.com/chai1-inference-depencencies/models_v2/confidence_head.pt \
 && wget -q -O /opt/chai_cache/esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt \
      https://chaiassets.com/chai1-inference-depencencies/esm2/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt \
 && test "$(stat -c %s /opt/chai_cache/conformers_v1.apkl)" = "124726115" \
 && test "$(stat -c %s /opt/chai_cache/models_v2/feature_embedding.pt)" = "4587962" \
 && test "$(stat -c %s /opt/chai_cache/models_v2/bond_loss_input_proj.pt)" = "5502" \
 && test "$(stat -c %s /opt/chai_cache/models_v2/token_embedder.pt)" = "6286772" \
 && test "$(stat -c %s /opt/chai_cache/models_v2/trunk.pt)" = "633042135" \
 && test "$(stat -c %s /opt/chai_cache/models_v2/diffusion_module.pt)" = "476546304" \
 && test "$(stat -c %s /opt/chai_cache/models_v2/confidence_head.pt)" = "55259496" \
 && test "$(stat -c %s /opt/chai_cache/esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt)" = "5678940466" \
 && find /opt/boltz_cache /opt/chai_cache -type f -exec chmod 0444 {} +

RUN echo "f2161256b565bbd84198da8b3e3d3978ae2f179f778d0995cd135c4c0c57c41f  /opt/chai_cache/conformers_v1.apkl" | sha256sum -c - \
 && echo "58600fb282827ee105b4bf33845c2eb2dce2a7d2ba7582951f76c9290c66c019  /opt/chai_cache/models_v2/feature_embedding.pt" | sha256sum -c - \
 && echo "98f3b2e415d11e460a684cdbfc213ca80447308022b1953cc5326ad468948851  /opt/chai_cache/models_v2/bond_loss_input_proj.pt" | sha256sum -c - \
 && echo "60cada19fd9c77d8298131a00368d162b6bc7d3c836dd36c715db04d21f7bbde  /opt/chai_cache/models_v2/token_embedder.pt" | sha256sum -c - \
 && echo "b7a08e104455f76491088a5eeaea9bb95a08168c48969b0b892971346e51ccbc  /opt/chai_cache/models_v2/trunk.pt" | sha256sum -c - \
 && echo "bd77cf4095ad6b8a4eae7e390d01d24be1ebd819922ca3f28e8744f9945c9cad  /opt/chai_cache/models_v2/diffusion_module.pt" | sha256sum -c - \
 && echo "82a4ac9f934fdd0e73870150f508cecbe010c1383234bdadcaa70d57323deb6b  /opt/chai_cache/models_v2/confidence_head.pt" | sha256sum -c - \
 && echo "074673b97e1c1ff9c3cf949294749dd446ad1c83cee50a466b4841bb4d89c27a  /opt/chai_cache/esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt" | sha256sum -c -

ARG PROTEIN_DATASET_REVISION=33868c4f399155a6ad8d838a8b4e2140eecb0993
RUN mkdir -p /app/data \
 && wget -q -O /tmp/visible.tar.gz \
      "https://huggingface.co/datasets/Jianwen/protein_ligand_cofolding_posebusters/resolve/${PROTEIN_DATASET_REVISION}/archives/visible.tar.gz?download=true" \
 && echo "8127c3697509c81b866cae0aa7739ede5980350e9f01b5f527c204091e405f8e  /tmp/visible.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/visible.tar.gz -C /app/data \
 && echo "1a627d2ddd9cbf1210349a0e451dc30adfb8c4dddf88d1b9cab7098d9f452754  /app/data/visible/items.json" | sha256sum -c - \
 && rm /tmp/visible.tar.gz

COPY methods/ /app/methods/
COPY metric.py /app/metric.py
COPY source_contract.py /app/source_contract.py
COPY selfcheck.py /app/selfcheck.py
