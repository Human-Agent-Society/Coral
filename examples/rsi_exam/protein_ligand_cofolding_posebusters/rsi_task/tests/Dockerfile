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

WORKDIR /tests

RUN apt-get update \
 && apt-get install -y --no-install-recommends libseccomp2 \
 && rm -rf /var/lib/apt/lists/* \
 && ldconfig -p | grep -q 'libseccomp\.so\.2'

RUN groupadd --gid 10001 runner \
 && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin runner

ARG PROTEIN_DATASET_REVISION=33868c4f399155a6ad8d838a8b4e2140eecb0993
RUN wget -q -O /tmp/heldout.tar.gz \
      "https://huggingface.co/datasets/Jianwen/protein_ligand_cofolding_posebusters/resolve/${PROTEIN_DATASET_REVISION}/archives/heldout.tar.gz?download=true" \
 && echo "f498ed3b574d4e3ac3900d1d1170c188c5c3d270f8c770e47cb9221f197e6cff  /tmp/heldout.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/heldout.tar.gz -C /tests \
 && echo "4de18eb171f0f6194930a7fa2ddcfd6f502fd18d4896f79c3149ad534683e124  /tests/heldout/items.json" | sha256sum -c - \
 && rm /tmp/heldout.tar.gz

COPY evaluate.py /tests/evaluate.py
COPY source_contract.py /tests/source_contract.py
COPY child_predict.py /tests/child_predict.py
COPY score_pose_worker.py /tests/score_pose_worker.py
COPY grade.py /tests/grade.py
COPY anchors.json /tests/anchors.json
COPY test.sh /tests/test.sh
RUN chown -R root:root /tests \
 && chmod 0700 /tests/heldout \
 && chmod 0400 /tests/anchors.json \
 && chmod 0555 /tests/test.sh \
 && chmod 0555 /tests/source_contract.py /tests/child_predict.py /tests/score_pose_worker.py /tests/grade.py \
 && chmod 0444 /tests/evaluate.py
