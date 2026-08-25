FROM lightx2v/lightllm_lightx2v:20260407@sha256:bb1900389c320b37dbcfe51fdf4db76a198d38a10c4c80d8b9b0726f1fb43ac7

ARG SENSENOVA_COMMIT=34ca2f66e7006489a0184eb8896a75f4081a0257
ARG LIGHTLLM_COMMIT=1d6803d7414213186b52f85930e91377c286fd99
ARG LIGHTX2V_COMMIT=f05032c3c1036449d425b27cd854f1c2a398747f

COPY docker/rl-engine/requirements.lock /tmp/mova-requirements.lock
RUN python -m pip install --no-cache-dir --no-deps -r /tmp/mova-requirements.lock

COPY evaluation/easi/lightllm-stack/LightLLM /workspace/LightLLM
COPY evaluation/easi/lightllm-stack/LightX2V /workspace/LightX2V
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation -e /workspace/LightLLM \
 && python -m pip install --no-cache-dir --no-deps --no-build-isolation -e /workspace/LightX2V

COPY scripts/rl_engine /workspace/mova-rl-engine/scripts
COPY examples/serving /workspace/mova-rl-engine/examples/serving

ENV PYTHONPATH=/workspace/LightLLM:/workspace/LightX2V \
    MOVA_SENSENOVA_COMMIT=${SENSENOVA_COMMIT} \
    MOVA_LIGHTLLM_COMMIT=${LIGHTLLM_COMMIT} \
    MOVA_LIGHTX2V_COMMIT=${LIGHTX2V_COMMIT} \
    MOVA_IMAGE_DIGEST=sha256:bb1900389c320b37dbcfe51fdf4db76a198d38a10c4c80d8b9b0726f1fb43ac7 \
    MOVA_RL_TRACE_DIR=/tmp/mova_rl_traces \
    MOVA_RL_TRACE_TTL=3600

WORKDIR /workspace/mova-rl-engine
