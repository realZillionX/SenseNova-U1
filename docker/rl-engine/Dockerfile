FROM mostar-u1-runtime:v4

ARG SENSENOVA_COMMIT=34ca2f66e7006489a0184eb8896a75f4081a0257
ARG LIGHTLLM_COMMIT=307ec195c1f5ce4779d56eae37a072b15294cfdb
ARG LIGHTX2V_COMMIT=f4aa8833ef0f6d8c256612df769896b6c8e04dd9

COPY docker/rl-engine /workspace/mova-rl-engine/docker/rl-engine
COPY evaluation/easi/lightllm-stack/LightLLM /workspace/mova-rl-engine/evaluation/easi/lightllm-stack/LightLLM
COPY evaluation/easi/lightllm-stack/LightX2V /workspace/mova-rl-engine/evaluation/easi/lightllm-stack/LightX2V
COPY scripts/rl_engine /workspace/mova-rl-engine/scripts/rl_engine
COPY examples/serving /workspace/mova-rl-engine/examples/serving
RUN bash /workspace/mova-rl-engine/docker/rl-engine/build_runtime.sh /workspace/mova-rl-engine

ENV PATH=/opt/mostar-u1-py312/bin:${PATH} \
    PYTHONPATH=/workspace/mova-rl-engine/evaluation/easi/lightllm-stack/LightLLM:/workspace/mova-rl-engine/evaluation/easi/lightllm-stack/LightX2V \
    MOVA_SENSENOVA_COMMIT=${SENSENOVA_COMMIT} \
    MOVA_LIGHTLLM_COMMIT=${LIGHTLLM_COMMIT} \
    MOVA_LIGHTX2V_COMMIT=${LIGHTX2V_COMMIT} \
    MOVA_PLATFORM_PARENT_IMAGE=mostar-u1-runtime:v4 \
    MOVA_IMAGE_DIGEST=inspire:mova-u15-lightllm-x2v-rl:v2 \
    MOVA_RL_TRACE_DIR=/tmp/mova_rl_traces \
    MOVA_RL_TRACE_TTL=3600

WORKDIR /workspace/mova-rl-engine
