ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG FORGE_COMMIT=uncommitted
ARG LIGHTLLM_COMMIT=f08f12b5ec243f63f02b46f83b587c9a952f99bb
ARG LIGHTX2V_COMMIT=06c8f8eb6e1c9d00fd1c42c2937f725e1e337c06

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential git ca-certificates python3.12 python3.12-dev python3.12-venv \
      libibverbs1 ibverbs-providers librdmacm1 rdma-core \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/sensenova-forge-py312

ENV PATH=/opt/sensenova-forge-py312/bin:${PATH} \
    PYTHON_BIN=/opt/sensenova-forge-py312/bin/python \
    FORGE_RUNTIME_IMAGE=sensenova-u15-forge:unified-v9 \
    FORGE_ROOT=/workspace/sensenova-forge \
    FORGE_COMMIT=${FORGE_COMMIT} \
    FORGE_LIGHTLLM_COMMIT=${LIGHTLLM_COMMIT} \
    FORGE_LIGHTX2V_COMMIT=${LIGHTX2V_COMMIT} \
    FORGE_LIGHTLLM_ROOT=/opt/sensenova-forge/sources/LightLLM \
    FORGE_LIGHTX2V_ROOT=/workspace/sensenova-forge/serving/third_party/LightX2V \
    FORGE_RL_TRACE_DIR=/dev/shm/sensenova_forge_rl_traces \
    FORGE_RL_TRACE_TTL=3600

COPY . /workspace/sensenova-forge
RUN bash /workspace/sensenova-forge/docker/rl-engine/build_runtime.sh /workspace/sensenova-forge

WORKDIR /workspace/sensenova-forge
