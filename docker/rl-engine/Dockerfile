ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG FORGE_COMMIT=uncommitted
ARG LIGHTLLM_COMMIT=32359540f094520519a5da98c71541c5631b11f0
ARG LIGHTX2V_COMMIT=a46d6c4d0dc00659c597078f86fb34c1d5b0f481

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential git ca-certificates python3.12 python3.12-dev python3.12-venv \
      libibverbs1 ibverbs-providers librdmacm1 rdma-core \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/sensenova-forge-py312

ENV PATH=/opt/sensenova-forge-py312/bin:${PATH} \
    PYTHON_BIN=/opt/sensenova-forge-py312/bin/python \
    FORGE_RUNTIME_IMAGE=sensenova-u15-forge:unified-v2 \
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
