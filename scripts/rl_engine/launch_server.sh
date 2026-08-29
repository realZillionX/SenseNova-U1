#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SOURCE_ROOT=${SOURCE_ROOT:-$SCRIPT_ROOT}
MODEL_ROOT=${MODEL_ROOT:?set MODEL_ROOT to the SenseNova-U1.5-8B-MoT HF checkpoint}
LIGHTLLM_ROOT="$SOURCE_ROOT/serving/third_party/LightLLM"
LIGHTX2V_ROOT="$SOURCE_ROOT/serving/third_party/LightX2V"
PYTHON_BIN=${PYTHON_BIN:-/opt/sensenova-forge-py312/bin/python}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$LIGHTLLM_ROOT:$LIGHTX2V_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export FORGE_RUNTIME_IMAGE=${FORGE_RUNTIME_IMAGE:-sensenova-u15-forge:rl-serving-v1}
export FORGE_ROOT="$SOURCE_ROOT"
export FORGE_COMMIT=${FORGE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}
export FORGE_LIGHTLLM_COMMIT=${FORGE_LIGHTLLM_COMMIT:-$(git -C "$LIGHTLLM_ROOT" rev-parse HEAD)}
export FORGE_LIGHTX2V_COMMIT=${FORGE_LIGHTX2V_COMMIT:-$(git -C "$LIGHTX2V_ROOT" rev-parse HEAD)}
export FORGE_LIGHTLLM_ROOT="$LIGHTLLM_ROOT"
export FORGE_LIGHTX2V_ROOT="$LIGHTX2V_ROOT"
# The pinned serving engine reads these two transport aliases internally;
# Forge owns the public configuration names above them.
export MOVA_SENSENOVA_COMMIT="$FORGE_COMMIT"
export MOVA_LIGHTLLM_COMMIT="$FORGE_LIGHTLLM_COMMIT"
export MOVA_LIGHTX2V_COMMIT="$FORGE_LIGHTX2V_COMMIT"
export MOVA_IMAGE_DIGEST="$FORGE_RUNTIME_IMAGE"
export MOVA_RL_TRACE_DIR=${FORGE_RL_TRACE_DIR:-/dev/shm/sensenova_forge_rl_traces}
export MOVA_RL_TRACE_TTL=${FORGE_RL_TRACE_TTL:-3600}
MAX_REQ_TOTAL_LEN=${MAX_REQ_TOTAL_LEN:-8192}

"$PYTHON_BIN" "$SOURCE_ROOT/scripts/rl_engine/preflight.py" \
  --model-path "$MODEL_ROOT" \
  --expected-gpus 2 \
  --output "${PREFLIGHT_OUTPUT:-/tmp/sensenova-u15-forge-preflight.json}"

exec "$PYTHON_BIN" -m lightllm.server.api_server \
  --model_dir "$MODEL_ROOT" \
  --enable_multimodal_x2i \
  --x2i_server_deploy_mode separate \
  --x2i_server_used_gpus 1 \
  --x2v_gen_model_config "$LIGHTX2V_ROOT/configs/neopp/neopp_dense.json" \
  --host 0.0.0.0 \
  --port 8000 \
  --max_req_total_len "$MAX_REQ_TOTAL_LEN" \
  --mem_fraction 0.75 \
  --tp 1
