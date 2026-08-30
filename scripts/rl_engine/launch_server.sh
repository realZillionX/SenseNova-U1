#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SOURCE_ROOT=${SOURCE_ROOT:-$SCRIPT_ROOT}
MODEL_ROOT=${MODEL_ROOT:?set MODEL_ROOT to the SenseNova-U1.5-8B-MoT HF checkpoint}
LIGHTLLM_SOURCE_ROOT="$SOURCE_ROOT/serving/third_party/LightLLM"
LIGHTLLM_ROOT=${FORGE_LIGHTLLM_ROOT:-/opt/sensenova-forge/sources/LightLLM}
LIGHTX2V_ROOT="$SOURCE_ROOT/serving/third_party/LightX2V"
X2V_CONFIG=${X2V_CONFIG:-$SOURCE_ROOT/serving/configs/neopp_u15_forge_512.json}
PYTHON_BIN=${PYTHON_BIN:-/opt/sensenova-forge-py312/bin/python}

[[ -f "$X2V_CONFIG" ]] || { echo "Forge LightX2V config is missing: $X2V_CONFIG" >&2; exit 2; }
[[ -d "$LIGHTLLM_ROOT" ]] || {
  echo "Patched LightLLM runtime is missing: $LIGHTLLM_ROOT (run scripts/rl_engine/prepare_runtime.sh first)" >&2
  exit 2
}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$LIGHTLLM_ROOT:$LIGHTX2V_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export FORGE_RUNTIME_IMAGE=${FORGE_RUNTIME_IMAGE:-sensenova-u15-forge:unified-v2}
export FORGE_ROOT="$SOURCE_ROOT"
export FORGE_COMMIT=${FORGE_COMMIT:-$(git -C "$SOURCE_ROOT" rev-parse HEAD)}
export FORGE_LIGHTLLM_COMMIT=${FORGE_LIGHTLLM_COMMIT:-$(git -C "$LIGHTLLM_SOURCE_ROOT" rev-parse HEAD)}
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
# Transformers applies repetition penalty to prompt tokens as well. Keep
# ordinary VQA aligned while the RL route explicitly disables every penalty.
export INPUT_PENALTY=${INPUT_PENALTY:-true}
# The official VQA/interleave profile permits 8192 generated tokens, so the
# server must also leave room for its prompt.
MAX_REQ_TOTAL_LEN=${MAX_REQ_TOTAL_LEN:-16384}
# The H200-only runtime reserves twenty percent of HBM for online publication,
# CUDA graphs, and transient kernels while keeping a large KV cache.
LIGHTLLM_MEM_FRACTION=${LIGHTLLM_MEM_FRACTION:-0.80}
# Adaptive level 1 tunes only missing kernels during the existing warmup and
# reuses the selected config for steady-state serving.
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=${LIGHTLLM_TRITON_AUTOTUNE_LEVEL:-1}

"$PYTHON_BIN" "$SOURCE_ROOT/scripts/rl_engine/preflight.py" \
  --model-path "$MODEL_ROOT" \
  --expected-gpus 2 \
  --output "${PREFLIGHT_OUTPUT:-/tmp/sensenova-u15-forge-preflight.json}"

exec "$PYTHON_BIN" -m lightllm.server.api_server \
  --model_dir "$MODEL_ROOT" \
  --enable_multimodal_x2i \
  --x2i_server_deploy_mode separate \
  --x2i_server_used_gpus 1 \
  --x2v_gen_model_config "$X2V_CONFIG" \
  --host 0.0.0.0 \
  --port 8000 \
  --max_req_total_len "$MAX_REQ_TOTAL_LEN" \
  --mem_fraction "$LIGHTLLM_MEM_FRACTION" \
  --tp 1
