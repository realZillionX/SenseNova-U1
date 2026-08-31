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

if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]; then
  GPU_COUNT=$(
    "$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())'
  )
  [[ "$GPU_COUNT" =~ ^[0-9]+$ ]] || {
    echo "could not discover the visible CUDA device count" >&2
    exit 2
  }
  CUDA_VISIBLE_DEVICES=""
  for ((device = 0; device < GPU_COUNT; device++)); do
    CUDA_VISIBLE_DEVICES+="${CUDA_VISIBLE_DEVICES:+,}$device"
  done
fi
export CUDA_VISIBLE_DEVICES
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$LIGHTLLM_ROOT:$LIGHTX2V_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export FORGE_RUNTIME_IMAGE=${FORGE_RUNTIME_IMAGE_OVERRIDE:-sensenova-u15-forge:unified-v4}
export FORGE_ROOT="$SOURCE_ROOT"
export FORGE_COMMIT=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
export FORGE_LIGHTLLM_COMMIT=$(git -C "$LIGHTLLM_SOURCE_ROOT" rev-parse HEAD)
export FORGE_LIGHTX2V_COMMIT=$(git -C "$LIGHTX2V_ROOT" rev-parse HEAD)
export FORGE_LIGHTLLM_ROOT="$LIGHTLLM_ROOT"
export FORGE_LIGHTX2V_ROOT="$LIGHTX2V_ROOT"
# The pinned serving engine reads these two transport aliases internally;
# Forge owns the public configuration names above them.
export MOVA_SENSENOVA_COMMIT="$FORGE_COMMIT"
export MOVA_LIGHTLLM_COMMIT="$FORGE_LIGHTLLM_COMMIT"
export MOVA_LIGHTX2V_COMMIT="$FORGE_LIGHTX2V_COMMIT"
export MOVA_IMAGE_DIGEST="$FORGE_RUNTIME_IMAGE"
TRACE_ROOT=${FORGE_RL_TRACE_DIR:-/dev/shm/sensenova_forge_rl_traces}
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

IFS=',' read -r -a VISIBLE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
for device in "${VISIBLE_GPUS[@]}"; do
  [[ "$device" =~ ^[0-9]+$ ]] || {
    echo "CUDA_VISIBLE_DEVICES must be a comma-separated physical GPU index list" >&2
    exit 2
  }
done
(( ${#VISIBLE_GPUS[@]} >= 2 && ${#VISIBLE_GPUS[@]} % 2 == 0 )) || {
  echo "SenseNova serving requires an even number of GPUs (LightLLM, LightX2V pairs)" >&2
  exit 2
}
AVAILABLE_REPLICAS=$((${#VISIBLE_GPUS[@]} / 2))
REPLICA_COUNT=${FORGE_SERVING_REPLICAS:-$AVAILABLE_REPLICAS}
[[ "$REPLICA_COUNT" =~ ^[0-9]+$ ]] || {
  echo "FORGE_SERVING_REPLICAS must be an integer" >&2
  exit 2
}
(( REPLICA_COUNT >= 1 && REPLICA_COUNT <= AVAILABLE_REPLICAS )) || {
  echo "FORGE_SERVING_REPLICAS must be in [1, $AVAILABLE_REPLICAS]" >&2
  exit 2
}
PORT_BASE=${FORGE_SERVING_PORT_BASE:-8000}
REPLICA_ID_OFFSET=${FORGE_SERVING_REPLICA_ID_OFFSET:-0}
[[ "$PORT_BASE" =~ ^[0-9]+$ && "$REPLICA_ID_OFFSET" =~ ^[0-9]+$ ]] || {
  echo "FORGE_SERVING_PORT_BASE and FORGE_SERVING_REPLICA_ID_OFFSET must be non-negative integers" >&2
  exit 2
}
(( PORT_BASE >= 1 && PORT_BASE + REPLICA_COUNT - 1 <= 65535 )) || {
  echo "Forge serving replica HTTP ports fall outside [1, 65535]" >&2
  exit 2
}
launch_replica() {
  local local_index=$1
  local replica_id=$((REPLICA_ID_OFFSET + local_index))
  local port=$((PORT_BASE + local_index))
  local device_pair="${VISIBLE_GPUS[$((2 * local_index))]},${VISIBLE_GPUS[$((2 * local_index + 1))]}"
  local preflight_output
  if (( REPLICA_COUNT == 1 )) && [[ -n ${PREFLIGHT_OUTPUT:-} ]]; then
    preflight_output=$PREFLIGHT_OUTPUT
  else
    preflight_output="${PREFLIGHT_OUTPUT_DIR:-/tmp}/sensenova-u15-forge-preflight-r${replica_id}.json"
  fi
  export CUDA_VISIBLE_DEVICES=$device_pair
  export MOVA_RL_REPLICA_ID=$replica_id
  export MOVA_RL_LOCAL_REPLICA_ID=$local_index
  export MOVA_RL_TRACE_DIR="$TRACE_ROOT/replica-$replica_id"
  local rdma_args=()
  if [[ ${FORGE_REQUIRE_RDMA:-false} == true ]]; then
    rdma_args+=(--require-rdma)
  fi
  "$PYTHON_BIN" "$SOURCE_ROOT/scripts/rl_engine/preflight.py" \
    --model-path "$MODEL_ROOT" \
    --x2v-config "$X2V_CONFIG" \
    --expected-gpus 2 \
    "${rdma_args[@]}" \
    --output "$preflight_output"
  echo "starting Forge serving replica=$replica_id GPUs=$device_pair port=$port" >&2
  exec "$PYTHON_BIN" -m lightllm.server.api_server \
    --model_dir "$MODEL_ROOT" \
    --enable_multimodal_x2i \
    --x2i_server_deploy_mode separate \
    --x2i_server_used_gpus 1 \
    --x2v_gen_model_config "$X2V_CONFIG" \
    --host 0.0.0.0 \
    --port "$port" \
    --max_req_total_len "$MAX_REQ_TOTAL_LEN" \
    --mem_fraction "$LIGHTLLM_MEM_FRACTION" \
    --tp 1
}

PIDS=()
for ((replica = 0; replica < REPLICA_COUNT; replica++)); do
  launch_replica "$replica" &
  PIDS+=("$!")
done
terminate_replicas() {
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap terminate_replicas INT TERM EXIT
set +e
wait -n "${PIDS[@]}"
status=$?
set -e
terminate_replicas
for pid in "${PIDS[@]}"; do
  wait "$pid" 2>/dev/null || true
done
trap - INT TERM EXIT
exit "$status"
