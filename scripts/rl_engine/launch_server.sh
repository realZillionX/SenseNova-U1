#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=${SOURCE_ROOT:-/inspire/hdd/global_user/liangtianyi-253208120278/train-asset/public/zlwang/MOVA-SenseNova-U1-lightllm-x2v-rl-engine-20260825}
MODEL_ROOT=${MODEL_ROOT:-/inspire/hdd/global_user/liangtianyi-253208120278/train-asset/models/SenseNova-U1.5-8B-MoT}
LIGHTLLM_ROOT="$SOURCE_ROOT/evaluation/easi/lightllm-stack/LightLLM"
LIGHTX2V_ROOT="$SOURCE_ROOT/evaluation/easi/lightllm-stack/LightX2V"
PYTHON_BIN=${PYTHON_BIN:-/opt/mostar-u1-py312/bin/python}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$LIGHTLLM_ROOT:$LIGHTX2V_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MOVA_PLATFORM_PARENT_IMAGE=${MOVA_PLATFORM_PARENT_IMAGE:-mostar-u1-runtime:v4}
export MOVA_IMAGE_DIGEST=${MOVA_IMAGE_DIGEST:-inspire:mova-u15-lightllm-x2v-rl:v2}
export MOVA_SENSENOVA_COMMIT=${MOVA_SENSENOVA_COMMIT:-34ca2f66e7006489a0184eb8896a75f4081a0257}
export MOVA_LIGHTLLM_COMMIT=${MOVA_LIGHTLLM_COMMIT:-a23a382c70ea7e3c31b2fafcd546c66c99c56fef}
export MOVA_LIGHTX2V_COMMIT=${MOVA_LIGHTX2V_COMMIT:-f453c1ef22d1be21c76d186816c3aa6ad5c135c5}
export MOVA_LIGHTLLM_ROOT="$LIGHTLLM_ROOT"
export MOVA_LIGHTX2V_ROOT="$LIGHTX2V_ROOT"
export MOVA_RL_TRACE_DIR=${MOVA_RL_TRACE_DIR:-/tmp/mova_rl_traces}
export MOVA_RL_TRACE_TTL=${MOVA_RL_TRACE_TTL:-3600}

"$PYTHON_BIN" "$SOURCE_ROOT/scripts/rl_engine/preflight.py" \
  --model-path "$MODEL_ROOT" \
  --expected-gpus 2 \
  --output /tmp/mova-u15-rl-preflight.json

exec "$PYTHON_BIN" -m lightllm.server.api_server \
  --model_dir "$MODEL_ROOT" \
  --enable_multimodal_x2i \
  --x2i_server_deploy_mode separate \
  --x2i_server_used_gpus 1 \
  --x2v_gen_model_config "$LIGHTX2V_ROOT/configs/neopp/neopp_dense.json" \
  --host 0.0.0.0 \
  --port 8000 \
  --max_req_total_len 65536 \
  --mem_fraction 0.75 \
  --tp 1
