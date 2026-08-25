#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=${1:?usage: prepare_runtime.sh SOURCE_ROOT}
LIGHTLLM_ROOT="$SOURCE_ROOT/evaluation/easi/lightllm-stack/LightLLM"
LIGHTX2V_ROOT="$SOURCE_ROOT/evaluation/easi/lightllm-stack/LightX2V"

python -m pip install --no-cache-dir --no-deps \
  -r "$SOURCE_ROOT/docker/rl-engine/requirements.lock"
python -m pip install --no-cache-dir --no-deps --no-build-isolation -e "$LIGHTLLM_ROOT"
python -m pip install --no-cache-dir --no-deps --no-build-isolation -e "$LIGHTX2V_ROOT"

export PYTHONPATH="$LIGHTLLM_ROOT:$LIGHTX2V_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python "$SOURCE_ROOT/scripts/rl_engine/preflight.py" --allow-no-gpu
