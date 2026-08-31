#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=${1:?usage: prepare_runtime.sh SOURCE_ROOT}
LIGHTLLM_SOURCE_ROOT="$SOURCE_ROOT/serving/third_party/LightLLM"
LIGHTLLM_ROOT=${FORGE_LIGHTLLM_ROOT:-/opt/sensenova-forge/sources/LightLLM}
LIGHTX2V_ROOT="$SOURCE_ROOT/serving/third_party/LightX2V"
PYTHON_BIN=${PYTHON_BIN:-/opt/sensenova-forge-py312/bin/python}
PYTHON_BOOTSTRAP=${PYTHON_BOOTSTRAP:-python3.12}

if [[ ! -x "$PYTHON_BIN" ]]; then
  command -v "$PYTHON_BOOTSTRAP" >/dev/null || {
    echo "Python 3.12 bootstrap interpreter is unavailable: $PYTHON_BOOTSTRAP" >&2
    exit 2
  }
  "$PYTHON_BOOTSTRAP" -m venv "$(dirname "$(dirname "$PYTHON_BIN")")"
fi

export FORGE_ROOT=$SOURCE_ROOT
export FORGE_LIGHTLLM_ROOT="$LIGHTLLM_ROOT"
export FORGE_LIGHTX2V_ROOT=$LIGHTX2V_ROOT
export FORGE_COMMIT=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
export FORGE_LIGHTLLM_COMMIT=$(git -C "$LIGHTLLM_SOURCE_ROOT" rev-parse HEAD)
export FORGE_LIGHTX2V_COMMIT=$(git -C "$LIGHTX2V_ROOT" rev-parse HEAD)
export FORGE_RUNTIME_IMAGE=${FORGE_RUNTIME_IMAGE_OVERRIDE:-sensenova-u15-forge:unified-v4}

bash "$SOURCE_ROOT/docker/rl-engine/build_runtime.sh" "$SOURCE_ROOT"

export PYTHONPATH="$LIGHTLLM_ROOT:$LIGHTX2V_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" "$SOURCE_ROOT/scripts/rl_engine/preflight.py" \
  --allow-no-gpu \
  --x2v-config "$SOURCE_ROOT/serving/configs/neopp_u15_forge_512.json"
