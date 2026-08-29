#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=${1:?usage: prepare_runtime.sh SOURCE_ROOT}
LIGHTLLM_ROOT="$SOURCE_ROOT/serving/third_party/LightLLM"
LIGHTX2V_ROOT="$SOURCE_ROOT/serving/third_party/LightX2V"
PYTHON_BIN=${PYTHON_BIN:-/opt/sensenova-forge-py312/bin/python}

bash "$SOURCE_ROOT/docker/rl-engine/build_runtime.sh" "$SOURCE_ROOT"

export PYTHONPATH="$LIGHTLLM_ROOT:$LIGHTX2V_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" "$SOURCE_ROOT/scripts/rl_engine/preflight.py" --allow-no-gpu
