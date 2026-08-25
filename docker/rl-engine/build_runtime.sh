#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=${1:?usage: build_runtime.sh SOURCE_ROOT}
LIGHTLLM_ROOT="$SOURCE_ROOT/evaluation/easi/lightllm-stack/LightLLM"
LIGHTX2V_ROOT="$SOURCE_ROOT/evaluation/easi/lightllm-stack/LightX2V"
RUNTIME_LOCK="$SOURCE_ROOT/docker/rl-engine/requirements.lock"
RUNTIME_CONTRACT="$SOURCE_ROOT/docker/rl-engine/runtime_contract.json"
PYTHON_BIN=${PYTHON_BIN:-/opt/mostar-u1-py312/bin/python}
FA3_WHEEL=${FA3_WHEEL:-/tmp/flash_attn_3-3.0.0+20260817.cu128torch280cxx11abitrue.25110-cp310-abi3-linux_x86_64.whl}
FA3_URL=${FA3_URL:-https://github.com/windreamer/flash-attention3-wheels/releases/download/2026.08.17-542a34a/flash_attn_3-3.0.0%2B20260817.cu128torch280cxx11abitrue.25110-cp310-abi3-linux_x86_64.whl}
EXPECTED_FA3_SHA256=c5f5450f09a847415afaa2efbeff857ed9690e7001a1c0c09a1659e05f5b36c3
MANIFEST_DIR=/opt/mova-runtime-manifests/lightllm-x2v

for path in "$PYTHON_BIN" "$RUNTIME_LOCK" "$RUNTIME_CONTRACT"; do
  [[ -e "$path" ]] || { echo "required path is missing: $path" >&2; exit 2; }
done
for path in "$LIGHTLLM_ROOT" "$LIGHTX2V_ROOT"; do
  [[ -d "$path" ]] || { echo "required source root is missing: $path" >&2; exit 2; }
done

if ! "$PYTHON_BIN" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
try:
    assert version("flash-attn-3") == "3.0.0+20260817.cu128torch280cxx11abitrue.25110"
except (AssertionError, PackageNotFoundError):
    raise SystemExit(1)
PY
then
  "$PYTHON_BIN" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import json
import sys

expected = {
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "triton": "3.4.0",
    "transformers": "4.57.1",
    "tokenizers": "0.22.1",
    "huggingface-hub": "0.36.2",
    "numpy": "2.5.2",
    "protobuf": "7.35.1",
}
assert sys.version_info[:2] == (3, 12), sys.version
for distribution, wanted in expected.items():
    actual = version(distribution)
    if actual != wanted:
        raise SystemExit(f"base invariant failed: {distribution}={actual}, expected {wanted}")
try:
    installed_fa3 = version("flash-attn-3")
except PackageNotFoundError:
    installed_fa3 = None
if installed_fa3 is not None:
    raise SystemExit(f"unexpected FA3 version in builder: {installed_fa3}")
print(json.dumps({"base": expected, "status": "clean"}, sort_keys=True))
PY

  if [[ ! -f "$FA3_WHEEL" ]] || [[ "$(sha256sum "$FA3_WHEEL" | awk '{print $1}')" != "$EXPECTED_FA3_SHA256" ]]; then
    unlink "$FA3_WHEEL" 2>/dev/null || true
    curl --fail --location --retry 5 --output "$FA3_WHEEL" "$FA3_URL"
  fi
  [[ "$(sha256sum "$FA3_WHEEL" | awk '{print $1}')" == "$EXPECTED_FA3_SHA256" ]] || {
    echo "FA3 wheel digest mismatch" >&2
    exit 2
  }

  # This is intentionally the only package-install command in the build.
  "$PYTHON_BIN" -m pip install -r "$RUNTIME_LOCK" "$FA3_WHEEL"
fi

pip_check_status=0
pip_check_output="$("$PYTHON_BIN" -m pip check 2>&1)" || pip_check_status=$?
if [[ "$pip_check_status" != 0 ]]; then
  if [[ "$pip_check_output" != "decord 0.6.0 is not supported on this platform" ]]; then
    echo "$pip_check_output" >&2
    exit "$pip_check_status"
  fi
  echo "accepted inherited v4 metadata issue: $pip_check_output"
else
  echo "$pip_check_output"
fi

PYTHONPATH="$LIGHTX2V_ROOT:$LIGHTLLM_ROOT" "$PYTHON_BIN" - <<'PY'
from importlib.metadata import version
from importlib.util import find_spec

import av
import pandas
import torchaudio

assert version("flash-attn-3") == "3.0.0+20260817.cu128torch280cxx11abitrue.25110"
for module in (
    "cupy",
    "flashinfer",
    "flash_attn_interface",
    "lightllm",
    "lightx2v",
    "sgl_kernel",
):
    assert find_spec(module) is not None, module
print("CPU metadata and non-CUDA import closure: ok")
PY

install -d -m 0755 "$MANIFEST_DIR"
install -m 0644 "$RUNTIME_LOCK" "$MANIFEST_DIR/requirements.lock"
install -m 0644 "$RUNTIME_CONTRACT" "$MANIFEST_DIR/runtime_contract.json"
"$PYTHON_BIN" -m pip freeze >"$MANIFEST_DIR/pip-freeze.txt"
sha256sum "$RUNTIME_LOCK" "$RUNTIME_CONTRACT" >"$MANIFEST_DIR/input-sha256.txt"

unlink "$FA3_WHEEL" 2>/dev/null || true
"$PYTHON_BIN" -m pip cache purge || true

echo "MOVA LightLLM + LightX2V runtime build completed"
