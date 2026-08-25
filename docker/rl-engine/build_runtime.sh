#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=${1:?usage: build_runtime.sh SOURCE_ROOT}
LIGHTLLM_ROOT="$SOURCE_ROOT/evaluation/easi/lightllm-stack/LightLLM"
LIGHTX2V_ROOT="$SOURCE_ROOT/evaluation/easi/lightllm-stack/LightX2V"
RUNTIME_LOCK="$SOURCE_ROOT/docker/rl-engine/requirements.lock"
RUNTIME_CONTRACT="$SOURCE_ROOT/docker/rl-engine/runtime_contract.json"
PYTHON_BIN=${PYTHON_BIN:-/opt/mostar-u1-py312/bin/python}
FA3_NEO_REPOSITORY=${FA3_NEO_REPOSITORY:-https://github.com/WANDY666/flash-attention.git}
FA3_NEO_BRANCH=${FA3_NEO_BRANCH:-support_neo}
FA3_NEO_COMMIT=e2077ee6e568e64d0d01c6b44d8ce4ee24e7932b
MANIFEST_DIR=/opt/mova-runtime-manifests/lightllm-x2v
export PATH="/usr/local/cuda/bin:$(dirname "$PYTHON_BIN"):$PATH"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

# The CUDA toolkit in the base image provides nvcc, while the matching CUDA
# development headers and libraries installed with Torch live under the
# environment's nvidia namespace packages.  Expose the entire closure to nvcc
# instead of reinstalling a second CUDA SDK into the image.
PYTHON_SITE_PACKAGES="$($PYTHON_BIN - <<'PY'
import site

paths = site.getsitepackages()
if not paths:
    raise SystemExit("Python environment has no site-packages directory")
print(paths[0])
PY
)"
NVIDIA_ROOT="$PYTHON_SITE_PACKAGES/nvidia"
cuda_include_paths=("$CUDA_HOME/include")
cuda_library_paths=("$CUDA_HOME/lib64")
for component in cuda_runtime cublas cusolver cusparse cufft curand nvjitlink nccl; do
  [[ -d "$NVIDIA_ROOT/$component/include" ]] && cuda_include_paths+=("$NVIDIA_ROOT/$component/include")
  [[ -d "$NVIDIA_ROOT/$component/lib" ]] && cuda_library_paths+=("$NVIDIA_ROOT/$component/lib")
done
for required_cuda_path in \
  "$CUDA_HOME/bin/nvcc" \
  "$NVIDIA_ROOT/cusolver/include/cusolverDn.h" \
  "$NVIDIA_ROOT/cusolver/lib/libcusolver.so.11"; do
  [[ -e "$required_cuda_path" ]] || {
    echo "required CUDA build dependency is missing: $required_cuda_path" >&2
    exit 2
  }
done
cuda_include_path="$(IFS=:; echo "${cuda_include_paths[*]}")"
cuda_library_path="$(IFS=:; echo "${cuda_library_paths[*]}")"
export CPATH="$cuda_include_path${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$cuda_include_path${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$cuda_library_path${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$cuda_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

for path in "$PYTHON_BIN" "$RUNTIME_LOCK" "$RUNTIME_CONTRACT"; do
  [[ -e "$path" ]] || { echo "required path is missing: $path" >&2; exit 2; }
done
for path in "$LIGHTLLM_ROOT" "$LIGHTX2V_ROOT"; do
  [[ -d "$path" ]] || { echo "required source root is missing: $path" >&2; exit 2; }
done

if ! "$PYTHON_BIN" - <<'PY'
import inspect
from importlib.metadata import PackageNotFoundError, version

try:
    if version("flash-attn-3") != "3.0.0":
        raise SystemExit(1)
    from flash_attn_interface import flash_attn_with_kvcache
except (ImportError, PackageNotFoundError):
    raise SystemExit(1)
if "image_token_end" not in inspect.signature(flash_attn_with_kvcache).parameters:
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
    "nvidia-cuda-runtime-cu12": "12.8.90",
    "nvidia-cusolver-cu12": "11.7.3.90",
    "nvidia-nccl-cu12": "2.27.3",
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
print(json.dumps({"base": expected, "replacing_fa3": installed_fa3}, sort_keys=True))
PY

  # Resolve the complete Python environment once, without allowing the
  # LightLLM requirements to replace the pinned Torch/CUDA core.
  "$PYTHON_BIN" -m pip install -r "$RUNTIME_LOCK"

  # NEO-Unify prefill is not standard causal attention.  The official
  # SenseNova inference documentation pins this FA3 fork because its Hopper
  # kernel accepts image_token_tag.  A stock FA3 wheel is not equivalent.
  fa3_build_dir="$(mktemp -d /tmp/mova-fa3-neo-build.XXXXXX)"
  cleanup_fa3_build() {
    case "$fa3_build_dir" in
      /tmp/mova-fa3-neo-build.*) rm -rf -- "$fa3_build_dir" ;;
      *) echo "refusing to remove unexpected FA3 build path: $fa3_build_dir" >&2 ;;
    esac
  }
  trap cleanup_fa3_build EXIT
  git -C "$fa3_build_dir" init -q src
  git -C "$fa3_build_dir/src" remote add origin "$FA3_NEO_REPOSITORY"
  git -C "$fa3_build_dir/src" fetch --depth 1 origin "$FA3_NEO_COMMIT"
  git -C "$fa3_build_dir/src" checkout -q --detach FETCH_HEAD
  [[ "$(git -C "$fa3_build_dir/src" rev-parse HEAD)" == "$FA3_NEO_COMMIT" ]] || {
    echo "FA3-Neo source commit mismatch" >&2
    exit 2
  }
  git -C "$fa3_build_dir/src" submodule update --init --depth 1 csrc/cutlass
  (
    cd "$fa3_build_dir/src/hopper"
    export MAX_JOBS=${MAX_JOBS:-8}
    export TORCH_CUDA_ARCH_LIST=9.0
    export FLASH_ATTENTION_FORCE_BUILD=TRUE
    export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE
    export FLASH_ATTENTION_DISABLE_SM80=TRUE
    export FLASH_ATTENTION_DISABLE_SPLIT=TRUE
    export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
    export FLASH_ATTENTION_DISABLE_FP16=TRUE
    export FLASH_ATTENTION_DISABLE_FP8=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM64=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
    export FLASH_ATTENTION_DISABLE_HDIM256=TRUE
    export FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE
    export FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE
    "$PYTHON_BIN" -m pip install --force-reinstall --no-build-isolation --no-deps .
  )
  cleanup_fa3_build
  trap - EXIT
fi

pip_check_status=0
pip_check_output="$("$PYTHON_BIN" -m pip check 2>&1)" || pip_check_status=$?
if [[ "$pip_check_status" != 0 ]]; then
  if [[ "$pip_check_output" != "decord 0.6.0 is not supported on this platform" ]]; then
    echo "$pip_check_output" >&2
    exit "$pip_check_status"
  fi
  echo "accepted inherited base metadata issue in v2 runtime: $pip_check_output"
else
  echo "$pip_check_output"
fi

SKIP_PLATFORM_CHECK=1 PYTHONPATH="$LIGHTX2V_ROOT:$LIGHTLLM_ROOT" "$PYTHON_BIN" - <<'PY'
from importlib.metadata import version
from importlib.util import find_spec

import inspect

import av
import pandas
import torchaudio

assert version("flash-attn-3") == "3.0.0"
for module in (
    "cupy",
    "flashinfer",
    "flash_attn_interface",
    "lightllm",
    "lightx2v",
    "sgl_kernel",
):
    assert find_spec(module) is not None, module
from flash_attn_interface import flash_attn_with_kvcache

assert "image_token_end" in inspect.signature(flash_attn_with_kvcache).parameters
from lightx2v.pipeline import _ensure_runner_registered
from lightx2v.utils.registry_factory import RUNNER_REGISTER

_ensure_runner_registered("neopp")
assert "neopp" in RUNNER_REGISTER
print("CPU metadata and actual NeoPP import closure: ok")
PY

install -d -m 0755 "$MANIFEST_DIR"
install -m 0644 "$RUNTIME_LOCK" "$MANIFEST_DIR/requirements.lock"
install -m 0644 "$RUNTIME_CONTRACT" "$MANIFEST_DIR/runtime_contract.json"
"$PYTHON_BIN" -m pip freeze >"$MANIFEST_DIR/pip-freeze.txt"
sha256sum "$RUNTIME_LOCK" "$RUNTIME_CONTRACT" >"$MANIFEST_DIR/input-sha256.txt"

"$PYTHON_BIN" -m pip cache purge || true

echo "MOVA LightLLM + LightX2V runtime build completed"
