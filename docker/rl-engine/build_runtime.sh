#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=${1:?usage: build_runtime.sh SOURCE_ROOT}
LIGHTLLM_SOURCE_ROOT="$SOURCE_ROOT/serving/third_party/LightLLM"
LIGHTLLM_ROOT=${FORGE_LIGHTLLM_ROOT:-/opt/sensenova-forge/sources/LightLLM}
LIGHTX2V_ROOT="$SOURCE_ROOT/serving/third_party/LightX2V"
LIGHTLLM_OVERLAY="$SOURCE_ROOT/serving/patches/lightllm-sensenova-policy.patch"
RUNTIME_LOCK="$SOURCE_ROOT/docker/rl-engine/requirements.lock"
RUNTIME_CONTRACT="$SOURCE_ROOT/docker/rl-engine/runtime_contract.json"
PYTHON_BIN=${PYTHON_BIN:-/opt/sensenova-forge-py312/bin/python}
FA3_NEO_REPOSITORY=${FA3_NEO_REPOSITORY:-https://github.com/WANDY666/flash-attention.git}
FA3_NEO_COMMIT=e2077ee6e568e64d0d01c6b44d8ce4ee24e7932b
MANIFEST_DIR=/opt/sensenova-forge/manifests/rl-serving

export FORGE_ROOT=${FORGE_ROOT:-$SOURCE_ROOT}
export PATH="/usr/local/cuda/bin:$(dirname "$PYTHON_BIN"):$PATH"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PYTHONPATH="$SOURCE_ROOT/src:$LIGHTLLM_ROOT:$LIGHTX2V_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for path in "$PYTHON_BIN" "$RUNTIME_LOCK" "$RUNTIME_CONTRACT" "$LIGHTLLM_SOURCE_ROOT" "$LIGHTX2V_ROOT" "$LIGHTLLM_OVERLAY"; do
  [[ -e "$path" ]] || { echo "required Forge runtime input is missing: $path" >&2; exit 2; }
done

if [[ "$LIGHTLLM_ROOT" == "$LIGHTLLM_SOURCE_ROOT" ]]; then
  echo "FORGE_LIGHTLLM_ROOT must be an isolated runtime directory" >&2
  exit 2
fi
install -d -m 0755 "$(dirname "$LIGHTLLM_ROOT")" "$LIGHTLLM_ROOT"
(
  cd "$LIGHTLLM_SOURCE_ROOT"
  tar --exclude=.git -cf - .
) | (
  cd "$LIGHTLLM_ROOT"
  tar -xf -
)

"$PYTHON_BIN" - "$RUNTIME_CONTRACT" "$LIGHTLLM_OVERLAY" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = contract["overlays"]["lightllm_sensenova_policy"]["sha256"]
observed = hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
if observed != expected:
    raise SystemExit(f"LightLLM overlay identity mismatch: {observed} != {expected}")
PY

if git -C "$LIGHTLLM_ROOT" apply --reverse --check "$LIGHTLLM_OVERLAY" 2>/dev/null; then
  echo "SenseNova LightLLM policy overlay is already applied"
elif git -C "$LIGHTLLM_ROOT" apply --check "$LIGHTLLM_OVERLAY"; then
  git -C "$LIGHTLLM_ROOT" apply "$LIGHTLLM_OVERLAY"
else
  echo "SenseNova LightLLM policy overlay does not match the pinned source" >&2
  exit 2
fi

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel ninja packaging psutil
"$PYTHON_BIN" -m pip install -r "$RUNTIME_LOCK"
"$PYTHON_BIN" -m pip install -e "$SOURCE_ROOT" --no-deps

if ! "$PYTHON_BIN" - <<'PY'
from importlib.metadata import version
import flash_attn

assert version("flash-attn") == "2.8.3"
assert callable(flash_attn.flash_attn_func)
PY
then
  MAX_JOBS=${MAX_JOBS:-8} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0} \
    "$PYTHON_BIN" -m pip install --no-build-isolation --no-deps "flash-attn==2.8.3"
fi

if ! "$PYTHON_BIN" - <<'PY'
from importlib.metadata import version
import inspect
from flash_attn_interface import flash_attn_with_kvcache

assert version("flash-attn-3") == "3.0.0"
assert "image_token_end" in inspect.signature(flash_attn_with_kvcache).parameters
PY
then
  fa3_build_dir="$(mktemp -d /tmp/sensenova-forge-fa3.XXXXXX)"
  cleanup_fa3_build() {
    case "$fa3_build_dir" in
      /tmp/sensenova-forge-fa3.*) rm -rf -- "$fa3_build_dir" ;;
      *) echo "refusing to remove unexpected FA3 build path: $fa3_build_dir" >&2 ;;
    esac
  }
  trap cleanup_fa3_build EXIT
  git -C "$fa3_build_dir" init -q src
  git -C "$fa3_build_dir/src" remote add origin "$FA3_NEO_REPOSITORY"
  git -C "$fa3_build_dir/src" fetch --depth 1 origin "$FA3_NEO_COMMIT"
  git -C "$fa3_build_dir/src" checkout -q --detach FETCH_HEAD
  git -C "$fa3_build_dir/src" submodule update --init --depth 1 csrc/cutlass
  (
    cd "$fa3_build_dir/src/hopper"
    export MAX_JOBS=${MAX_JOBS:-8}
    export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
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

"$PYTHON_BIN" -m pip check
SKIP_PLATFORM_CHECK=1 "$PYTHON_BIN" - <<'PY'
from importlib.metadata import version
from importlib.util import find_spec
import inspect

import torch
from flash_attn_interface import flash_attn_with_kvcache
from lightx2v.pipeline import _ensure_runner_registered
from lightx2v.utils.registry_factory import RUNNER_REGISTER

assert torch.__version__.startswith("2.8.0")
assert torch.version.cuda == "12.8"
assert version("transformers") == "4.57.1"
assert version("flash-attn") == "2.8.3"
assert version("flash-attn-3") == "3.0.0"
assert "image_token_end" in inspect.signature(flash_attn_with_kvcache).parameters
for module in ("lightllm", "lightx2v", "flashinfer", "sgl_kernel", "sensenova_u1"):
    assert find_spec(module) is not None, module
_ensure_runner_registered("neopp")
assert "neopp" in RUNNER_REGISTER
print("Forge RL + serving import closure: ok")
PY

install -d -m 0755 "$MANIFEST_DIR"
install -m 0644 "$RUNTIME_LOCK" "$MANIFEST_DIR/requirements.lock"
install -m 0644 "$RUNTIME_CONTRACT" "$MANIFEST_DIR/runtime_contract.json"
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

commits = {
    "forge": os.environ["FORGE_COMMIT"],
    "lightllm": os.environ["FORGE_LIGHTLLM_COMMIT"],
    "lightx2v": os.environ["FORGE_LIGHTX2V_COMMIT"],
}
contract = json.loads(Path(os.environ["FORGE_ROOT"], "docker/rl-engine/runtime_contract.json").read_text())
commits["lightllm_overlay_sha256"] = contract["overlays"]["lightllm_sensenova_policy"]["sha256"]
invalid = {
    name: value
    for name, value in commits.items()
    if value in {"", "unknown", "UNSET", "uncommitted"}
}
if invalid:
    raise SystemExit(f"runtime source commits must be immutable: {invalid}")
Path("/opt/sensenova-forge/manifests/rl-serving/source-commits.json").write_text(
    json.dumps(commits, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
"$PYTHON_BIN" -m pip freeze >"$MANIFEST_DIR/pip-freeze.txt"
sha256sum "$RUNTIME_LOCK" "$RUNTIME_CONTRACT" >"$MANIFEST_DIR/input-sha256.txt"
"$PYTHON_BIN" "$SOURCE_ROOT/scripts/rl_engine/preflight.py" \
  --allow-no-gpu \
  --x2v-config "$SOURCE_ROOT/serving/configs/neopp_u15_forge_rl.json"
"$PYTHON_BIN" -m pip cache purge || true

echo "SenseNova-U1.5 Forge RL + serving runtime completed"
