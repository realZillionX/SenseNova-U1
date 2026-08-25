#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=${SOURCE_ROOT:-/inspire/hdd/global_user/liangtianyi-253208120278/train-asset/public/zlwang/MOVA-SenseNova-U1-lightllm-x2v-rl-engine-20260825}
OUT_ROOT=${OUT_ROOT:-/tmp/mova-u15-claim-smoke}
PYTHON_BIN=${PYTHON_BIN:-/opt/mostar-u1-py312/bin/python}
mkdir -p "$OUT_ROOT"

"$PYTHON_BIN" "$SOURCE_ROOT/examples/serving/client.py" \
  --mode vqa \
  --prompt "What is 17 + 25? Answer briefly." \
  --url http://127.0.0.1:8000/v1 \
  --out-dir "$OUT_ROOT/ti2t"

"$PYTHON_BIN" "$SOURCE_ROOT/examples/serving/client.py" \
  --mode interleave \
  --prompt "Create an image of a red cube on a white table, then describe it." \
  --url http://127.0.0.1:8000/v1 \
  --out-dir "$OUT_ROOT/ti2ti" \
  --height 512 \
  --width 512 \
  --max-tokens 1024
