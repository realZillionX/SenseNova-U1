# TI2T/TI2TI RL engine integration

This integration keeps the existing `/v1/chat/completions` implementation and
adds a rollout/control plane around the LightLLM + LightX2V serving stack. It
does not contain a trainer, verifier, optimizer, or checkpoint manager; the
formal trainer and live-FSDP weight publisher belong to MOSTAR.

## Runtime pins

- SenseNova integration source: this repository checkout plus its exact
  submodule gitlinks; image-bundled source is not a substitute.
- LightLLM fork: `expectqwq/LightLLM@mova/sensenova-u15-rl-engine`, pinned by
  the parent gitlink.
- LightX2V fork: `expectqwq/LightX2V@mova/sensenova-u15-rl-engine`, pinned by
  the parent gitlink.
- requested upstream image:
  `lightx2v/lightllm_lightx2v:20260407@sha256:bb1900389c320b37dbcfe51fdf4db76a198d38a10c4c80d8b9b0726f1fb43ac7`

The Inspire registry cannot resolve the external digest directly. Environment
construction therefore happens once in the CPU Notebook
`mova-u15-lightllm-x2v-rl-build`: `mostar-u1-runtime:v4` is used only as the
Torch 2.8.0/CUDA 12.8 bootstrap layer, the fully resolved 122-entry lock is
installed once, and the pinned FA3-Neo source is compiled against that exact
environment. The result is saved as `mova-u15-lightllm-x2v-rl:v3`. H200
validation must use that saved image, not v4 and not an ad-hoc virtual
environment.

The build is intentionally reproducible and fail-fast:

- interpreter: `/opt/mostar-u1-py312/bin/python`;
- official NEO-Unify FA3 fork:
  `WANDY666/flash-attention@e2077ee6e568e64d0d01c6b44d8ce4ee24e7932b`
  (`support_neo`, with scoped `image_token_end`);
- package lock and runtime contract are copied to
  `/opt/mova-runtime-manifests/lightllm-x2v`;
- `pip check`, CPU-side import closure, exact package versions, source commits,
  CUDA/NCCL and FA3 are recorded by preflight. A stock FA3 wheel or the Neo
  Triton fallback fails preflight; both LightLLM and LightX2V must resolve the
  pinned FA3-Neo runtime.

## Rollout route

`POST /v1/rl/rollouts` accepts one prompt and a list of seeds. The seeds are
submitted concurrently so LightLLM can continuous-batch their text spans.
The first milestone intentionally leaves LightX2V image work serial.

TI2T masks the model's image-action token. TI2TI preserves that token in the
text policy trace, invokes the official LightX2V handoff, re-encodes the
generated image, and resumes text generation. `max_new_tokens` is shared by
all text spans in the complete interleaved trajectory.

Each text event contains token IDs, selected-token log-probabilities, response
mask, stop token, and decoded text. Each image event contains the final image,
SDE geometry, and a trace bundle ID. Trace bundles are short-lived
safetensors files under `/dev/shm/mova_rl_traces` and are exposed through:

- `GET /v1/rl/traces/{bundle_id}`
- `DELETE /v1/rl/traces/{bundle_id}`
- `WS /v1/rl/traces/ws` for bounded multi-bundle trainer streaming and cleanup

The RL-only image path disables CFG and applies the request's `t_eps`,
`timestep_shift`, noise level, and selected SDE window. The ordinary
LightX2V `generate()` path and its CFG behavior are unchanged.

## Policy update barrier

The HTTP controller stops new full-request admission and drains existing
text/image trajectories. Language, vision, and NeoPP consumers then join
separate publisher-to-consumer NCCL process groups and validate tensor names,
shapes, dtypes, their own model-derived parameter closure, and the pending
policy version before applying an in-place update. Active policy version
changes only after all three consumers ACK and relevant caches are cleared. A
failure leaves the service paused.

The formal MOSTAR publisher consumes live FSDP2 DTensor shards on the training
GPUs. Language, vision, and X2V each use a persistent training-side NCCL group;
their owner buckets are all-gathered concurrently, rebuilt one bucket at a time
on training rank 0, and immediately broadcast GPU-to-GPU over the corresponding
external NCCL group. The online path does not build a full CPU state dict, read
safetensors from disk, stage payloads through CPU memory, calculate payload
SHA256, or read updated weights back for value comparison. Metadata closure,
consumer ACKs, and atomic policy-version commit remain mandatory control-plane
checks.

## Fixed smoke entry points

```bash
# Run once in the CPU builder, then save mova-u15-lightllm-x2v-rl:v3.
bash docker/rl-engine/build_runtime.sh "$PWD"

# Run in the fixed two-H200 Notebook created from the saved image.
bash scripts/rl_engine/launch_server.sh
bash scripts/rl_engine/run_claim_smoke.sh
/opt/mostar-u1-py312/bin/python examples/serving/rl_smoke.py \
  --base-url http://127.0.0.1:8000 \
  --model-path /inspire/hdd/global_user/liangtianyi-253208120278/train-asset/models/SenseNova-U1.5-8B-MoT \
  --output-dir /tmp/mova-u15-rl-smoke
```

The standalone smoke performs a checkpoint publication, G=2 TI2T and TI2TI
rollouts, trace replay checks, controlled tensor mutation/restoration, stale
version rejection, missing-tensor/wrong-shape failures, trace cleanup, and
writes one JSON receipt. It validates the API contract; it is not the formal
live-FSDP publisher or a throughput benchmark.
