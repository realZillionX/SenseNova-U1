# TI2T/TI2TI RL engine smoke integration

This development integration keeps the existing `/v1/chat/completions`
implementation and adds a rollout/control plane around the LightLLM +
LightX2V serving stack. It does not contain a trainer, verifier, optimizer, or
checkpoint manager.

## Runtime pins

- SenseNova integration base: `34ca2f66e7006489a0184eb8896a75f4081a0257`
- LightLLM fork: `1d6803d7414213186b52f85930e91377c286fd99`
- LightX2V fork: `f05032c3c1036449d425b27cd854f1c2a398747f`
- requested upstream image:
  `lightx2v/lightllm_lightx2v:20260407@sha256:bb1900389c320b37dbcfe51fdf4db76a198d38a10c4c80d8b9b0726f1fb43ac7`

The Inspire registry currently cannot resolve the external digest directly.
The development notebook therefore starts from the existing
`mostar-u1-runtime:v4` platform image and the preflight treats its parent
identity as unverified until Torch 2.8.0, CUDA 12.8, FA/NCCL availability, and
all source commits have been recorded. The provided Dockerfile remains the
reproducible direct build from the requested digest.

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
safetensors files under `/tmp/mova_rl_traces` and are exposed through:

- `GET /v1/rl/traces/{bundle_id}`
- `DELETE /v1/rl/traces/{bundle_id}`

The RL-only image path disables CFG and applies the request's `t_eps`,
`timestep_shift`, noise level, and selected SDE window. The ordinary
LightX2V `generate()` path and its CFG behavior are unchanged.

## Policy update barrier

The HTTP controller stops new full-request admission and drains existing
text/image trajectories. Language, vision, and NeoPP consumers then join
separate publisher-to-consumer process groups, validate tensor metadata,
bucket/tensor checksums, and their own model-derived parameter closure before
applying an in-place update. Active policy version changes only after all
three consumers ACK. A failure leaves the service paused.

NCCL is the production transport when the publisher owns a distinct training
GPU. The fixed two-H200 smoke topology already assigns GPU 0 to LightLLM and
GPU 1 to LightX2V, so its standalone checkpoint publisher uses Gloo/CPU to
avoid creating duplicate NCCL ranks on a serving GPU. The receipt records this
transport deviation.

## Fixed smoke entry points

```bash
bash scripts/rl_engine/prepare_runtime.sh "$PWD"
bash scripts/rl_engine/launch_server.sh
bash scripts/rl_engine/run_claim_smoke.sh
python examples/serving/rl_smoke.py \
  --base-url http://127.0.0.1:8000 \
  --model-path /inspire/hdd/global_user/liangtianyi-253208120278/train-asset/models/SenseNova-U1.5-8B-MoT \
  --output-dir /tmp/mova-u15-rl-smoke
```

The smoke performs a full checkpoint publication, G=2 TI2T and TI2TI
rollouts, trace replay checks, controlled tensor mutation/restoration, stale
version rejection, checksum/missing-tensor/wrong-shape failures, trace cleanup,
and writes one JSON receipt.
