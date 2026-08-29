# LightLLM + LightX2V serving

Production inference uses two pinned engines under `serving/third_party`:

- GPU 0: LightLLM text decode, multimodal encode, KV cache and scheduling;
- GPU 1: LightX2V NeoPP image transitions.

They expose one OpenAI-compatible HTTP server. TI2TI stops a text span on the
image-action token, generates an image, re-encodes it, and resumes text under
the same total sequence budget.

RL routes add selected-token old log-probabilities, ordered text/image events,
hybrid SDE–ODE trace bundles, policy versions, and online weight-control
receipts. Trace bundles live only in shared memory and are deleted after
WebSocket transfer or disconnect.

```bash
MODEL_ROOT=/models/SenseNova-U1.5-8B-MoT \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/rl_engine/launch_server.sh
```

Run `examples/serving/client.py` for protocol inspection and
`examples/serving/rl_smoke.py` for rollout/trace/weight-control validation.

Ordinary inference and RL use different, explicit profiles:

- VQA follows the published sampled text recipe: temperature 0.6, top-p 0.95,
  top-k 20 and repetition penalty 1.05, including prompt tokens.
- T2I, editing and interleave use greedy text decode. Image generation uses 50
  flow steps, CFG 4, image CFG 1, `cfg_norm=none` and timestep shift 3; T2I and
  editing use the 2K buckets while interleave defaults to the published 1.5K
  buckets.
- RL text uses the unmodified full-softmax categorical policy. RL image rollout
  disables CFG and takes its resolution, step count, timestep shift, t-epsilon,
  noise level and SDE window from the immutable plan.

`serving/configs/neopp_u15_forge_512.json` is the 512×512, 30-step, shift-1 RL
startup fallback. Every request now updates the live scheduler step count; the
RL response reports the actual trace geometry, and replay refuses a schedule
mismatch.
