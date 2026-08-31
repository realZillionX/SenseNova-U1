# LightLLM + LightX2V serving

Production inference uses two pinned engines under `serving/third_party` in
each GPU pair:

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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/rl_engine/launch_server.sh
```

If `CUDA_VISIBLE_DEVICES` is absent, the launcher uses all devices visible to
the container. It creates `0/1, 2/3, ...` LightLLM/LightX2V pairs. Multi-node
serving runs the same launcher per node and assigns consecutive global ids via
`FORGE_SERVING_REPLICA_ID_OFFSET`; private ports remain node-local.

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
  hard-disables CFG, copies the first prompt image's width and height (with only
  the model's factor-of-32 normalization), and takes step count, timestep shift,
  t-epsilon, noise level and SDE window from the immutable plan.

`serving/configs/neopp_u15_forge_rl.json` is the CFG-free, 30-step, shift-1 RL
startup profile. Every request updates the live scheduler step count and
first-input geometry; the RL response reports the actual trace geometry, and
replay refuses a schedule mismatch.

Trace TTL is a garbage-collection guard, not a guessed performance constant.
The default remains one hour until a full `max_images=10` H200 rollout measures
the oldest bundle age and shared-memory peak. A trace is consumed immediately
after its rollout group returns, then deleted; disconnect cleanup is also
eager. Formal serving seals the measured TTL with margin rather than silently
inheriting the default.

LightLLM reserves 70% of its post-weight memory for KV cache by default. The
remaining headroom is part of the online-publication contract: a paused server
must still receive a full-parameter bucket without OOM. Override
`LIGHTLLM_MEM_FRACTION` only after measuring both rollout capacity and the
largest planned weight bucket.

Missing LightLLM Triton kernel configurations are adaptively tuned during the
existing startup warmup (`LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1`) and reused for
steady-state requests. Set level 0 only when the pinned runtime already ships
complete configs for the exact GPU and U1.5 shapes.
