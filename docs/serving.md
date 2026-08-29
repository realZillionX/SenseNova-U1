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

The production profile is sealed at 512×512, 30 flow steps and timestep shift
1.0 in `serving/configs/neopp_u15_forge_512.json`; inference, rollout and replay
must use that same physical schedule.
