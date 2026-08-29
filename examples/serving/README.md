# Serving protocol tools

`client.py` exercises the OpenAI-compatible LightLLM + LightX2V endpoint.
Its per-mode defaults match the published U1.5 profiles: sampled VQA, greedy
generation/interleave text, and the 50-step full image schedule. Every value is
still exposed as an explicit CLI override.
`rl_smoke.py` validates detached rollout, SDE trace transport, policy versions,
and online weight-update failures. Neither script loads a checkpoint locally.
