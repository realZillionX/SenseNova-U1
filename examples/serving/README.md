# Serving protocol tools

`client.py` exercises the OpenAI-compatible LightLLM + LightX2V endpoint.
`rl_smoke.py` validates detached rollout, SDE trace transport, policy versions,
and online weight-update failures. Neither script loads a checkpoint locally.
