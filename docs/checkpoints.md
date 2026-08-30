# Checkpoint handoff

SFT and RL use the same FSDP2 runtime and share one model ABI.

```text
FSDP2 SFT
  ├── DCP model + optimizer + per-rank EMA/RNG → exact SFT state restore
  └── atomic HF safetensors publication
        ├── FSDP2 RL initialization/reference
        └── LightLLM + LightX2V serving
```

The publisher gathers the full ordinary policy state after the final optimizer
step, preserves both MoT branches and the U1.5 pixel head, copies the immutable
configuration/tokenizer closure, prunes empty unreferenced shards, and renames
the completed staging directory atomically. RL and serving accept only the
complete HF closure; optimizer and EMA state remain in DCP.
