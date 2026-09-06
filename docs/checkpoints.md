# Checkpoint handoff

SFT and RL use the same FSDP2 runtime and share one model ABI.

```text
FSDP2 SFT
  ├── model-only DCP shards + sample metadata → model checkpoints
  └── atomic HF safetensors publication
        ├── FSDP2 RL initialization/reference
        └── LightLLM + LightX2V serving
FSDP2 RL
  └── model-only DCP shards + budget/policy metadata → model checkpoints
```

Checkpoints preserve the full model, including both MoT branches and the U1.5
pixel head. They do not contain optimizer, scheduler, EMA, scaler or RNG state.
SFT and RL start fresh; interrupted runs are not resumed. Runtime optimizers
and frozen reference snapshots remain necessary for training and stay in memory.
The model-only DCP loader restores weights without touching an optimizer.

The SFT publisher gathers the ordinary policy state after the final optimizer
step, copies the immutable configuration/tokenizer closure, prunes empty
unreferenced shards, and renames the completed staging directory atomically.
RL initialization and serving accept the complete HF closure.
