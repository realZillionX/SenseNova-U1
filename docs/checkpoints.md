# Checkpoint handoff

SFT and RL keep separate optimizer formats but share one model ABI.

```text
InternalEvo SFT checkpoint
  ├── model/optimizer/sampler/scheduler → SFT resume only
  └── training/tools/revert2hf.py
        └── HF safetensors + config + tokenizer
              ├── FSDP2 RL initialization
              └── LightLLM + LightX2V serving
```

The converter merges weight-parallel shards, preserves both MoT branches and
the U1.5 pixel head, and writes a complete `model.safetensors.index.json`.
Forge RL accepts only a complete HF closure. FSDP2 checkpoints then use
Distributed Checkpoint and are not converted back into InternalEvo optimizer
state.
