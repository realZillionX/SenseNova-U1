# Checkpoint 交接

SFT 与 RL 使用不同 optimizer 格式，但共享同一个模型 ABI。InternalEvo
checkpoint 中的 optimizer/sampler/scheduler 只用于 SFT resume；
`training/tools/revert2hf.py` 合并 weight-parallel shard，保留 MoT 双分支与
U1.5 Pixel Head，生成完整 HF safetensors。HF closure 同时供 FSDP2 RL 与
LightLLM+LightX2V 使用。RL checkpoint 之后统一使用 Distributed Checkpoint。
