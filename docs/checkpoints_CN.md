# Checkpoint 交接

SFT 与 RL 使用同一套 FSDP2 runtime，并共享一个模型 ABI。

```text
FSDP2 SFT
  ├── DCP model + optimizer + 逐 rank EMA/RNG → SFT 精确状态恢复
  └── 原子发布 HF safetensors
        ├── FSDP2 RL 初始化/reference
        └── LightLLM + LightX2V serving
```

publisher 在最后一个 optimizer step 后汇集普通 policy 的完整状态，保留 MoT
双分支与 U1.5 Pixel Head，复制不可变 config/tokenizer 闭包，清理未引用空 shard，
再原子重命名完整 staging 目录。RL 与 serving 只接受完整 HF closure；
optimizer 与 EMA 状态保留在 DCP 中。
