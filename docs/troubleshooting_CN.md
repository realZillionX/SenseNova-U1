# 排障

- SFT 必须使用 `training/.venv`，不能复用 RL 环境。
- FlashAttention 出现 undefined symbol 时，应在精确 Torch/CUDA 环境内重编译。
- FA3-Neo preflight 失败时，stock FA3 不能替代带 `image_token_end` 的版本。
- RL 返回 HTTP 409 表示 policy version 已过期，应先读取状态并发布目标 policy。
- serving 保持 paused 表示至少一个 language/vision/X2V consumer 拒绝更新，不能
  强制恢复半更新模型。
- SFT checkpoint 进入 RL 前必须转换成 shard 完整的 HF safetensors 目录。
