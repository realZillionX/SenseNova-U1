# 排障

- 生产入口拒绝 GPU：SFT、RL 与 serving 只支持 NVIDIA H200，应申请 H200
  训练组，不增加低显存 fallback。
- FlashAttention 出现 undefined symbol：在精确 Torch 2.8/CUDA 12.8 环境重编译。
- FA3-Neo preflight 失败：stock FA3 不能替代带 `image_token_end` 的版本。
- RL 返回 HTTP 409：先读取状态并发布目标 policy。
- serving 保持 paused：检查 language/vision/X2V receipt，不能强制恢复半更新模型。
- RL 拒绝 SFT 输出：确认 index 命名的所有 safetensors shard 都存在，且没有把
  staging 目录当成正式 checkpoint。
