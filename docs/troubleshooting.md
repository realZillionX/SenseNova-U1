# Troubleshooting

- **SFT imports a different Transformers API:** run from `training/.venv`; do
  not reuse the RL environment.
- **FlashAttention undefined symbol:** rebuild it inside the exact Torch/CUDA
  environment instead of copying a wheel from another runtime.
- **Serving preflight fails FA3-Neo:** stock FA3 is not compatible with the
  scoped `image_token_end` ABI; rebuild the checked-in runtime.
- **RL request returns HTTP 409:** the requested policy version is stale. Read
  `/v1/rl/status` and publish the intended FSDP policy before new rollouts.
- **Serving remains paused:** at least one language/vision/X2V consumer rejected
  a weight update. Inspect the three receipts; never force resume a half-updated
  policy.
- **SFT checkpoint will not load in RL:** convert to a complete HF safetensors
  directory and verify every shard named by the index exists.
