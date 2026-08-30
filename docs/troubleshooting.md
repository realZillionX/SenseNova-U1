# Troubleshooting

- **A production entry rejects the GPU:** SFT, RL, and serving intentionally
  support only NVIDIA H200; request the H200 training group rather than adding
  a lower-memory fallback.
- **FlashAttention undefined symbol:** rebuild it inside the exact Torch
  2.8/CUDA 12.8 runtime.
- **Serving preflight fails FA3-Neo:** stock FA3 does not expose the scoped
  `image_token_end` ABI; rebuild the checked-in runtime.
- **RL request returns HTTP 409:** read `/v1/rl/status` and publish the intended
  policy before requesting new rollouts.
- **Serving remains paused:** inspect the language/vision/X2V receipts; never
  force-resume a half-updated policy.
- **SFT output is rejected by RL:** verify that every shard named by
  `model.safetensors.index.json` exists and that no partial staging directory
  was supplied.
