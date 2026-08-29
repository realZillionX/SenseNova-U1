# 运行环境身份

每个 run 都应记录 Forge commit、LightLLM/LightX2V gitlink、checkpoint、
package lock、Python/Torch/CUDA/NCCL、attention backend、启动参数和 active
policy version。SFT lock 位于 `training/`；RL/serving 合同位于
`docker/rl-engine/`，服务启动前由 `scripts/rl_engine/preflight.py` 验证。
