# 运行环境身份

每个 run 都应记录 Forge commit、LightLLM/LightX2V gitlink、checkpoint、
package lock、Python/Torch/CUDA/NCCL、attention backend、启动参数和 active
policy version。SFT lock 位于 `training/`；RL/serving 合同位于
`docker/rl-engine/`，服务启动前由 `scripts/rl_engine/preflight.py` 验证。

LightLLM base commit 叠加 checksum 固定的
`serving/patches/lightllm-sensenova-policy.patch`。runtime preparation 幂等应用
该 overlay 到隔离的 runtime source 目录，不污染 Git checkout；若 request
schedule 或 RL 中性采样 marker 缺失，preflight 会拒绝启动。
