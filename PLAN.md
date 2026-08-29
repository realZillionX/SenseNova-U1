# PLAN

- [ ] 在相同 SenseNova-U1.5-8B-MoT 全参数闭包、ordered data、batch、
  序列/图像分布、BF16/FP32 口径、activation checkpoint 和硬件下，受控对拍
  InternEvo 与 Torch 2.8 FSDP2：分别测量 SFT 与固定轨迹 RL replay 的吞吐、
  峰值显存、通信暴露、checkpoint 成本和数值一致性；只有替代实现同时通过
  正确性门且性能不低于现有实现时，才统一 trainer 并删除较慢路径。
