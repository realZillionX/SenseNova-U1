# 全参数 SFT

U1.5 只保留 PyTorch 2.8 FSDP2 Trainer。每张 H200 运行一个进程；
block-level FSDP 分片完整的语言、理解视觉、生成视觉、MoT 与 Pixel Head 参数闭包。
计算和梯度归约使用 BF16，optimizer master 保持 FP32；生产配置启用 forward 后
reshard、两层 prefetch、fused AdamW 与原生分辨率 packing。

本 preset 关闭 text、image 与 joint CFG-drop augmentation，避免从 DiVR
cold-start 数据删除 authored reasoning，从而改变 TI2T/TI2TI 的受控监督。

```bash
MODEL_NAME_OR_PATH=/models/SenseNova-U1.5-8B-MoT \
VOCAB_FILE=/models/SenseNova-U1.5-8B-MoT \
TOKENIZER_PATH=/models/SenseNova-U1.5-8B-MoT \
mm_data_path=/datasets/u15/meta.json \
JOB_NAME=u15-full-sft \
RUN_ROOT=/runs/u15-ti2t-sft \
bash training/shell/train_u1/U1.5_8B_SFT.sh
```

每个 checkpoint 包含分片 model/optimizer 和逐 rank 的 EMA/RNG；
`SFT_RESUME_CHECKPOINT` 恢复该闭包，并在恢复 sampled-model RNG 前重放封存数量的
optimizer batch。最终普通 policy 权重原子发布为完整 HF safetensors，供 RL 与
serving 使用。正式 plan 必须封存 global batch、activation checkpoint 比例、
checkpoint 间隔、reshard、prefetch、图像限制与有序数据身份。
正式 checkpoint 间隔仍是待实机确定的 run 输入。runner 保留全部已提交中间
checkpoint 供跑分与过拟合分析，只能在下游消费者审计后清理。周期 DCP 可恢复，但
当前原子 HF publication 只在计划终点发生；直接停止大上限 Job 不是正式 early stop，
除非后续 launcher 补齐 publication 与 receipt。

短程系统与超参数探针可设置 `SFT_BENCHMARK_REPORT`、
`SFT_BENCHMARK_WARMUP_STEPS` 和 `SFT_BENCHMARK_MEASURED_STEPS`，记录真实
optimizer step 的时间、loss、gradient norm 与峰值显存。额外设置
`SFT_BENCHMARK_ONLY=true` 会关闭周期/终点 DCP 和 HF 发布；该模式必须提供报告
路径、禁止 resume 与 HF 输出，不能作为正式或可恢复训练。
