# 全参数 SFT

U1.5 preset 使用单卡单进程、`wp=8`、`tp=1`、`pp=1`、全 DP ZeRO-1、
BF16 forward、FP32 optimizer、权重通信重叠和原生分辨率 packing。语言模型、
理解视觉分支、生成视觉分支、MoT generation block 与 Pixel Head 全部可训练。
本 preset 将 text、image 与 joint CFG-drop augmentation 全部关闭：U1.5 base
checkpoint 已具备 CFG 能力，而在 DiVR cold-start 数据上丢弃 condition 会删除
authored reasoning，破坏 TI2T/TI2TI 的受控监督。

```bash
MODEL_NAME_OR_PATH=/models/SenseNova-U1.5-8B-MoT \
VOCAB_FILE=/models/SenseNova-U1.5-8B-MoT \
TOKENIZER_PATH=/models/SenseNova-U1.5-8B-MoT \
mm_data_path=/datasets/u15/meta.json \
JOB_NAME=u15-full-sft \
RUN_ROOT=/runs/u15-ti2t-sft \
bash training/shell/train_u1/U1.5_8B_SFT.sh
```

meta JSON 使用 `root`、`annotation`、`repeat_time` 与 `task`，row 使用
`conversations` 和 `image`。正式 run 必须保存 model、optimizer、sampler、
scheduler 与 RNG 续训状态。`JOB_NAME` 是必填项，两条独立 arm 必须使用不同
namespace，checkpoint 中转目录也按该名称隔离。launcher 会在 torchrun 前拒绝
缺失的 checkpoint/tokenizer/data 资产，以及与 `wp*tp*pp` 不兼容的 world size。
`RUN_ROOT` 显式指定输出与 checkpoint 根；正式训练应指向封存的 run 目录，不把
运行产物写入源码 checkout。

## 受控 Trainer 消融

`training/shell/ablation/U1.5_8B_SFT_FSDP2.sh` 是优化后的 Torch 2.8
FSDP2 对照入口。它复用与 InternEvo 完全相同的 U1.5 模型、原生分辨率 packed
loader、forward、loss、BF16 通信、FP32 optimizer master、EMA 与 activation
checkpoint 比例；只改变 Trainer 拓扑：每个 data rank 处理一条完整 packed row、
block-level FSDP2、显式 forward prefetch 与 fused AdamW。loss 在各 rank 内先按
InternEvo 的 microbatch 口径归约，再由 FSDP 汇总梯度。

该入口把 `FSDP2_RESHARD_AFTER_FORWARD`、`FSDP2_PREFETCH_DEPTH` 与
`FSDP2_FUSED_ADAMW` 作为必须封存的调优输入。公平对拍先用
`SFT_MATERIALIZE_ONLY=true` 生成与 world size 无关的 optimizer batch，再让两种
Trainer 都通过 `SFT_ABLATION_BATCHES` 消费；InternEvo 同时开启
`SFT_ABLATION_DETERMINISTIC_MICROBATCH`。该资产固定每一条有序 packed row、图像
tensor、label、padding 布局与位置级 RNG seed，并由 sidecar 绑定文件身份和逐
microbatch 身份。

该入口目前是消融 runner，不是正式 SFT 交接入口。替换 InternEvo 之前，还必须补齐
封存的 DCP 精确续训 lineage，以及 RL 消费的同等 full-model Hugging Face
发布与 receipt 合同。
