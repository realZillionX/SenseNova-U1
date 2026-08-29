# 全参数 SFT

U1.5 preset 使用单卡单进程、`wp=8`、`tp=1`、`pp=1`、全 DP ZeRO-1、
BF16 forward、FP32 optimizer、权重通信重叠和原生分辨率 packing。语言模型、
理解视觉分支、生成视觉分支、MoT generation block 与 Pixel Head 全部可训练。

```bash
MODEL_NAME_OR_PATH=/models/SenseNova-U1.5-8B-MoT \
VOCAB_FILE=/models/SenseNova-U1.5-8B-MoT \
TOKENIZER_PATH=/models/SenseNova-U1.5-8B-MoT \
mm_data_path=/datasets/u15/meta.json \
JOB_NAME=u15-full-sft \
bash training/shell/train_u1/U1.5_8B_SFT.sh
```

meta JSON 使用 `root`、`annotation`、`repeat_time` 与 `task`，row 使用
`conversations` 和 `image`。正式 run 必须保存 model、optimizer、sampler、
scheduler 与 RNG 续训状态。
