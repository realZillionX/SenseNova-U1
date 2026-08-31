# SenseNova-U1.5-8B-MoT Forge

<p align="center">
  <a href="./README.md">English</a> · <strong>简体中文</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT"><img src="https://img.shields.io/badge/🤗%20模型-SenseNova--U1.5--8B--MoT-yellow" alt="模型"></a>
  <img src="https://img.shields.io/badge/SFT-FSDP2-6f42c1" alt="FSDP2 SFT">
  <img src="https://img.shields.io/badge/RL-GDPO%20%7C%20UniGDPO-2459B8" alt="GDPO 与 UniGDPO">
  <img src="https://img.shields.io/badge/推理-LightLLM%20%2B%20LightX2V-0b8f6a" alt="LightLLM 与 LightX2V">
  <img src="https://img.shields.io/badge/许可证-Apache--2.0-blue" alt="Apache-2.0">
</p>

<p align="center">
  <img src="docs/assets/teaserU1.5.png" alt="SenseNova-U1.5-8B-MoT" width="100%">
</p>

Forge 是一个只面向 **SenseNova-U1.5-8B-MoT** 的高性能全参数 SFT、可验证
强化学习与生产推理框架。SFT 与 GDPO/UniGDPO 都使用 PyTorch 2.8 FSDP2，
文本与图像轨迹统一通过钉死版本的 LightLLM + LightX2V 服务生成；全栈只在
同一套 H200 环境运行。

## 核心能力

| SFT | RL | 推理与 rollout |
| --- | --- | --- |
| U1.5 全参数训练 | GDPO 与 UniGDPO | 文本 continuous batching |
| 原生分辨率 packing | 文本 token / 图像 SDE 双分支 PPO | T2T、T2I、IT2I、图文交错 |
| block-level FSDP2 | old/current/reference policy | Hybrid SDE–ODE trace |
| EMA + DCP 完整状态 | block-level FSDP2 + DCP | 原子在线 NCCL 权重更新 |
| 原子发布 HF safetensors | 下游 reward-provider 协议 | policy-version barrier |

## 架构

```text
监督轨迹 → FSDP2 全参 SFT → DCP + HF safetensors → FSDP2 GDPO/UniGDPO
                                              ↕ rollout / trace / 权重
                                  LightLLM + LightX2V 两卡服务
                                              ↕ reward request
                                      下游 verifier / reward provider
```

Forge 拥有 policy、rollout、replay、advantage、optimizer、checkpoint 与 serving
控制面；任务语义和 verifier 由下游仓库提供。

## 快速开始

```bash
git submodule update --init --recursive \
  serving/third_party/LightLLM \
  serving/third_party/LightX2V
```

### 全参数 SFT

```bash
uv sync --locked

MODEL_NAME_OR_PATH=/models/SenseNova-U1.5-8B-MoT \
VOCAB_FILE=/models/SenseNova-U1.5-8B-MoT \
TOKENIZER_PATH=/models/SenseNova-U1.5-8B-MoT \
mm_data_path=/datasets/u15/meta.json \
JOB_NAME=u15-ti2t-sft \
RUN_ROOT=/runs/u15-ti2t-sft \
bash training/shell/train_u1/U1.5_8B_SFT.sh
```

### 生产推理与 rollout

```bash
docker build -f docker/rl-engine/Dockerfile \
  --build-arg FORGE_COMMIT="$(git rev-parse HEAD)" \
  -t sensenova-u15-forge:unified-v6 .

MODEL_ROOT=/models/SenseNova-U1.5-8B-MoT \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/rl_engine/launch_server.sh
```

launcher 默认使用全部可见 GPU，并按本地 `0/1、2/3、…` 组成
LightLLM/LightX2V 副本。任意数量 Serving 节点可各自运行同一 launcher；每个节点
只需用 `FORGE_SERVING_REPLICA_ID_OFFSET` 接续全局副本编号，节点内私有端口只依赖
本地 pair index，不会让全局副本数受单节点端口区间限制。

### GDPO / UniGDPO

```bash
sensenova-forge rl-plan rl-plan-input.json
sensenova-forge rl-run /runs/u15-rl/plan.json
```

TI2T 使用文本 GDPO；TI2TI 的文本 token 与选中图像 SDE action 消费同一个
trajectory advantage，同时保留独立 ratio clip、text KL、velocity-MSE 与分支权重。

## 运行环境

| Profile | Python | Torch/CUDA | Transformers | Attention |
| --- | --- | --- | --- | --- |
| SFT + RL + serving | 3.12 | 2.8.0 / 12.8 | 4.57.1 | FA2 backward + FA3-Neo forward |

运行环境与所有训练入口都会拒绝非 H200 硬件。

## 图像示例

<table>
  <tr>
    <td width="33%"><img src="docs/assets/u15_cafe_lifestyle.webp" width="100%"></td>
    <td width="33%"><img src="docs/assets/u15_nature_heals_poster.webp" width="100%"></td>
    <td width="33%"><img src="docs/assets/u15_sky_dragon.webp" width="100%"></td>
  </tr>
</table>

<details>
<summary><strong>图文交错视觉推理示例</strong></summary>
<p align="center"><img src="docs/assets/reasoning.png" width="80%"></p>
</details>

<details>
<summary><strong>SenseNova-U1.5 模型能力参考</strong></summary>
<p align="center"><img src="docs/assets/u1.5_combined.webp" width="92%"></p>
<p align="center"><img src="docs/assets/u15_perform_vs_speed_6bench.webp" width="78%"></p>
</details>

## 文档

- [安装](docs/installation_CN.md)
- [全参数 SFT](docs/sft_CN.md)
- [GDPO / UniGDPO](docs/rl_CN.md)
- [LightLLM + LightX2V 服务](docs/serving_CN.md)
- [Checkpoint 交接](docs/checkpoints_CN.md)
- [运行环境](docs/runtime_CN.md)
- [排障](docs/troubleshooting_CN.md)

## 范围

Forge 不承载任务数据、Benchmark runner、evaluator 副本、verifier、ComfyUI、
GGUF/offload 或本地串行推理脚本。正式生成统一走 LightLLM + LightX2V。

## 许可证

Apache License 2.0。第三方版权与组件许可证保留在 `NOTICE`、
`training/NOTICE` 和源码文件头中。
