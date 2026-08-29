# SenseNova-U1.5-8B-MoT Forge

<p align="center">
  <strong>English</strong> · <a href="./README_CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT"><img src="https://img.shields.io/badge/🤗%20Model-SenseNova--U1.5--8B--MoT-yellow" alt="Model"></a>
  <img src="https://img.shields.io/badge/SFT-InternEvo-6f42c1" alt="InternEvo SFT">
  <img src="https://img.shields.io/badge/RL-GDPO%20%7C%20UniGDPO-2459B8" alt="GDPO and UniGDPO">
  <img src="https://img.shields.io/badge/Serving-LightLLM%20%2B%20LightX2V-0b8f6a" alt="LightLLM and LightX2V">
  <img src="https://img.shields.io/badge/License-Apache--2.0-blue" alt="Apache-2.0">
</p>

<p align="center">
  <img src="docs/assets/teaserU1.5.png" alt="SenseNova-U1.5-8B-MoT" width="100%">
</p>

Forge is a checkpoint-specific, high-performance stack for full-parameter
supervised fine-tuning, verifiable reinforcement learning, and production
inference of **SenseNova-U1.5-8B-MoT**. It keeps the mature InternEvo training
path for SFT, uses PyTorch 2.8 FSDP2 for GDPO/UniGDPO, and serves text/image
trajectories through pinned LightLLM + LightX2V engines.

## Highlights

| SFT | RL | Inference & rollout |
| --- | --- | --- |
| Full-parameter U1.5 training | GDPO and UniGDPO | Continuous-batch text decode |
| Native-resolution packing | Text-token and image-SDE PPO objectives | T2T, T2I, IT2I and interleaved generation |
| ISP + weight parallel + ZeRO-1 | Frozen old policy and SFT reference | Hybrid SDE–ODE rollout traces |
| EMA, checkpoint and exact resume | Block-level FSDP2 + DCP checkpoint | Atomic online NCCL weight updates |
| InternalEvo → HF safetensors handoff | Downstream reward-provider protocol | Policy-version admission barrier |

## Architecture

```text
                         ┌──────────────────────────────┐
 authored trajectories ─►  InternEvo full-parameter SFT│
                         └──────────────┬───────────────┘
                                        │ HF safetensors
                                        ▼
 ┌───────────────────────┐   rollout  ┌───────────────────────┐
 │ LightLLM + LightX2V   │◄───────────►│ PyTorch 2.8 FSDP2 RL │
 │ 2-GPU serving engine  │ trace/weights│ GDPO / UniGDPO       │
 └───────────┬───────────┘             └───────────┬───────────┘
             │                                     │ reward request
             ▼                                     ▼
 OpenAI-compatible API                   downstream reward provider
                                         (task/verifier owned by caller)
```

<p align="center">
  <img src="docs/assets/lightllm_x2v.png" alt="LightLLM and LightX2V serving topology" width="88%">
</p>

The reward provider is deliberately external. Forge owns the model policy,
rollout, replay, advantage construction, optimizer, checkpoint, and serving
control plane; callers own task semantics and verification.

## Quick start

### 1. Initialize production engines

```bash
git submodule update --init --recursive \
  serving/third_party/LightLLM \
  serving/third_party/LightX2V
```

### 2. Full-parameter SFT

SFT is an independent locked environment based on Torch 2.5.1/CUDA 12.4.

```bash
uv --directory training sync --locked
uv --directory training sync --locked --extra flash-build
uv --directory training sync --locked --extra flash-build --extra flash \
  --no-build-isolation-package flash-attn

MODEL_NAME_OR_PATH=/models/SenseNova-U1.5-8B-MoT \
VOCAB_FILE=/models/SenseNova-U1.5-8B-MoT \
TOKENIZER_PATH=/models/SenseNova-U1.5-8B-MoT \
mm_data_path=/datasets/u15/meta.json \
JOB_NAME=u15-ti2t-sft \
bash training/shell/train_u1/U1.5_8B_SFT.sh
```

Convert an InternalEvo checkpoint for FSDP2 RL or serving:

```bash
training/.venv/bin/python training/tools/revert2hf.py \
  --src /runs/u15-sft/checkpoint \
  --tgt /runs/u15-sft/hf \
  --extras-from /models/SenseNova-U1.5-8B-MoT
```

See [SFT](docs/sft.md) and [checkpoint handoff](docs/checkpoints.md).

### 3. Production serving

Build the self-contained Torch 2.8/CUDA 12.8 runtime and start the two-GPU
service:

```bash
docker build -f docker/rl-engine/Dockerfile \
  --build-arg FORGE_COMMIT="$(git rev-parse HEAD)" \
  -t sensenova-u15-forge:rl-serving-v1 .

MODEL_ROOT=/models/SenseNova-U1.5-8B-MoT \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/rl_engine/launch_server.sh
```

The service exposes `/v1/chat/completions`, `/v1/rl/rollouts`, trace streaming,
status, and online weight-control endpoints. Ordinary inference follows the
published U1.5 task profiles; RL deliberately uses a neutral full-softmax text
policy and a separately sealed hybrid SDE–ODE image schedule. See
[serving](docs/serving.md).

### 4. GDPO / UniGDPO

Forge consumes prompt-only JSONL rows and delegates reward semantics to a
persistent NDJSON command. Seal a plan, then launch FSDP2:

```bash
sensenova-forge rl-plan rl-plan-input.json
sensenova-forge rl-run /runs/u15-rl/plan.json
```

TI2T uses the text GDPO branch. TI2TI uses the same detached trajectory
advantage for text tokens and selected image SDE actions, with independent
ratio clipping, text KL, velocity-MSE, and explicit branch weights. See
[RL](docs/rl.md).

## Runtime profiles

| Profile | Python | Torch/CUDA | Transformers | Attention |
| --- | --- | --- | --- | --- |
| SFT | 3.10–3.12 | 2.5.1 / 12.4 | 4.43.x | FlashAttention 2 backward |
| RL + serving | 3.12 | 2.8.0 / 12.8 | 4.57.1 | FA2 backward + FA3-Neo forward |

These environments exchange only stable checkpoint and service protocols; an
InternalEvo optimizer checkpoint never crosses into the FSDP2 runtime.

## U1.5 gallery

<table>
  <tr>
    <td width="33%"><img src="docs/assets/u15_cafe_lifestyle.webp" width="100%"></td>
    <td width="33%"><img src="docs/assets/u15_nature_heals_poster.webp" width="100%"></td>
    <td width="33%"><img src="docs/assets/u15_sky_dragon.webp" width="100%"></td>
  </tr>
  <tr>
    <td align="center">Text-to-image</td>
    <td align="center">Text rendering</td>
    <td align="center">High-resolution composition</td>
  </tr>
</table>

<details>
<summary><strong>Image editing example</strong></summary>
<table>
  <tr>
    <th width="50%">Input</th><th width="50%">Output</th>
  </tr>
  <tr>
    <td><img src="docs/assets/u15_official_case01_input01.webp" width="100%"></td>
    <td><img src="docs/assets/u15_official_case01_result.webp" width="100%"></td>
  </tr>
</table>
</details>

<details>
<summary><strong>Interleaved visual reasoning example</strong></summary>
<p align="center"><img src="docs/assets/reasoning.png" width="80%"></p>
</details>

<details>
<summary><strong>SenseNova-U1.5 model capability reference</strong></summary>
<p align="center"><img src="docs/assets/u1.5_combined.webp" width="92%"></p>
<p align="center"><img src="docs/assets/u15_perform_vs_speed_6bench.webp" width="78%"></p>
</details>

## Documentation

- [Installation](docs/installation.md)
- [Full-parameter SFT](docs/sft.md)
- [GDPO / UniGDPO](docs/rl.md)
- [LightLLM + LightX2V serving](docs/serving.md)
- [Checkpoint handoff](docs/checkpoints.md)
- [Runtime identities](docs/runtime.md)
- [Troubleshooting](docs/troubleshooting.md)

## Scope

Forge intentionally does not ship task datasets, benchmark runners, evaluator
copies, verifier implementations, ComfyUI nodes, GGUF/offload paths, or serial
local-inference demos. Production generation goes through LightLLM + LightX2V.

## License

Apache License 2.0. Third-party copyrights and component licenses remain in
`NOTICE`, `training/NOTICE`, and source-file headers.
