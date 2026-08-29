# 安装

```bash
git clone git@github.com:realZillionX/SenseNova-U1.5-8B-MoT-Forge.git
cd SenseNova-U1.5-8B-MoT-Forge
git submodule update --init --recursive \
  serving/third_party/LightLLM serving/third_party/LightX2V
```

SFT 使用独立的 Torch 2.5.1/CUDA 12.4 环境：

```bash
uv --directory training sync --locked
uv --directory training sync --locked --extra flash-build
uv --directory training sync --locked --extra flash-build --extra flash \
  --no-build-isolation-package flash-attn
```

RL 与 serving 使用自包含的 Torch 2.8/CUDA 12.8 镜像：

```bash
docker build -f docker/rl-engine/Dockerfile \
  -t sensenova-u15-forge:rl-serving-v1 .
```

两个 pyproject 不应装进同一个 venv；它们通过 HF safetensors 交接。
