# 安装

```bash
git clone git@github.com:realZillionX/SenseNova-U1.5-8B-MoT-Forge.git
cd SenseNova-U1.5-8B-MoT-Forge
git submodule update --init --recursive \
  serving/third_party/LightLLM serving/third_party/LightX2V
```

SFT、RL 与 serving 共用仓库根目录的 Python 3.12、Torch 2.8/CUDA 12.8
依赖锁：

```bash
uv sync --locked
docker build -f docker/rl-engine/Dockerfile \
  --build-arg FORGE_COMMIT="$(git rev-parse HEAD)" \
  -t sensenova-u15-forge:unified-v8 .
```

镜像同时包含 SFT/RL backward 使用的 FlashAttention 2 和 serving 使用的
FA3-Neo。生产入口会拒绝任何非 NVIDIA H200 的训练或 serving GPU。
