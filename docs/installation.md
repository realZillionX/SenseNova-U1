# Installation

Clone Forge and initialize the serving engines:

```bash
git clone git@github.com:realZillionX/SenseNova-U1.5-8B-MoT-Forge.git
cd SenseNova-U1.5-8B-MoT-Forge
git submodule update --init --recursive \
  serving/third_party/LightLLM serving/third_party/LightX2V
```

SFT, RL, and serving share the repository-root Python 3.12, Torch 2.8/CUDA
12.8 dependency lock:

```bash
uv sync --locked
docker build -f docker/rl-engine/Dockerfile \
  --build-arg FORGE_COMMIT="$(git rev-parse HEAD)" \
  -t sensenova-u15-forge:unified-v7 .
```

The image includes FlashAttention 2 for SFT/RL backward and FA3-Neo for
serving. Production entrypoints fail closed unless every visible training or
serving GPU is an NVIDIA H200.
