# Installation

Clone the private repository and initialize only the production engines:

```bash
git clone git@github.com:realZillionX/SenseNova-U1.5-8B-MoT-Forge.git
cd SenseNova-U1.5-8B-MoT-Forge
git submodule update --init --recursive \
  serving/third_party/LightLLM serving/third_party/LightX2V
```

## SFT environment

```bash
uv --directory training sync --locked
uv --directory training sync --locked --extra flash-build
uv --directory training sync --locked --extra flash-build --extra flash \
  --no-build-isolation-package flash-attn
```

The SFT lock is validated with Torch 2.5.1/CUDA 12.4 and Transformers 4.43.x.

## RL and serving environment

The supported path is the checked-in container, which builds Torch 2.8/CUDA
12.8, FlashAttention 2 for replay backward, and FA3-Neo for Hopper serving.

```bash
docker build -f docker/rl-engine/Dockerfile \
  -t sensenova-u15-forge:rl-serving-v1 .
```

Do not install both pyprojects into one virtual environment: they intentionally
pin incompatible Torch and Transformers ABIs and exchange only safetensors.
