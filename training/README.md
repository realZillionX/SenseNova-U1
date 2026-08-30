# SenseNova-U1.5-8B-MoT full-parameter SFT

This directory contains the checkpoint-specific U1.5 model, native-resolution
packer, loss, and the sole production trainer:
`train_sensenovau1_fsdp2.py`. SFT runs in the repository-root Torch 2.8/CUDA
12.8 environment on H200, saves FSDP2 Distributed Checkpoint state, and
atomically publishes a complete Hugging Face safetensors checkpoint.

See [the SFT guide](../docs/sft.md).
