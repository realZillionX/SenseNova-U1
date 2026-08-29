# SenseNova-U1.5-8B-MoT full-parameter SFT

This independent Torch 2.5.1/CUDA 12.4 project owns InternEvo training,
native-resolution packing, ISP/weight parallel/ZeRO-1, EMA, checkpoint/resume,
and InternalEvo→HF conversion. The only public preset is
`shell/train_u1/U1.5_8B_SFT.sh`; it trains the complete language, vision,
generation, and pixel-head parameter closure.

See [the SFT guide](../docs/sft.md).
