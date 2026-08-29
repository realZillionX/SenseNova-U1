# LightLLM + LightX2V 服务

正式推理只走 `serving/third_party` 中钉死的两个 engine：GPU 0 上的 LightLLM
负责文本 decode、多模态编码、KV cache 与调度；GPU 1 上的 LightX2V 负责
NeoPP 图像 transition。

TI2TI 遇到 image action token 后暂停文本、生成并重新编码图片，再在同一总序列
预算内继续文本。RL route 额外返回 selected-token old log-prob、图文事件、
hybrid SDE–ODE trace、policy version 和在线权重更新 receipt。trace 只暂存在
共享内存，WebSocket 传输成功或断连后删除。

```bash
MODEL_ROOT=/models/SenseNova-U1.5-8B-MoT \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/rl_engine/launch_server.sh
```

普通推理与 RL 使用两套显式 profile：VQA 对齐官方 sampled text 配方
（temperature 0.6、top-p 0.95、top-k 20、repetition penalty 1.05，并计入 prompt
token）；T2I、编辑和 interleave 使用 greedy 文本，图像采用 50 个 flow step、
CFG 4、image CFG 1、`cfg_norm=none`、timestep shift 3，T2I/编辑使用 2K bucket，
interleave 默认使用 1.5K bucket。

RL 不继承上述 VQA penalty：文本固定为未修饰的 full-softmax 随机策略；图像关闭
CFG，并从不可变 plan 读取分辨率、step、shift、t-epsilon、noise level 和 SDE
window。`serving/configs/neopp_u15_forge_512.json` 只是 512×512、30-step、shift-1
的 RL 启动 fallback；每个请求都会更新真实 scheduler step，RL 返回实际 trace
geometry，replay 在调度不一致时直接拒绝。

LightLLM 默认只把加载权重后的 70% 可用显存交给 KV cache，剩余 headroom 属于
online publication 合同：服务暂停后仍必须能接收完整参数 bucket 而不 OOM。只有
同时测量 rollout 容量与最大权重 bucket 后，才可覆盖 `LIGHTLLM_MEM_FRACTION`。

LightLLM 缺失的 Triton kernel config 会在既有启动 warmup 中自适应调优
（`LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1`），随后供稳态请求复用。只有 runtime 已
包含当前 GPU 与 U1.5 精确 shape 的完整配置时，才应改回 level 0。
