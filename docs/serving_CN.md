# LightLLM + LightX2V 服务

正式推理在每个 GPU pair 上只走 `serving/third_party` 中钉死的两个 engine：
偶数本地 GPU 上的 LightLLM 负责文本 decode、多模态编码、KV cache 与调度；
随后的奇数 GPU 上的 LightX2V 负责 NeoPP 图像 transition。

TI2TI 遇到 image action token 后暂停文本、生成并重新编码图片，再在同一总序列
预算内继续文本。RL route 额外返回 selected-token old log-prob、图文事件、
hybrid SDE–ODE trace、policy version 和在线权重更新 receipt。trace 只暂存在
共享内存，WebSocket 传输成功或断连后删除。

```bash
MODEL_ROOT=/models/SenseNova-U1.5-8B-MoT \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash scripts/rl_engine/launch_server.sh
```

未设置 `CUDA_VISIBLE_DEVICES` 时 launcher 使用容器内全部可见 GPU，并按
`0/1、2/3、…` 配对。多节点 Serving 在每个节点运行同一 launcher，通过
`FORGE_SERVING_REPLICA_ID_OFFSET` 接续全局副本编号；LightLLM 私有端口只依赖
节点内 pair index。

普通推理与 RL 使用两套显式 profile：VQA 对齐官方 sampled text 配方
（temperature 0.6、top-p 0.95、top-k 20、repetition penalty 1.05，并计入 prompt
token）；T2I、编辑和 interleave 使用 greedy 文本，图像采用 50 个 flow step、
CFG 4、image CFG 1、`cfg_norm=none`、timestep shift 3。纯 T2I 可选择逻辑 bucket；
有输入图的编辑与图文交错 evaluation 启用 dynamic resolution，跟随第一张 prompt 图。

RL 不继承上述 VQA penalty：文本固定为未修饰的 full-softmax 随机策略；图像硬
关闭 CFG、固定 512×512，并从不可变 plan 读取 step、shift、t-epsilon、noise level
和 SDE window。`serving/configs/neopp_u15_forge_512.json` 是普通 Serving 使用的
512×512、CFG-4、30-step、shift-1 启动 profile；RL route 显式覆盖 CFG 与 geometry，
replay 在调度不一致时直接拒绝。

trace TTL 是共享内存垃圾回收护栏，不是可凭直觉确定的性能常量。当前一小时 fallback 保留到
`max_images=10` 的 H200 完整 rollout 实测最老 bundle age 与共享内存峰值；正式
Serving 必须按测量结果留裕量并封存 TTL。bundle 在 group 返回后立即传输并删除，
断连同样主动清理。

LightLLM 默认只把加载权重后的 70% 可用显存交给 KV cache，剩余 headroom 属于
online publication 合同：服务暂停后仍必须能接收完整参数 bucket 而不 OOM。只有
同时测量 rollout 容量与最大权重 bucket 后，才可覆盖 `LIGHTLLM_MEM_FRACTION`。

LightLLM 缺失的 Triton kernel config 会在既有启动 warmup 中自适应调优
（`LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1`），随后供稳态请求复用。只有 runtime 已
包含当前 GPU 与 U1.5 精确 shape 的完整配置时，才应改回 level 0。
