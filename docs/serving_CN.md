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

生产配置由 `serving/configs/neopp_u15_forge_512.json` 封存为 512×512、30 个
flow step、timestep shift 1.0；inference、rollout 与 replay 必须共享这套物理调度。
