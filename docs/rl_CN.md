# GDPO 与 UniGDPO

Forge RL 在 Torch 2.8 FSDP2 mesh 上表达一个完整 policy，不是多份独立
replica。每个 optimizer batch 先在冻结 behavior policy 上生成多个 prompt group，
调用下游 reward provider，计算一次 GDPO advantage，再执行计划声明的 PPO update，
最后把 live shard 发布到 serving。

Prompt JSONL schema 为 `sensenova.u15.forge.prompt.v1`，包含 `sample_id`、
`modality`、可选 `system_message`、`prompt`、`images` 与
`rollout_group_key`。相对媒体路径以 prompt 文件为基准解析。

`reward_command` 是持久 NDJSON 进程，返回 reward matrix、dimension names、
availability、group ids、errors 和 diagnostics。维度语义、hard gate、Task 与
verifier 都由下游拥有；Forge 在 GDPO 前拒绝任何 unscorable row。

文本与图像分支消费同一个 detached trajectory advantage：文本使用 token PPO
clip 和精确、不截断的 k3 KL；图像使用选中 SDE action、RatioNorm clip 和可选
velocity-MSE。两个分支都先在每条 trajectory 内对自身 action 取均值，再对
trajectory 取均值，最后用显式权重组合，变长输出不会改变样本总权重。

LightLLM 产生 rollout 后、参数更新前，Forge 会用 no-grad FSDP replay geometry
重建一次 old likelihood；同一 batch 的全部 PPO update 固定使用这份 detached
anchor，replay 中不做 straight-through 修正。RL 文本采样固定为 temperature 1、
top-p 1、不限 top-k、presence/frequency penalty 为 0、repetition penalty 为 1；
它有意不同于普通 VQA 推理 profile，以确保 serving 与 replay 表达同一个策略。
持久 reward provider 与这次必需的 no-grad anchor 并行执行，verifier 等待不会和
数值对齐 replay 串行叠加。

```bash
sensenova-forge rl-plan rl-plan-input.json
sensenova-forge rl-run /runs/u15-rl/plan.json
```

checkpoint 使用 DCP 保存 FSDP2 model、AdamW、逐 rank RNG、预算与 active
serving policy version。
