# GDPO and UniGDPO

Forge RL is one full policy sharded over a Torch 2.8 FSDP2 mesh. It is not a
collection of independent replicas. Each optimizer batch samples multiple
prompt groups from the frozen behavior policy, obtains downstream reward
evidence, computes one GDPO advantage tensor, and performs the planned number
of PPO updates before publishing the new live policy to serving.

## Prompt input

Each JSONL row uses schema `sensenova.u15.forge.prompt.v1`:

```json
{
  "schema": "sensenova.u15.forge.prompt.v1",
  "sample_id": "sample-001",
  "modality": "ti2ti",
  "system_message": "Optional caller-owned policy context.",
  "prompt": "<image> Solve the task and return Answer: ...",
  "images": ["media/input.png"],
  "rollout_group_key": "sample-001"
}
```

Relative media paths resolve against the prompt file.

## Reward provider

`reward_command` starts one persistent downstream process. Forge writes one
JSON request per line containing `sample_id`, `modality`, `rollout_group_key`,
and ordered response items. The provider returns:

```json
{
  "schema": "sensenova.u15.forge.reward.response.v1",
  "reward": {
    "matrix": [[1.0, 1.0]],
    "dimension_names": ["parse", "semantic"],
    "availability": [[true, true]],
    "group_ids": ["sample-001"],
    "errors": [null],
    "diagnostics": [{}]
  },
  "budget": {}
}
```

Dimension meaning, hard-gate conditioning, task lookup and verifier execution
belong to the provider. Forge rejects unscorable rows before GDPO.

## Policy objectives

- reward dimensions normalize inside each prompt group before aggregation;
- the aggregate normalizes once across the complete optimizer batch;
- text and image branches consume the same detached trajectory advantage;
- each branch averages its own actions per trajectory before averaging
  trajectories; variable action count never changes rollout weight;
- text uses token PPO clipping and exact, unclipped k3 reference KL;
- image uses selected hybrid-SDE actions, RatioNorm clipping, and optional
  velocity-MSE to the fixed SFT reference;
- branch-local reductions are combined with explicit weights;
- serving likelihoods are rebuilt once in no-grad FSDP replay geometry before
  the first update; that detached old anchor remains fixed for every update on
  the batch and is never straight-through corrected during replay.

The persistent reward provider runs concurrently with that mandatory no-grad
anchor, so verifier latency and numerical alignment do not serialize on the
critical path.

RL text sampling is intentionally not the ordinary VQA profile. It is sealed
as temperature 1, top-p 1, unrestricted top-k, zero presence/frequency penalty,
and repetition penalty 1 so the serving sampler and replay implement the same
categorical policy.

## Plan and run

Create a JSON object matching `sensenova_u1.rl.plan.RlPlan`, then:

```bash
sensenova-forge rl-plan rl-plan-input.json
sensenova-forge rl-run /runs/u15-rl/plan.json
```

The runner saves FSDP2 model and AdamW state through Distributed Checkpoint,
per-rank RNG, budget counters, and the active serving policy version.
