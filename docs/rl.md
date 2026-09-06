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
and ordered response items. Request schema `sensenova.u15.forge.reward.request.v2`
also supplies `max_sequence_length` and an aligned `usage` list with integer
`text_tokens` and `image_context_tokens` for each rollout. These are actual
generated counts, excluding prompt context; providers can define explicit
resource objectives without re-tokenizing decoded text. Forge carries these
counts through distributed rollout materialization and its checkpoint budget
ledger, and emits reward means/weights and output counts in `reward_batch`
events. The provider returns:

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

The runner saves FSDP2 model weights through Distributed Checkpoint, budget
counters and the active serving policy version. Optimizer and RNG state stay
in memory and are not serialized; interrupted runs are not resumed.
AdamW explicitly uses betas `(0.9, 0.95)` and epsilon `1e-8`, with constant
text/shared and visual-generation learning rates of `1e-6` and zero weight
decay. Plan defaults use text KL `0.04`, activation checkpointing and a
50-update checkpoint interval. TI2TI resolves image objective/MSE weights to
`1.0/0.01`; TI2T resolves both to zero. Explicit overrides are sealed, including
zero MSE for a named method-removal experiment.
`torchrun.nnodes` may be any positive node count with one H200 process per GPU;
the rollout URL list may independently contain any positive number of serving
replicas. Each rank owns one prompt group, so `prompts_per_batch` equals the
FSDP world size. The 50-update checkpoint interval is sealed in the plan;
hardware validation measures its cost without opening a cadence search.
Retention has no automatic count cap: the runner keeps every committed checkpoint until score comparison and
downstream-consumer audit authorize cleanup.

Reaching `max_images` does not discard a trajectory: the image-action token is
masked for the remaining text tail so the policy can still close `</think>` and
emit `Answer:`. Reaching the joint `max_sequence_length=8192` limit returns a
`length` trajectory. It is still verified and optimized; a missing typed final
therefore receives the ordinary parse/semantic failure rather than being
silently filtered or resampled. Budget state separately records length
truncations and image-limit hits. Launch RL serving with `MAX_SEQUENCE_LENGTH=8192`;
its actual capacity must equal the plan limit. There is no separate text ceiling.
