# Full-parameter SFT

The public U1.5 preset uses one process per GPU with `wp=8`, `tp=1`, `pp=1`,
full-DP ZeRO-1, BF16 compute, FP32 optimizer state, weight communication
overlap, and native-resolution sequence packing. Language, understanding
vision, generation vision, MoT generation blocks, and the pixel head all remain
trainable.

This SFT preset disables text, image and joint CFG-drop augmentation. U1.5's
base checkpoint already provides CFG capability, while dropping conditions in
DiVR cold-start data would delete authored reasoning and change the controlled
TI2T/TI2TI supervision.

Required inputs:

| Variable | Meaning |
| --- | --- |
| `MODEL_NAME_OR_PATH` | Complete U1.5 Hugging Face checkpoint |
| `VOCAB_FILE`, `TOKENIZER_PATH` | Matching tokenizer directory |
| `mm_data_path` | InternEvo-loader meta JSON |
| `JOB_NAME` | New run namespace |
| `RUN_ROOT` | Output/checkpoint root; defaults to local `RUN`, set it to the sealed run directory for formal training |

The meta JSON points to datasets with `root`, `annotation`, `repeat_time`, and
`task`. Rows use the established `conversations` and `image` fields. Mixed
understanding/generation training may combine multimodal, interleaved, T2I and
IT2I tasks.

```bash
MODEL_NAME_OR_PATH=/models/SenseNova-U1.5-8B-MoT \
VOCAB_FILE=/models/SenseNova-U1.5-8B-MoT \
TOKENIZER_PATH=/models/SenseNova-U1.5-8B-MoT \
mm_data_path=/datasets/u15/meta.json \
JOB_NAME=u15-full-sft \
RUN_ROOT=/runs/u15-ti2t-sft \
bash training/shell/train_u1/U1.5_8B_SFT.sh
```

`enable_save_ckpt`, `checkpoint_every`, `checkpoint_snapshot_every`, loader
worker counts, packing buffers, sequence length, image limits, learning rates,
and activation-checkpoint fraction are environment overrides. Formal runs must
retain optimizer, sampler and RNG state for resume. `JOB_NAME` is mandatory and
must be unique per independent arm; its checkpoint staging directory is also
namespaced by the job. The launcher rejects missing checkpoint/tokenizer/data
assets and incompatible `world_size` versus `wp*tp*pp` before torchrun starts.

## Controlled trainer ablation

`training/shell/ablation/U1.5_8B_SFT_FSDP2.sh` is the optimized Torch 2.8
FSDP2 comparator. It reuses the same U1.5 model, native-resolution packed
loader, forward, losses, BF16 communication, FP32 optimizer masters, EMA, and
activation-checkpoint fraction as the InternEvo entry. It changes only the
trainer topology: one full packed row per data rank, block-level FSDP2,
explicit forward prefetch, and fused AdamW. Per-rank loss denominators preserve
InternEvo's equal-microbatch reduction before FSDP averages gradients.

The comparator accepts `FSDP2_RESHARD_AFTER_FORWARD`,
`FSDP2_PREFETCH_DEPTH`, and `FSDP2_FUSED_ADAMW` as sealed tuning inputs. A
fair comparison first materializes world-size-independent optimizer batches
with `SFT_MATERIALIZE_ONLY=true`, then passes the artifact to both trainers via
`SFT_ABLATION_BATCHES` and enables `SFT_ABLATION_DETERMINISTIC_MICROBATCH` on
InternEvo. The artifact fixes every ordered packed row, image tensor, label,
padding layout, and per-position RNG seed; its sidecar binds the byte identity
and ordered microbatch identities.

This entry remains an ablation runner rather than the production SFT handoff.
Replacing InternEvo also requires a sealed DCP resume lineage and the same
validated full-model Hugging Face publication/receipt consumed by RL.
