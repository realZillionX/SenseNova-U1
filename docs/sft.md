# Full-parameter SFT

The only U1.5 SFT trainer is PyTorch 2.8 FSDP2. One process runs per H200;
block-level FSDP shards the complete language, understanding-vision,
generation-vision, MoT, and pixel-head parameter closure. Compute and gradient
reduction use BF16, optimizer masters remain FP32, and the production profile
uses forward resharding, two-layer prefetch, fused AdamW, and native-resolution
packing.

The public launcher defaults to learning rate `2e-5`, 1% sample warmup and a
constant schedule. AdamW uses betas `(0.9, 0.95)`, epsilon `1e-8` and zero
weight decay. The text CE/image velocity coefficients remain `0.1/1.0`;
velocity conversion uses `t_eps=0.02`, matching RL replay.

Text and image actions first average within each original sample. Each rank
then divides the sum by the complete optimizer batch's raw sample count
(including accumulation) divided by world size. FSDP's averaged gradients
therefore give a global sample mean; long responses, multiple images and
uneven packing do not increase a sample's total weight. Zero-image samples
contribute zero to the visual branch and remain in the sample denominator.
Padding copies have zero loss and do not advance exposure.

The SFT reader shuffles global annotation row indices by seed and epoch before
sharding across ranks and workers. Equal ordered source ids give the two arms
the same semantic permutation despite different response sizes. Byte offsets
are indexed in memory without rewriting annotation files. The packer decodes
each row once; malformed or oversized supervision raises instead of silently
skipping data. The packer drains its buffers and terminates at the end of each finite epoch.
Exhausted ranks execute zero-loss collective padding until all ranks exhaust;
readers never restart independently. `SFT_SAMPLE_AUDIT_DIR` optionally records
the raw sample ids consumed by each update and rank, excluding padding.
Real-data acceptance must compare these journals with the sealed selection;
the exposure clock alone is not coverage.

The preset disables text, image, and joint CFG-drop augmentation: deleting a
condition from DiVR cold-start data would delete authored reasoning and change
the controlled TI2T/TI2TI supervision.

Required inputs:

| Variable | Meaning |
| --- | --- |
| `MODEL_NAME_OR_PATH` | Complete U1.5 Hugging Face checkpoint |
| `VOCAB_FILE`, `TOKENIZER_PATH` | Matching tokenizer directory |
| `mm_data_path` | U1.5 loader meta JSON |
| `samples_per_epoch` | Sealed number of raw dataset rows (at least ten) |
| `max_samples` | Global raw-sample visits; defaults to one epoch |
| `warmup_samples`, `logging_samples` | Sample-based warmup and logging intervals |
| `JOB_NAME` | New run namespace |
| `RUN_ROOT` | Durable output root |
| `SFT_CHECKPOINT_ROOT` | DCP checkpoints; defaults below the run |
| `SFT_HF_OUTPUT` | Atomic final HF publication target |

```bash
MODEL_NAME_OR_PATH=/models/SenseNova-U1.5-8B-MoT \
VOCAB_FILE=/models/SenseNova-U1.5-8B-MoT \
TOKENIZER_PATH=/models/SenseNova-U1.5-8B-MoT \
mm_data_path=/datasets/u15/meta.json \
samples_per_epoch=1185000 \
JOB_NAME=u15-full-sft \
RUN_ROOT=/runs/u15-ti2t-sft \
bash training/shell/train_u1/U1.5_8B_SFT.sh
```

The global clock counts raw samples that actually participate in training,
summed over every rank and accumulation microbatch. Packing, token counts,
prefetch and padding copies do not advance it. An exposure epoch is
`samples_per_epoch` visits; this count alone does not prove unique-ID coverage.
Warmup, constant/cosine learning-rate progress, stopping, and logs use samples.
Optimizer update counts exist only for execution and checkpoint metadata.

Every full exposure epoch saves ten checkpoints at the first completed update
reaching each 10% boundary through 100% of its raw-sample count. Integer targets
round upward independently, preventing cadence drift when the row count is not
divisible by ten. The epoch-end checkpoint is the tenth save, not an eleventh.
A partial final epoch also saves its final sample boundary. Metadata records
both the target and actual sample count; an update can overshoot a target by
less than its raw batch size. Crossing two targets in one update is rejected
before changing weights: reduce the batch or increase the epoch size.

All committed checkpoints are retained for downstream evaluation. Intermediate
DCP directories use `samples-<actual count>`; the final DCP uses `final`, with its
sample count in `checkpoint.json`. Each contains full-model parameter shards
only. Optimizer, scheduler, EMA and RNG are not serialized; interrupted runs
are not resumed. Final ordinary policy weights are
atomically published as a complete Hugging Face safetensors directory for RL
and serving. Killing a run does not produce this final publication.

A committed DCP can also be exported independently on CPU with
the following command:

```bash
PYTHONPATH=training python training/tools/export_sft_checkpoint.py \
  --checkpoint <committed-directory> --target <new-HF-directory> \
  --base-model <base-checkpoint>
```

`--latest-from <checkpoint root>` selects
the latest complete boundary and ignores staging. Checkpoint metadata carries
the conversion configuration and original/actual sample budgets. Both live
and independent publication cast FP32 optimizer master weights to BF16 for
the HF checkpoint; DCP retains the full model masters.

System probes set `max_samples`, `SFT_BENCHMARK_REPORT` and optionally
`SFT_BENCHMARK_WARMUP_SAMPLES`. Reports contain actual sample counts, update
wall times, loss, gradient norms and peak HBM. `SFT_BENCHMARK_ONLY=true`
suppresses DCP/HF publication, requires a report, and forbids HF output; it is
never a formal training run.
