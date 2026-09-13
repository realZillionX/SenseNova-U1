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

`batch_samples` is the required global number of original examples per
optimizer update. No packed-sequence count or fixed accumulation-step count
sets the training batch. The reader globally shuffles annotation indices by
seed and epoch, defines sample batches, and only then assigns and packs their
members. Metadata-only work estimates balance ranks without decoding media
on ranks that do not consume the sample. Equal source ids and seeds therefore give both arms the same batch
membership despite different response sizes. Partial epoch/final batches use
only their actual remainder; `max_samples` is exact.

Byte offsets are indexed in memory without rewriting annotations. Each row
is fully decoded once; malformed or oversized supervision raises. Packing stays
inside the sample batch and pads physical sequences only to the necessary
kernel alignment, and orders long physical sequences together across ranks
to avoid taking turns as the straggler. Accumulation adapts to the required FSDP forwards; ranks
with fewer physical sequences contribute zero-loss padding calls. These
calls do not add examples. `SFT_SAMPLE_AUDIT_DIR` records actual ids per rank
and update so membership, duplicates and omissions can be checked directly.

The preset disables text, image, and joint CFG-drop augmentation: deleting a
condition from DiVR cold-start data would delete authored reasoning and change
the controlled TI2T/TI2TI supervision.

Required inputs:

| Variable | Meaning |
| --- | --- |
| `MODEL_NAME_OR_PATH` | Complete U1.5 Hugging Face checkpoint |
| `VOCAB_FILE`, `TOKENIZER_PATH` | Matching tokenizer directory |
| `mm_data_path` | U1.5 loader meta JSON |
| `batch_samples` | Global original examples per optimizer update; explicit, no default |
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
batch_samples=${SFT_BATCH_SAMPLES:?set the global original-sample batch} \
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

Every full exposure epoch saves four checkpoints at the first completed update
reaching each 25% boundary through 100% of its raw-sample count. Integer targets
round upward independently, preventing cadence drift when the row count is not
divisible by four. The epoch-end checkpoint is the fourth save, not a fifth.
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

Research adapters can pass `validation_callback` and `checkpoint_writer` to
the trainer's `main`. Validation runs before training and at completed sample
checkpoint boundaries. Its `training_seconds` clock excludes validation and
checkpoint I/O. The default writer retains the ordinary ten-save contract;
an experimental writer policy must be declared in that experiment.
`tools.sft_validation.SampleValidation` evaluates fixed held-out sample IDs
with fixed image-noise seeds, preserves training RNGs and mode, restores all
FSDP parameter shards after no-grad inference, and performs
no backward pass. It reports per-sample text and image objective losses,
not generation quality or verifier correctness. Batch-size decisions require
validation progress versus sample exposure and time, not throughput alone.

GPU visibility masks are node-local allocation identities. Every trainer rank
checks its visible device count and H200 type and, when sample auditing is
enabled, records its hostname, GPU UUID and memory beside the sample journal.
Shared software-runtime validation does not require different nodes to expose
the same GPU UUIDs.

`--profiling` captures rank zero with sample-based windows:
`SFT_PROFILE_START_SAMPLES`, `SFT_PROFILE_WARMUP_SAMPLES`, and
`SFT_PROFILE_ACTIVE_SAMPLES`. Defaults are two, one, and one global sample
batches respectively; explicit windows must fit the run budget and include
at least one batch of warmup and recording. Traces live under
`SFT_PROFILE_ROOT` and are diagnostic artifacts. The progress clock advances
before the profiler selects the next window. Timing reports use the
nearest-rank definition for P95.
