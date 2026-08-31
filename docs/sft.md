# Full-parameter SFT

The only U1.5 SFT trainer is PyTorch 2.8 FSDP2. One process runs per H200;
block-level FSDP shards the complete language, understanding-vision,
generation-vision, MoT, and pixel-head parameter closure. Compute and gradient
reduction use BF16, optimizer masters remain FP32, and the production profile
uses forward resharding, two-layer prefetch, fused AdamW, and native-resolution
packing.

The preset disables text, image, and joint CFG-drop augmentation: deleting a
condition from DiVR cold-start data would delete authored reasoning and change
the controlled TI2T/TI2TI supervision.

Required inputs:

| Variable | Meaning |
| --- | --- |
| `MODEL_NAME_OR_PATH` | Complete U1.5 Hugging Face checkpoint |
| `VOCAB_FILE`, `TOKENIZER_PATH` | Matching tokenizer directory |
| `mm_data_path` | U1.5 loader meta JSON |
| `JOB_NAME` | New run namespace |
| `RUN_ROOT` | Durable output root |
| `SFT_CHECKPOINT_ROOT` | DCP checkpoints; defaults below the run |
| `SFT_HF_OUTPUT` | Atomic final HF publication target |

```bash
MODEL_NAME_OR_PATH=/models/SenseNova-U1.5-8B-MoT \
VOCAB_FILE=/models/SenseNova-U1.5-8B-MoT \
TOKENIZER_PATH=/models/SenseNova-U1.5-8B-MoT \
mm_data_path=/datasets/u15/meta.json \
JOB_NAME=u15-full-sft \
RUN_ROOT=/runs/u15-ti2t-sft \
bash training/shell/train_u1/U1.5_8B_SFT.sh
```

Each checkpoint contains sharded model and optimizer state plus per-rank EMA
and RNG state. `SFT_RESUME_CHECKPOINT` restores that closure, and the loader
replays the sealed number of optimizer batches before restoring sampled-model
RNG. The final ordinary policy weights are published as a complete HF
safetensors directory for RL and serving. Formal runs must seal the global
batch, activation checkpoint fraction, checkpoint interval, reshard setting,
prefetch depth, image limits, and ordered dataset identity. The production
default checkpoints every 1000 optimizer steps and retains only the newest two
committed recovery points; both values remain explicit run inputs. Periodic
DCP state is resumable, but the current atomic HF publication happens at the
planned final step. Killing a large-ceiling job is therefore not a formal early
stop unless a later launcher closes publication and receipt creation.
