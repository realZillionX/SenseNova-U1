# Full-parameter SFT

The public U1.5 preset uses one process per GPU with `wp=8`, `tp=1`, `pp=1`,
full-DP ZeRO-1, BF16 compute, FP32 optimizer state, weight communication
overlap, and native-resolution sequence packing. Language, understanding
vision, generation vision, MoT generation blocks, and the pixel head all remain
trainable.

Required inputs:

| Variable | Meaning |
| --- | --- |
| `MODEL_NAME_OR_PATH` | Complete U1.5 Hugging Face checkpoint |
| `VOCAB_FILE`, `TOKENIZER_PATH` | Matching tokenizer directory |
| `mm_data_path` | InternEvo-loader meta JSON |
| `JOB_NAME` | New run namespace |

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
bash training/shell/train_u1/U1.5_8B_SFT.sh
```

`enable_save_ckpt`, `checkpoint_every`, `checkpoint_snapshot_every`, loader
worker counts, packing buffers, sequence length, image limits, learning rates,
and activation-checkpoint fraction are environment overrides. Formal runs must
retain optimizer, sampler and RNG state for resume.
