#!/bin/bash
# Controlled Torch 2.8 FSDP2 contrast for the U1.5 InternEvo SFT preset.

set -euo pipefail
cd "$(dirname "$0")/../.."

export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}

export CONFIG_NAME="configs/sensenovavl_qwen3_gen/sensenovau1_5_8b_mot_sft.py"
export MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:?set MODEL_NAME_OR_PATH}
export VOCAB_FILE=${VOCAB_FILE:?set VOCAB_FILE}
export TOKENIZER_PATH=${TOKENIZER_PATH:?set TOKENIZER_PATH}
export mm_data_path=${mm_data_path:?set mm_data_path}
export JOB_NAME=${JOB_NAME:?set JOB_NAME}
export SFT_BENCHMARK_REPORT=${SFT_BENCHMARK_REPORT:?set SFT_BENCHMARK_REPORT}

# FSDP2 is the only model-parallel dimension in this contrast. InternEvo uses
# wp=8 and accumulates eight samples; FSDP2 uses pure DP and consumes one
# ordered sample per rank, giving the same eight-sample optimizer batch.
export zero1_size=1
export wp_size=1
export tp_size=1
export pp_size=1
export grad_accm=${grad_accm:-1}

export SEED=${SEED:-42}
export lr=${lr:-2e-4}
export lr_scheduler_type=${lr_scheduler_type:-constant}
export min_lr_ratio=${min_lr_ratio:-0.5}
export mlp_lr_scale=1
export weight_decay=${weight_decay:-0}
export init_steps=${init_steps:-0}
export SFT_BENCHMARK_WARMUP_STEPS=${SFT_BENCHMARK_WARMUP_STEPS:-3}
export SFT_BENCHMARK_MEASURED_STEPS=${SFT_BENCHMARK_MEASURED_STEPS:-10}
export total_steps=$((SFT_BENCHMARK_WARMUP_STEPS + SFT_BENCHMARK_MEASURED_STEPS))
export activation_checkpoint_fraction=${activation_checkpoint_fraction:-0.75}
export metric_interval_steps=1
export enable_save_ckpt=false

export num_imgs=${num_imgs:-144}
export seq_len=${seq_len:-8192}
export max_sample_tokens=${max_sample_tokens:-$seq_len}
export dataset_replacement=true
export dataloader_num_workers=${dataloader_num_workers:-1}
export dataloader_prefetch_factor=${dataloader_prefetch_factor:-1}
export dataloader_persistent_workers=${dataloader_persistent_workers:-false}
export packed_buffer_max_size=${packed_buffer_max_size:-10}
export packed_buffer_stale_threshold=${packed_buffer_stale_threshold:-200}
export min_num_frame=1
export max_num_frame=128
export dynamic_image_version=native_resolution
export CONV_STYLE=sensenovalm2-chat-v3
export down_sample_ratio=0.5
export max_pixels=$((1024 * 1024))
export min_pixels=$((256 * 256))
export max_pixels_gen=$((1024 * 1024))
export min_pixels_gen=$((256 * 256))
export LLM_DATA_WEIGHTS=0
export MM_CC_DATA_WEIGHTS=0

export mot_random_init=false
export freeze_llm=false
export freeze_backbone=false
export unfreeze_mot_gen=true
export time_schedule=standard
export time_shift_type=exponential
export time_base_dist=logit_normal
export base_shift=0.5
export max_shift=1.15
export base_image_seq_len=64
export max_image_seq_len=4096
export noise_scale_mode=resolution
export noise_scale_base_image_seq_len=64
export add_noise_scale_embedding=true
export noise_scale_max_value=16
export use_pixel_head=true
export P_mean=-0.8
export P_std=0.8
export cfg_txt_uncond_drop_prob=0
export cfg_img_uncond_drop_prob=0
export cfg_txtimg_uncond_drop_prob=0
export cfg_is_uncond_drop_independent=false
export ema_decay=0.9999
export enable_ema=${enable_ema:-true}
export pad_dummy_image_gen=true
export ce_loss_weight=0.1
export enable_und_loss=true
export thinking_method=tag
export RUN_ROOT=${RUN_ROOT:-RUN}

[[ -d "$MODEL_NAME_OR_PATH" ]] || { echo "MODEL_NAME_OR_PATH is not a directory" >&2; exit 2; }
[[ -f "$mm_data_path" ]] || { echo "mm_data_path is not a file" >&2; exit 2; }
[[ $((NPROC_PER_NODE * NNODES)) -ge 2 ]] || { echo "FSDP2 requires at least two ranks" >&2; exit 2; }

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"
PROFILE_ARGS=()
if [[ ${SFT_BENCHMARK_PROFILE:-false} == true ]]; then
  PROFILE_ARGS+=(--profiling)
fi

torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  train_sensenovau1_fsdp2.py \
    --config "$CONFIG_NAME" \
    --launcher torch \
    --seed "$SEED" \
    --backend nccl \
    "${PROFILE_ARGS[@]}"
