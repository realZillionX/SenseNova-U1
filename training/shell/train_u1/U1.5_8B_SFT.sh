#!/bin/bash
# Full-parameter PyTorch 2.8 FSDP2 SFT launcher for SenseNova-U1.5-8B-MoT.
#
# Single-node (8 GPUs):
#   bash shell/train_u1/U1.5_8B_SFT.sh
#
# Multi-node — run on each node, replacing NODE_RANK / MASTER_ADDR:
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash shell/train_u1/U1.5_8B_SFT.sh
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 bash shell/train_u1/U1.5_8B_SFT.sh

set -euo pipefail
cd "$(dirname "$0")/../.."  # repo root

# ============================ Distributed (torchrun) ============================ #
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}

# ============================ Model & data (placeholders — fill in!) ============================ #
export CONFIG_NAME="configs/sensenovavl_qwen3_gen/sensenovau1_5_8b_mot_sft.py"
export MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:?set MODEL_NAME_OR_PATH to a complete U1.5 HF checkpoint}
export VOCAB_FILE=${VOCAB_FILE:?set VOCAB_FILE to the matching tokenizer directory}
export TOKENIZER_PATH=${TOKENIZER_PATH:?set TOKENIZER_PATH to the matching tokenizer directory}
export mm_data_path=${mm_data_path:?set mm_data_path to the U1.5 data meta JSON}

# ============================ Parallelism ============================ #
# FSDP2 owns the complete sharding topology. The legacy parallel fields stay
# fixed at one because model construction still consumes this shared config.
export zero1_size=1
export wp_size=1
export tp_size=1
export pp_size=1
export tensor_parallel_mode=mtp
export grad_accm=${grad_accm:-1}
export FSDP2_RESHARD_AFTER_FORWARD=${FSDP2_RESHARD_AFTER_FORWARD:-true}
export FSDP2_PREFETCH_DEPTH=${FSDP2_PREFETCH_DEPTH:-2}
export FSDP2_FUSED_ADAMW=${FSDP2_FUSED_ADAMW:-true}
export SFT_PER_RANK_LOSS_REDUCTION=${SFT_PER_RANK_LOSS_REDUCTION:-true}

# ============================ Optimization ============================ #
export SEED=${SEED:-42}
export lr=${lr:-2e-4}
export lr_scheduler_type=${lr_scheduler_type:-"constant"}
export min_lr_ratio=${min_lr_ratio:-0.5}
export mlp_lr_scale=${mlp_lr_scale:-1.0}
export weight_decay=${weight_decay:-0}
export total_steps=${total_steps:-200000}
export init_steps=${init_steps:-2000}
export metric_interval_steps=${metric_interval_steps:-10}
# The production H200 profile keeps this conservative recomputation level so
# native-resolution outliers retain ample headroom.
export activation_checkpoint_fraction=${activation_checkpoint_fraction:-0.75}
export checkpoint_every=${checkpoint_every:-1000}
export checkpoint_keep_last=${checkpoint_keep_last:-2}
export SFT_CHECKPOINT_ROOT=${SFT_CHECKPOINT_ROOT:-"${RUN_ROOT:-RUN}/${JOB_NAME:-unset}/checkpoints"}
export SFT_HF_OUTPUT=${SFT_HF_OUTPUT:-"${RUN_ROOT:-RUN}/${JOB_NAME:-unset}/hf"}

# ============================ Data / sequence ============================ #
export num_imgs=${num_imgs:-144}
export seq_len=${seq_len:-8192}
export max_sample_tokens=${max_sample_tokens:-$seq_len}
export dataset_replacement=true
export dataloader_num_workers=${dataloader_num_workers:-8}
export dataloader_prefetch_factor=${dataloader_prefetch_factor:-1}
export dataloader_persistent_workers=${dataloader_persistent_workers:-false}
export packed_buffer_max_size=${packed_buffer_max_size:-10}
export packed_buffer_stale_threshold=${packed_buffer_stale_threshold:-200}
export min_num_frame=${min_num_frame:-1}
export max_num_frame=${max_num_frame:-128}
export dynamic_image_version="native_resolution"
export CONV_STYLE="sensenovalm2-chat-v3"
export down_sample_ratio=0.5
export max_pixels=$((1024 * 1024))
export min_pixels=$((256 * 256))
export max_pixels_gen=$((1024 * 1024))
export min_pixels_gen=$((256 * 256))
export LLM_DATA_WEIGHTS=0
export MM_CC_DATA_WEIGHTS=0

# ============================ Freeze / trainable modules ============================ #
export mot_random_init=false
export freeze_llm=false
export freeze_backbone=false
export unfreeze_mot_gen=true

# ============================ Generation / diffusion ============================ #
export time_schedule="standard"
export time_shift_type="exponential"
export time_base_dist="logit_normal"
export base_shift=0.5
export max_shift=1.15
export base_image_seq_len=64
export max_image_seq_len=4096
export noise_scale_mode="resolution"
export noise_scale_base_image_seq_len=64
export add_noise_scale_embedding=true
export noise_scale_max_value=16
export use_pixel_head=true
export P_mean=-0.8
export P_std=0.8
# DiVR SFT supervises authored reasoning exactly. The base checkpoint already
# owns CFG capability; unconditional-drop augmentation would delete the very
# CoT whose text/visual medium defines the controlled arms.
export cfg_txt_uncond_drop_prob=0
export cfg_img_uncond_drop_prob=0
export cfg_txtimg_uncond_drop_prob=0
export cfg_is_uncond_drop_independent='false'
export ema_decay=0.9999
export enable_ema=${enable_ema:-true}
export thinking_method="tag"

# ============================ Understanding ============================ #
# Mixed understanding/generation corpora can give an individual data rank no
# generation tokens even when other ranks have them. Keep the trainable MoT
# branch collective-safe with the U1.5 zero-loss dummy image.
export pad_dummy_image_gen=${pad_dummy_image_gen:-true}
export ce_loss_weight=${ce_loss_weight:-0.1}
export enable_und_loss='true'

# ============================ Job / logging ============================ #
export JOB_NAME=${JOB_NAME:?set JOB_NAME to a unique arm/run namespace}
export RUN_ROOT=${RUN_ROOT:-"RUN"}
# export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"
# export WANDB_PROJECT="neo_unify"

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd):$(pwd)/../src"

# ============================ Fail-fast contract ============================ #
[[ -d "$MODEL_NAME_OR_PATH" ]] || { echo "MODEL_NAME_OR_PATH is not a directory: $MODEL_NAME_OR_PATH" >&2; exit 2; }
[[ -f "$MODEL_NAME_OR_PATH/config.json" ]] || { echo "model config.json is missing" >&2; exit 2; }
[[ -f "$MODEL_NAME_OR_PATH/model.safetensors.index.json" ]] || { echo "model safetensors index is missing" >&2; exit 2; }
[[ -d "$VOCAB_FILE" ]] || { echo "VOCAB_FILE is not a directory: $VOCAB_FILE" >&2; exit 2; }
[[ -d "$TOKENIZER_PATH" ]] || { echo "TOKENIZER_PATH is not a directory: $TOKENIZER_PATH" >&2; exit 2; }
[[ -f "$mm_data_path" ]] || { echo "mm_data_path is not a file: $mm_data_path" >&2; exit 2; }
[[ -f "$CONFIG_NAME" ]] || { echo "training config is missing: $CONFIG_NAME" >&2; exit 2; }

WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
(( WORLD_SIZE >= 2 )) || { echo "FSDP2 requires at least two H200 GPUs" >&2; exit 2; }

# ============================ Launch ============================ #
PROFILE_ARGS=()
if [[ ${SFT_BENCHMARK_PROFILE:-false} == true ]]; then
    PROFILE_ARGS+=(--profiling)
fi

torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train_sensenovau1_fsdp2.py \
        --config "${CONFIG_NAME}" \
        --launcher torch \
        --seed "${SEED}" \
        --backend nccl \
        "${PROFILE_ARGS[@]}"
