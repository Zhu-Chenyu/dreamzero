#!/bin/bash
# DreamZero RMBench training with Wan2.2-TI2V-5B backbone
# Target machine: sheep (2x RTX 6000 Ada, 49GB each)
#
# Usage:
#   bash scripts/train/rmbench_training_wan22.sh
#
# Environment variables you can override:
#   RMBENCH_DATA_ROOT  - path to gr00t_rmbench_student_v2 LeRobot dataset
#   OUTPUT_DIR         - checkpoint output directory
#   NUM_GPUS           - number of GPUs (default 2)
#   WAN22_CKPT_DIR     - Wan2.2-TI2V-5B model weights
#   IMAGE_ENCODER_DIR  - Wan2.1-I2V-14B-480P directory (for CLIP encoder)
#   TOKENIZER_DIR      - umt5-xxl tokenizer directory
#   MAX_STEPS          - training steps (default 10000 for baseline)

export HYDRA_FULL_ERROR=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAMZERO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ============ USER CONFIGURATION ============
RMBENCH_DATA_ROOT=${RMBENCH_DATA_ROOT:-"$DREAMZERO_ROOT/data/gr00t_rmbench_student_v2"}
OUTPUT_DIR=${OUTPUT_DIR:-"$DREAMZERO_ROOT/checkpoints/dreamzero_rmbench_wan22"}
NUM_GPUS=${NUM_GPUS:-2}
PER_DEVICE_BS=${PER_DEVICE_BS:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((NUM_GPUS * PER_DEVICE_BS))}
MAX_STEPS=${MAX_STEPS:-10000}

WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.2-TI2V-5B"}
IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-"$DREAMZERO_ROOT/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"$DREAMZERO_ROOT/checkpoints/umt5-xxl"}
# =============================================

# ============ AUTO-DOWNLOAD WEIGHTS ============
if [ ! -d "$WAN22_CKPT_DIR" ] || [ -z "$(ls -A "$WAN22_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.2-TI2V-5B not found at $WAN22_CKPT_DIR. Downloading..."
    huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir "$WAN22_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi

if [ ! -f "$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]; then
    echo "CLIP encoder not found. Downloading Wan2.1-I2V-14B-480P (CLIP only needed)..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P \
        --include "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
        --local-dir "$IMAGE_ENCODER_DIR"
fi
# ================================================

if [ ! -d "$RMBENCH_DATA_ROOT" ]; then
    echo "ERROR: RMBench dataset not found at $RMBENCH_DATA_ROOT"
    echo "Expected: gr00t_rmbench_student_v2 in LeRobot format"
    exit 1
fi

EXPERIMENT_PY="$DREAMZERO_ROOT/groot/vla/experiment/experiment.py"

DEEPSPEED_CFG=${DEEPSPEED_CFG:-zero2_offload}

cd "$DREAMZERO_ROOT"
echo "Training DreamZero on RMBench with Wan22 5B"
echo "  Data:       $RMBENCH_DATA_ROOT"
echo "  Output:     $OUTPUT_DIR"
echo "  GPUs:       $NUM_GPUS"
echo "  Max steps:  $MAX_STEPS"

torchrun --nproc_per_node "$NUM_GPUS" --standalone "$EXPERIMENT_PY" \
    report_to=none \
    data=dreamzero/rmbench_relative_wan22 \
    wandb_project=dreamzero_rmbench \
    train_architecture=full \
    num_frames=33 \
    action_horizon=24 \
    num_views=2 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan22 \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    "training_args.deepspeed=groot/vla/configs/deepspeed/${DEEPSPEED_CFG}.json" \
    save_steps=1000 \
    training_args.warmup_ratio=0.05 \
    output_dir="$OUTPUT_DIR" \
    per_device_train_batch_size="$PER_DEVICE_BS" \
    global_batch_size="$GLOBAL_BATCH_SIZE" \
    max_steps="$MAX_STEPS" \
    weight_decay=1e-5 \
    save_total_limit=5 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=2 \
    image_resolution_width=320 \
    image_resolution_height=160 \
    save_lora_only=false \
    max_chunk_size=4 \
    frame_seqlen=50 \
    save_strategy=steps \
    rmbench_data_root="$RMBENCH_DATA_ROOT" \
    dit_version="$WAN22_CKPT_DIR" \
    text_encoder_pretrained_path="$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    image_encoder_pretrained_path="$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    vae_pretrained_path="$WAN22_CKPT_DIR/Wan2.2_VAE.pth" \
    tokenizer_path="$TOKENIZER_DIR"
