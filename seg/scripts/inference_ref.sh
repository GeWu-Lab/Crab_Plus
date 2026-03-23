#!/bin/bash
# ============================================================
# SAM2 Segmentation Inference for Ref-AVS task
# Usage: cd sam2 && bash scripts/inference_ref.sh
# ============================================================
JSONL_PATH="inference_ref_avs.jsonl"  # Path to JSONL from Crab inference (ref_avs)
CHECKPOINT="./checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG="configs/sam2.1/sam2.1_hiera_l.yaml"
OUTPUT_DIR="./sam2_results_ref_avs"
BATCH_SIZE=48
NUM_GPUS=8

mkdir -p $OUTPUT_DIR

export MASTER_ADDR=localhost
export MASTER_PORT=6688

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    scripts/inference_ref_3.py \
    --jsonl_path $JSONL_PATH \
    --checkpoint $CHECKPOINT \
    --model_cfg $MODEL_CFG \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --split "test_u" #test_u/s/n