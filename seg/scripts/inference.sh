#!/bin/bash
JSONL_PATH="inference_s4.jsonl" #s4/ms3 inference.jsonl
CHECKPOINT="sam2.1_hiera_large.pt"
MODEL_CFG="configs/sam2.1/sam2.1_hiera_l.yaml"
OUTPUT_DIR=""
BATCH_SIZE=48
NUM_GPUS=8

mkdir -p $OUTPUT_DIR

export MASTER_ADDR=localhost
export MASTER_PORT=6688

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    scripts/inference_3.py \
    --jsonl_path $JSONL_PATH \
    --checkpoint $CHECKPOINT \
    --model_cfg $MODEL_CFG \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE