#!/bin/bash
# ============================================================
# User Configuration — Please modify the following paths
# ============================================================
# Path to Qwen2.5-Omni-7B base model (local path or HuggingFace model ID)
QWEN_OMNI_PATH="Qwen/Qwen2.5-Omni-7B"
# Path to this project root directory
DATA_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Path to fine-tuned LoRA weight directory
CKPT_DIR="${DATA_ROOT}/weight"
# ============================================================

# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=2
MASTER_PORT=6687
RANK=0
qwen_omni=$QWEN_OMNI_PATH
# Training Arguments
LOCAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1
GLOBAL_BATCH_SIZE=$WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE*$GRADIENT_ACCUMULATION_STEPS
# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=inference
RUN_NAME=inference
OUTP_DIR=results
export TOKENIZERS_PARALLELISM='true'

torchrun --nproc_per_node $NPROC_PER_NODE \
    --master_port $MASTER_PORT \
    scripts/finetune/inference_omni.py \
    --data_root $DATA_ROOT \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/${RUN_NAME} \
    --deepspeed deepspeed/stage2-offload.json \
    --model_name_or_path $qwen_omni \
    --exp_desc 'inference qwen-omni' \
    --freeze_backbone True \
    --lora_enable True \
    --ckpt_dir $CKPT_DIR \
    --bits 16 \
    --lora_alpha 256 \
    --lora_dropout 0.10 \
    --lora_r 128 \
    --lora_num 3 \
    --batchsize 4 \
    --fp16 False \
    --bf16 True \
    --tf32 False \
    --meld_task False \
    --cremad_task False \
    --ks_task False \
    --ucf_task False \
    --mafw_task False \
    --dfew_task False \
    --avqa_thu_task False \
    --avqa_task False \
    --ave_task False \
    --avvp_task False \
    --arig_task False \
    --a2v_task False \
    --v2a_task False \
    --s4_task False \
    --ms3_task False \
    --ref_avs_task False \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size $LOCAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --dataloader_num_workers 8 \