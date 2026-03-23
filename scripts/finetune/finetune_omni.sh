#!/bin/bash

# ============================================================
# User Configuration — Please modify the following paths
# ============================================================
# Path to Qwen2.5-Omni-7B base model (local path or HuggingFace model ID)
QWEN_OMNI_PATH="Qwen/Qwen2.5-Omni-7B"
# Path to this project root directory
DATA_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# ============================================================

# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=2
MASTER_PORT=6600
RANK=0

qwen_omni=$QWEN_OMNI_PATH

# Training Arguments
LOCAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=1
GLOBAL_BATCH_SIZE=$WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE*$GRADIENT_ACCUMULATION_STEPS
# 16*8*4
# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=finetune
RUN_NAME=finetune
OUTP_DIR=results
export TOKENIZERS_PARALLELISM='true'

torchrun --nproc_per_node $NPROC_PER_NODE \
    --master_port $MASTER_PORT \
    scripts/finetune/finetune_omni.py \
    --data_root $DATA_ROOT \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/${RUN_NAME} \
    --deepspeed deepspeed/stage2.json \
    --model_name_or_path $qwen_omni \
    --exp_desc 'finetune qwen-omni' \
    --freeze_backbone True \
    --save_modules 'lora' \
    --lora_enable True \
    --bits 16 \
    --lora_alpha 256 \
    --lora_dropout 0.10 \
    --lora_r 128 \
    --fp16 False \
    --bf16 True \
    --tf32 False \
    --meld_task True \
    --cremad_task True \
    --ks_task True \
    --ucf_task True \
    --mafw_task True \
    --dfew_task True \
    --mer24_task True \
    --avqa_thu_task True \
    --unav_task True \
    --avqa_task True \
    --ave_task True \
    --avvp_task True \
    --arig_task True \
    --avcap_task True \
    --a2v_task True \
    --v2a_task True \
    --s4_task True \
    --ms3_task True \
    --ref_avs_task True \
    --num_train_epochs 5 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size $LOCAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.3 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --use_reentrant False \
    --gradient_checkpointing True \
    --half_precision_backend "auto" \
    --lr_scheduler_type "cosine" \
    --save_total_limit 10 \
    --logging_steps 1 \
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --ddp_find_unused_parameters True \
    --run_name $RUN_NAME \