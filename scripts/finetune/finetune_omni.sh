#!/bin/bash

# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=8
MASTER_PORT=6666
RANK=0

qwen_omni=/dockerdata/Qwen2.5-Omni-7B

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
export ASCEND_LAUNCH_BLOCKING='1'

torchrun --nproc_per_node $NPROC_PER_NODE \
    --master_port $MASTER_PORT \
    scripts/finetune/finetune_omni.py \
    --data_root /group/40061/cdn/Unifiedllm/Crab-Qwen2.5-Omni/ \
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
    --meld_task False \
    --cremad_task False \
    --ks_task False \
    --ucf_task False \
    --mafw_task False \
    --dfew_task True \
    --mer24_task False \
    --avqa_thu_task False \
    --unav_task False \
    --avqa_task False \
    --ave_task False \
    --avvp_task False \
    --arig_task False \
    --avcap_task False \
    --a2v_task False \
    --v2a_task False \
    --s4_task False \
    --ms3_task False \
    --ref_avs_task False \
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