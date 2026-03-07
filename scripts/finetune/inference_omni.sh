#!/bin/bash  
# Environment Variables 
WORLD_SIZE=1 
NPROC_PER_NODE=2
MASTER_PORT=6687 
RANK=0  
qwen_omni=/dockerdata/Qwen2.5-Omni-7B
# Training Arguments 
LOCAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1 
GLOBAL_BATCH_SIZE=$WORLD_SIZE$NPROC_PER_NODE$LOCAL_BATCH_SIZE$GRADIENT_ACCUMULATION_STEPS # 168*4 
# Log Arguments
export TRANSFORMERS_OFFLINE=1 
export WANDB_PROJECT=inference 
RUN_NAME=inference 
OUTP_DIR=results 
export TOKENIZERS_PARALLELISM='true' 
export ASCEND_LAUNCH_BLOCKING='1'  

torchrun --nproc_per_node $NPROC_PER_NODE \
    --master_port $MASTER_PORT \
    scripts/finetune/inference_omni.py \
    --data_root  \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/${RUN_NAME} \
    --deepspeed deepspeed/stage2-offload.json \
    --model_name_or_path $qwen_omni \
    --exp_desc 'inference qwen-omni' \
    --freeze_backbone True \
    --lora_enable True \
    --ckpt_dir /dockerdata/finetune \
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