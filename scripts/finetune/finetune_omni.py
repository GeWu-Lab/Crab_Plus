import os,sys
sys.path.append(os.getcwd())
import logging
import torch
import json
from dataclasses import asdict
from os.path import join

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    print('no npu!')
import pathlib

from scripts.finetune.Omni_trainer import Omni_trainer,OrderedOmni_trainer
from models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
from models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from dataset.qwen_omni_utils import process_mm_info
from transformers import HfArgumentParser
from transformers import Qwen2VLImageProcessor, WhisperFeatureExtractor, Qwen2TokenizerFast
from configs.config_omni import ModelArgs,TrainingArgs,DataArgs
from dataset.qwen2_5_omni.omni_dataset import OmniDataset,DataCollatorForOmniDataset

from utils.util import (
    set_seed,
    find_all_linear_names,
    rank0_print,
    get_mm_adapter_state_maybe_zero_3,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    write2txt,
    print_trainable_parameters
)

local_rank=None

def main(attn_implementation=None):

    global local_rank
    set_seed(42)

    parser = HfArgumentParser([ModelArgs, DataArgs, TrainingArgs])
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    output_dir = training_args.output_dir
    save_config = {
        'model_args':asdict(model_args),
        'data_args':asdict(data_args),
        'training_args':asdict(training_args),
    }
    os.makedirs(output_dir,exist_ok=True)
    with open(join(output_dir,'saved_config.json'),'w') as f:
        f.write(json.dumps(save_config,indent=4))

    local_rank = training_args.local_rank
    compute_dtype = torch.float32
    if training_args.bf16:
        compute_dtype = torch.bfloat16

    model_name_or_path = model_args.model_name_or_path
    
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype = compute_dtype,
    )
    model.disable_talker()
    model.config.use_cache = False

    if model_args.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    thinker_model = model.thinker

    if training_args.gradient_checkpointing:
        if hasattr(thinker_model, "enable_input_require_grads"):
            thinker_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            thinker_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft_hyper import LoraConfig,get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_nums = 3,
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj', 'down_proj','up_proj'],
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)

        rank0_print(local_rank, "Adding LoRA adapters to thinker language model...")
        lora_thinker_lm = get_peft_model(thinker_model, lora_config)
        thinker_model = lora_thinker_lm
        model.thinker = thinker_model

        for name, param in model.thinker.named_parameters():
            if 'lora' in name:
                param.requires_grad = True

    audio_processor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
    vision_processor = Qwen2VLImageProcessor.from_pretrained(model_name_or_path)
    tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
    mm_processor = Qwen2_5OmniProcessor.from_pretrained(model_name_or_path)

    if local_rank == 0:
        write2txt(fp=join(output_dir, 'model.txt'), info=str(model), mode='w')
        trainable_params, trainable_names = print_trainable_parameters(model)
        write2txt(fp=join(output_dir, 'model_trainable_params.txt'), info=f'trainable_params: {trainable_params/1e6:.2f}MB', mode='w')
        for name in trainable_names:
            param = next(p for n, p in model.named_parameters() if n == name)
            write2txt(fp=join(output_dir, 'model_trainable_params.txt'), info=f"{name} {param.shape}")

    save_modules = training_args.save_modules
    rank0_print(f'save_modules: {save_modules}')
    matched_keys = save_modules.split(',')

    dataset = OmniDataset(data_args=data_args, mode='train', tokenizer=tokenizer, audio_processor=audio_processor, 
                         vision_processor=vision_processor, mm_processor=mm_processor)
    collator = DataCollatorForOmniDataset(mm_processor=mm_processor, mode='train')

    trainer = Omni_trainer(
        model=model.thinker,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator
    )
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    model.config.use_cache = True

    if training_args.lora_enable:
        lora_state_dict = get_peft_state_maybe_zero_3(model.thinker.named_parameters(), training_args.lora_bias)
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.thinker.config.save_pretrained(training_args.output_dir)
            torch.save(lora_state_dict, os.path.join(training_args.output_dir, 'lora_weights.bin'))
            print(f"Saved LoRA weights with {len(lora_state_dict)} parameters")

if __name__ == "__main__":
    main()