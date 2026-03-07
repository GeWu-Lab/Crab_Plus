import os
import sys
sys.path.append(os.getcwd())
import logging
import torch
import json
from dataclasses import asdict
from torch.utils.data import DataLoader
from os.path import join

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    print('no npu!')

import pathlib
from models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
from models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor
from transformers import HfArgumentParser
from transformers import Qwen2VLImageProcessor, WhisperFeatureExtractor, Qwen2TokenizerFast
from configs.config_omni import ModelArgs, TrainingArgs, DataArgs, InferenceArgs
from dataset.qwen2_5_omni.omni_dataset import OmniTestDataset, DataCollatorForOmniTestDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils.util import (
    set_seed,
    find_all_linear_names,
    rank0_print,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    write2txt,
    prepare_sample,
    write2json
)

local_rank = None

class Test_DistributedSampler(DistributedSampler): 
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False): 
        super(Test_DistributedSampler, self).__init__(dataset, num_replicas, rank, shuffle)
        N = len(self.dataset)
        R = self.num_replicas
        base_num_samples = N // R
        remainder = N % R
        if self.rank < remainder:
            self.num_samples = base_num_samples + 1
        else:
            self.num_samples = base_num_samples

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices = indices[self.rank::self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

def run_inference(task_name, dataloader, ckpt_dir, model, mm_processor):
    save_dir = join(ckpt_dir, 'inference_results')
    os.makedirs(save_dir, exist_ok=True)
    
    pbar = tqdm(total=len(dataloader), desc=f'inference {task_name}')
    fp = join(save_dir, f'inference_{task_name}.jsonl')
    
    for step, sample in enumerate(dataloader):
        batch_metadata = sample.pop('metadata')
        bs = len(batch_metadata)
        input_lengths = sample['input_ids'].shape[1]
        
        sample = prepare_sample(data=sample, dtype=torch.bfloat16)
        sample.update({
            'use_cache': True,
            'max_new_tokens': 512,
        })
        
        with torch.no_grad():
            output = model.generate(**sample)
            generated_tokens = output[:, input_lengths:]
            responses = mm_processor.batch_decode(
                generated_tokens, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
        
        for i in range(bs):
            metadata = batch_metadata[i]
            metadata['predict'] = responses[i].strip()
            write2json(fp=fp, dict_data=metadata)
        
        pbar.update(1)
    pbar.close()


def main(attn_implementation=None):
    global local_rank
    set_seed(42)
    if not dist.is_initialized():
        dist.init_process_group(backend="hccl" if torch.npu.is_available() else "nccl", init_method="env://")

    parser = HfArgumentParser([ModelArgs, DataArgs, TrainingArgs, InferenceArgs])
    model_args, data_args, training_args, infer_args = parser.parse_args_into_dataclasses()

    output_dir = training_args.output_dir
    save_config = {
        'model_args': asdict(model_args),
        'data_args': asdict(data_args),
        'training_args': asdict(training_args),
        'infer_args': asdict(infer_args)
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(join(output_dir, 'saved_config.json'), 'w') as f:
        f.write(json.dumps(save_config, indent=4))

    local_rank = training_args.local_rank
    compute_dtype = torch.bfloat16
    model_name_or_path = model_args.model_name_or_path

    rank0_print(local_rank, f"Loading base model from {model_name_or_path}...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=compute_dtype,
    )
    model.disable_talker()

    model.config.use_cache = True
    ckpt_dir = infer_args.ckpt_dir

    thinker_model = model.thinker

    if training_args.lora_enable:
        from peft_hyper import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_nums = training_args.lora_num,
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj', 'down_proj','up_proj'],
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)

        lora_thinker_lm = get_peft_model(thinker_model, lora_config)
        thinker_model = lora_thinker_lm

        lora_weights_path = join(ckpt_dir, 'finetune_weights.bin')
        if os.path.exists(lora_weights_path):
            rank0_print(local_rank, f"Loading LoRA weights from {lora_weights_path}")
            state_dict = torch.load(lora_weights_path, map_location='cpu')
            thinker_model.load_state_dict(state_dict, strict=False)
            rank0_print(local_rank, "LoRA weights loaded successfully")

    audio_processor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
    vision_processor = Qwen2VLImageProcessor.from_pretrained(model_name_or_path)
    tokenizer = Qwen2TokenizerFast.from_pretrained(model_name_or_path)
    mm_processor = Qwen2_5OmniProcessor.from_pretrained(model_name_or_path)

    thinker_model.eval()
    thinker_model.cuda(local_rank)
    model = DDP(thinker_model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
    
    rank0_print(local_rank, "Loading test dataset...")
    dataset = OmniTestDataset(data_args=data_args, mode='test', tokenizer=tokenizer, audio_processor=audio_processor, 
                              vision_processor=vision_processor, mm_processor=mm_processor)
    collator = DataCollatorForOmniTestDataset(mm_processor=mm_processor, mode='test')

    batch_size = training_args.batchsize
    sampler = Test_DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=local_rank, shuffle=False)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler, 
                            collate_fn=collator, drop_last=False, num_workers=8)

    supported_tasks = [
        's4', 'ms3', 'avqa', 'ave', 'avvp', 'arig', 'avqa_thu', 'meld', 
        'mer24', 'dfew', 'mafw', 'cremad', 'ks', 'ucf', 'a2v', 'v2a', 'ref_avs'
    ]

    for task in supported_tasks:
        task_flag = f"{task}_task"
        if getattr(data_args, task_flag, False):
            run_inference(
                task_name=task,
                dataloader=dataloader,
                ckpt_dir=ckpt_dir,
                model=model.module,
                mm_processor=mm_processor
            )

if __name__ == "__main__":
    main()