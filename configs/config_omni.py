from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import transformers

@dataclass
class ModelArgs:  
    # llm
    model_name_or_path: Optional[str] = field(default="/dockerdata/Qwen2.5-Omni-7B")
    freeze_backbone: bool = field(default=True, metadata={"help": "Whether to freeze the LLM backbone."})
    llm_name: str = field(default='qwen')
    attn_impl: Optional[str] = field(
        default=None,
        metadata={'help': 'The implementation of attention.'}
    )
    
@dataclass
class InferenceArgs:
    # used for inference
    ckpt_dir: str = field(default='')
    device: str = field(default='cuda:0')

@dataclass
class DataArgs:
    n_frms: int = field(default=10)
    data_root: str = field(default="dataset")
    sample_rate: int = field(default=16000)
    avqa_thu_task: bool = field(default=False)
    meld_task: bool = field(default=False)
    unav_task: bool = field(default=False)
    avqa_task: bool = field(default=False)
    ave_task: bool = field(default=False)
    avvp_task: bool = field(default=False)
    arig_task: bool = field(default=False)
    avcap_task: bool = field(default=False)
    cremad_task: bool = field(default=False)
    dfew_task: bool = field(default=False)
    mafw_task: bool = field(default=False)
    mer24_task: bool = field(default=False)
    ks_task: bool = field(default=False)
    ucf_task: bool = field(default=False)
    a2v_task: bool = field(default=False)
    v2a_task: bool = field(default=False)
    s4_task: bool = field(default=False)
    ms3_task: bool = field(default=False)
    ref_avs_task: bool = field(default=False)


@dataclass
class TrainingArgs(transformers.TrainingArguments):
    output_dir: str = field(default="results")

    per_device_train_batch_size: int = field(
        default=1,
        metadata={'help': 'Train batch size.'}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={'help': 'Evaluation batch size.'}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={'help': 'Remove columns in the dataset that are not registered in the forward function?'}
    )
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={'help': 'Find unusuable parameters?'}
    )
    # NOTE: essential to keep comuputation graph because we need gradients for beacon tokens
    use_reentrant: Optional[bool] = field(
        default=None,
        metadata={'help': "Use reetrant in gradient checkpointing?"}
    )
    report_to: str = field(
        default="none",
        metadata={'help': 'Log results by external tools?'}
    )

    min_length: int = field(
        default=0,
        metadata={'help': 'How many tokens at minimum for training?'}
    )
    group_by_stride: Optional[str] = field(
        default=None,
        metadata={'help': 'Group the training data instances by the number of strides in the beacon model. {relaxed, strict}'}
    )
    sort_by_stride: Optional[str] = field(
        default=None,
        metadata={'help': 'Sort the training data instances by the number of strides in the beacon model. {ascend, descend}'}
    )
    only_train_beacon: bool = field(
        default=True,
        metadata={'help': 'Freeze LLM parameters when training beacon parameters?'}
    )
    
    eval_method: str = field(
        default="perplexity",
        metadata={'help': 'How to evaluate during training? {perplexity, generation}'}
    )
    eval_max_length: int = field(
        default=4096,
        metadata={'help': 'How many tokens at maximum for each input in evaluation.'},
    )
    eval_min_length: int = field(
        default=512,
        metadata={'help': 'How many tokens at minimum for each input in evaluation.'},
    )
    eval_beacon_ratio: List[int] = field(
        default_factory=lambda: [32],
        metadata={'help': 'Condensing ratios for beacons in evaluation.'}
    )
    eval_beacon_ratio_mix: str = field(
        default="adapt-1024",
        metadata={'help': 'How to determine the beacon_ratio for each input. {step-random, instance-random, adapt-x}'}
    )
    max_eval_num: Optional[int] = field(
        default=None,
        metadata={'help': 'How many samples for validation?'}
    )

    lora_enable: bool = field(default=False)
    lora_tune: bool = field(
        default=False,
        metadata={"help": "Use LoRA fine-tuning?"},
    )
    lora_rank: int = field(
        default=32,
        metadata={'help': 'LoRA rank.'}
    )

    lora_alpha: int = field(
        default=16,
        metadata={'help': 'LoRA scaling factor.'}
    )
    lora_dropout: float = field(
        default=0.,
        metadata={'help': 'LoRA dropout p.'}
    )
    lora_targets: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        metadata={"help": "Module name patterns to add LoRA."},
    )
    lora_extra_params: List[str] = field(
        default_factory=lambda: ["embed_tokens", "norm"],
        metadata={"help": "Extra trainable parameters except LoRA weights, if low rank training."},
    )

    metrics: List[str] = field(
        default_factory=lambda: [],
        metadata={'help': 'List of metrics. {rouge, save_result}'}
    )
    log_path: str = field(
        default="results/metrics.log",
        metadata={'help': 'Log file path.'}
    )

    lora_r: int = field(
        default=32,
        metadata={'help': 'LoRA rank.'}
    )
    lora_num: int = field(
        default=3,
        metadata={'help': 'LoRA num.'}
    )
    lora_bias: str = field(default='none')
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    save_modules: str = field(default='lora')
    exp_desc: str = field(default='exp')

    batchsize: int = 8