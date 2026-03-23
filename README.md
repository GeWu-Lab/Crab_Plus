# Crab: Audio-Visual Understanding, Interaction, and Editing with Qwen2.5-Omni

This repository provides the official implementation of **Crab**, built upon [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) with custom **MMoELoRA** fine-tuning. Crab is a unified framework for **19 audio-visual understanding, interaction, and editing (AV-UIE) tasks**, including emotion recognition, action recognition, audio-visual question answering, audio-visual segmentation, and more.

## 📋 Table of Contents

- [Supported Tasks](#supported-tasks)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Model Weights](#model-weights)
- [Fine-tuning](#fine-tuning)
- [Inference](#inference)
- [Segmentation (SAM2)](#segmentation-sam2)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Supported Tasks

Crab supports 19 audio-visual tasks organized into the following categories:

| Category | Tasks |
|----------|-------|
| **Audio-Visual Matching** | Audio-to-Video Retrieval (A2V), Video-to-Audio Retrieval (V2A) |
| **Action Recognition** | Kinetics-Sound (KS), UCF-101 (UCF) |
| **Emotion Recognition** | MELD, MER2024 (MER24), CREMA-D, MAFW, DFEW |
| **Audio-Visual QA** | AVQA, AVQA-THU |
| **Event Localization** | AVE, UnAV |
| **Video Parsing** | AVVP |
| **Audio-Visual Captioning** | AVCap |
| **Audio-Visual Segmentation** | S4 (single-source), MS3 (multi-source) |
| **Referring Audio-Visual Segmentation** | Ref-AVS |
| **Audio-Referred Image Grounding** | ARIG |

## Project Structure

```
Crab-Qwen2.5-Omni/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── AVUIE_2/                           # Dataset annotations (JSON only, see Data Preparation)
│   ├── a2v/                           #   ├── train.json, test.json
│   ├── v2a/                           #   ├── train.json, test.json
│   ├── ks/                            #   ├── train.json, test.json
│   ├── ucf/                           #   ├── train.json, test.json
│   ├── meld/                          #   ├── train.json, test.json
│   ├── mer24/                         #   ├── train.json (train only)
│   ├── cremad/                        #   ├── train.json, test.json
│   ├── mafw/                          #   ├── train.json, test.json
│   ├── dfew/                          #   ├── train.json, test.json
│   ├── avqa/                          #   ├── train.json, test.json
│   ├── avqa_thu/                      #   ├── train.json, test.json
│   ├── ave/                           #   ├── train.json, test.json
│   ├── unav/                          #   ├── train.json (train only)
│   ├── avvp/                          #   ├── train.json, test.json
│   ├── avcap/                         #   ├── train.json (train only)
│   ├── ms3/                           #   ├── train.json, test.json
│   ├── s4/                            #   ├── train.json, test.json
│   ├── ref_avs/                       #   ├── train.json, test.json
│   └── arig/                          #   ├── train.json, test.json
├── configs/
│   └── config_omni.py                 # Model, data, and training configuration
├── dataset/
│   ├── qwen2_5_omni/
│   │   └── omni_dataset.py            # Core dataset class (OmniDataset & OmniTestDataset)
│   └── qwen_omni_utils/              # Multimodal processing utilities
├── deepspeed/                         # DeepSpeed configuration files
├── models/
│   └── qwen2_5_omni/                 # Qwen2.5-Omni model implementation
├── peft_hyper/                        # Custom PEFT/LoRA (MMoELoRA) implementation
├── scripts/
│   └── finetune/
│       ├── finetune_omni.sh           # Fine-tuning launch script
│       ├── finetune_omni.py           # Fine-tuning entry point
│       ├── inference_omni.sh          # Inference launch script
│       ├── inference_omni.py          # Inference entry point
│       └── Omni_trainer.py            # Custom Trainer
├── seg/
│   └── scripts/
│       ├── inference.sh               # SAM2 segmentation for S4/MS3
│       ├── inference_3.py             # SAM2 segmentation script for S4/MS3
│       ├── inference_ref.sh           # SAM2 segmentation for Ref-AVS
│       └── inference_ref_3.py         # SAM2 segmentation script for Ref-AVS
├── sam2/                              # SAM2 repository (clone separately)
├── weight/                            # Fine-tuned LoRA weights
│   └── finetune_weights.bin
└── utils/                             # Utility functions
```

## Environment Setup

### 1. Create conda environment

```bash
conda create -n crab python=3.10 -y
conda activate crab
```

### 2. Install PyTorch (match your CUDA version)

```bash
# For CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install SAM2 (required only for segmentation tasks: S4, MS3, Ref-AVS)

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e ".[notebooks]"
cd ..
```

Download the SAM2 checkpoint:

```bash
# Download sam2.1_hiera_large.pt to sam2/checkpoints/
mkdir -p sam2/checkpoints
wget -P sam2/checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

## Data Preparation

### Download Dataset Annotations

The JSON annotation files are hosted on HuggingFace. Download and extract them:

```bash
# Method 1: Using huggingface-cli
huggingface-cli download Jayson236/Crab_Plus AVUIE_2.zip --repo-type dataset --local-dir .
unzip AVUIE_2.zip

# Method 2: Using wget
wget https://huggingface.co/datasets/Jayson236/Crab_Plus/resolve/main/AVUIE_2.zip
unzip AVUIE_2.zip
```

After extraction, you should see the `AVUIE_2/` directory containing JSON annotation files for all 19 tasks.

### Download Media Files

Each dataset requires its original audio/video/image files. Below is the **expected directory structure** for media files. The code in `omni_dataset.py` resolves media paths as:

- **Video+Audio datasets**: `AVUIE_2/{task}/video/{filename}` and `AVUIE_2/{task}/audio/{filename}`
- **Image+Audio datasets** (ms3, s4, ref_avs, arig): `AVUIE_2/{task}/{relative_path}` (paths stored directly in JSON)

#### Video + Audio Datasets

These datasets store video files in a `video/` subdirectory and audio files in an `audio/` subdirectory:

```
AVUIE_2/
├── a2v/
│   ├── train.json, test.json
│   ├── video/          # *.mp4 files (video1_path, video2_path in JSON)
│   └── audio/          # *.mp3 files (audio_path in JSON)
├── v2a/
│   ├── train.json, test.json
│   ├── video/          # *.mp4 files (video_path in JSON)
│   └── audio/          # *.mp3 files (audio1_path, audio2_path in JSON)
├── ks/
│   ├── train.json, test.json
│   ├── video/          # *.mp4 files
│   └── audio/          # *.wav files
├── ucf/
│   ├── train.json, test.json
│   ├── video/          # *.avi files (in subdirectories, e.g., ApplyEyeMakeup/)
│   └── audio/          # *.wav files (in subdirectories)
├── meld/
│   ├── train.json, test.json
│   ├── video/          # *.mp4 files (in train/ or test/ subdirectory)
│   └── audio/          # *.mp3 files (in train/ or test/ subdirectory)
├── mer24/
│   ├── train.json      # (train only, no test split)
│   ├── video/          # *.mp4 files
│   └── audio/          # *.wav files
├── cremad/
│   ├── train.json, test.json
│   ├── video/          # *.mp4 files
│   └── audio/          # *.wav files
├── mafw/
│   ├── train.json, test.json
│   ├── video/          # *.mp4 files
│   └── audio/          # *.wav files
├── dfew/
│   ├── train.json, test.json
│   ├── video/          # *.mp4 files
│   └── audio/          # *.wav files
├── avqa/
│   ├── train.json, test.json
│   ├── video/          # *.mp4 files
│   └── audio/          # *.mp3 files
├── avqa_thu/
│   ├── train.json, test.json
│   ├── video/          # *.mp4 files
│   └── audio/          # *.mp3 files
├── ave/
│   ├── train.json, test.json
│   ├── video/          # *.mp4 files
│   └── audio/          # *.mp3 files
├── unav/
│   ├── train.json      # (train only, no test split)
│   ├── video/          # *.mp4 files
│   └── audio/          # *.mp3 files
├── avvp/
│   ├── train.json, test.json
│   ├── video/          # *.mp4 files
│   └── audio/          # *.mp3 files
└── avcap/
    ├── train.json      # (train only, no test split)
    ├── video/          # *.mp4 files
    └── audio/          # *.mp3 files
```

#### Image + Audio Datasets (Segmentation-related)

These datasets use relative paths stored directly in JSON (no separate `video/` or `audio/` subdirectory). Media files should be placed directly under the task directory:

```
AVUIE_2/
├── s4/
│   ├── train.json, test.json
│   └── AVS/v1s/       # Contains per-clip directories:
│       └── {clip_id}/
│           ├── audio.wav
│           ├── frames/
│           │   ├── 0.jpg, 1.jpg, ...
│           ├── labels_rgb/          # Ground-truth masks (for S4 evaluation)
│           │   └── 0.png, 1.png, ...
│           └── labels_semantic/     # Semantic masks (for ARIG)
│               └── 0.png, 1.png, ...
├── ms3/
│   ├── train.json, test.json
│   └── AVS/v1m/       # Same structure as s4 but for multi-source clips
│       └── {clip_id}/
│           ├── audio.wav
│           ├── frames/
│           │   └── 0.jpg, 1.jpg, ...
│           └── labels_rgb/
│               └── 0.png, 1.png, ...
├── ref_avs/
│   ├── train.json, test.json
│   └── REFAVS/media/  # Contains per-clip directories:
│       └── {clip_id}/
│           ├── audio.wav
│           ├── frames/
│           │   └── 0.jpg, 1.jpg, ...
│           └── gt_mask/     # Ground-truth masks for evaluation
└── arig/
    ├── train.json, test.json
    └── AVS/v1s/        # Shares the same media files as s4
        └── {clip_id}/
            ├── audio.wav
            ├── frames/
            │   └── 0.jpg, 1.jpg, ...
            └── labels_semantic/
                └── 0.png, 1.png, ...
```

> **Note on JSON field naming**: 
> - **Training JSON** (`train.json`): Image/audio datasets use `image_path` for the visual field.
> - **Test JSON** (`test.json`): Image/audio datasets use `visual_path` for the visual field.
> - Video/audio datasets consistently use `video_path` in both train and test JSON files.
> - This is by design — the code in `omni_dataset.py` correctly handles both field names.

## Model Weights

### Base Model

Download [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) from HuggingFace:

```bash
# Option 1: Use HuggingFace model ID directly (requires internet)
# The scripts default to "Qwen/Qwen2.5-Omni-7B"

# Option 2: Download to local path
huggingface-cli download Qwen/Qwen2.5-Omni-7B --local-dir /path/to/Qwen2.5-Omni-7B
```

If using a local path, update `QWEN_OMNI_PATH` in the shell scripts.

### Fine-tuned LoRA Weights

Download the fine-tuned MMoELoRA weights from HuggingFace and place them under `weight/`:

```bash
mkdir -p weight

# Method 1: Using huggingface-cli
huggingface-cli download Jayson236/Crab_Plus finetune_weights.bin --repo-type dataset --local-dir weight/

# Method 2: Using wget
wget -P weight/ https://huggingface.co/datasets/Jayson236/Crab_Plus/resolve/main/finetune_weights.bin
```

The weight directory should contain:
```
weight/
└── finetune_weights.bin    # MMoELoRA adapter weights (~1.74 GB)
```

## Fine-tuning

### Configuration

Edit the top of `scripts/finetune/finetune_omni.sh` to configure paths:

```bash
# Path to Qwen2.5-Omni-7B base model
QWEN_OMNI_PATH="Qwen/Qwen2.5-Omni-7B"   # or local path
# Project root is auto-detected from script location
```

### Run Fine-tuning

```bash
bash scripts/finetune/finetune_omni.sh
```

Key training arguments in the script:
- `NPROC_PER_NODE=2`: Number of GPUs per node
- `LOCAL_BATCH_SIZE=4`: Per-GPU batch size
- `--num_train_epochs 5`: Number of epochs
- `--lora_r 128`, `--lora_alpha 256`: LoRA rank and scaling
- `--deepspeed deepspeed/stage2.json`: DeepSpeed ZeRO Stage 2

Each task can be enabled/disabled via flags (e.g., `--meld_task True`, `--s4_task True`).

## Inference

### Configuration

Edit the top of `scripts/finetune/inference_omni.sh`:

```bash
QWEN_OMNI_PATH="Qwen/Qwen2.5-Omni-7B"   # Base model path
CKPT_DIR="${DATA_ROOT}/weight"              # LoRA weights directory
```

### Run Inference

```bash
bash scripts/finetune/inference_omni.sh
```

Enable the tasks you want to evaluate by setting the corresponding flags to `True`:

```bash
--meld_task True \
--s4_task True \
# ... etc.
```

Inference results are saved as JSONL files in the output directory.

## Segmentation (SAM2)

For the segmentation tasks (S4, MS3, Ref-AVS), a two-stage pipeline is used:

1. **Stage 1**: Run Crab inference to generate predictions with bounding boxes and point coordinates
2. **Stage 2**: Feed predictions into SAM2 for mask generation and evaluation

### S4 / MS3 Segmentation

```bash
cd sam2

# Edit seg/scripts/inference.sh to set the JSONL path from Stage 1
bash ../seg/scripts/inference.sh
```

Or run directly:

```bash
cd sam2
torchrun --nproc_per_node=8 \
    ../seg/scripts/inference_3.py \
    --jsonl_path <path_to_inference_s4.jsonl> \
    --checkpoint ./checkpoints/sam2.1_hiera_large.pt \
    --output_dir ./sam2_results_s4 \
    --batch_size 48
```

### Ref-AVS Segmentation

```bash
cd sam2

# Edit seg/scripts/inference_ref.sh to set the JSONL path and split type
bash ../seg/scripts/inference_ref.sh
```

The `--split` argument controls the evaluation subset:
- `test_s`: Seen categories
- `test_u`: Unseen categories  
- `test_n`: Null (no sounding object) evaluation

## Acknowledgement

This project is built upon the following open-source projects:

- [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni): The base omni-modal model
- [SAM2](https://github.com/facebookresearch/sam2): Segment Anything Model 2 for mask prediction
- [PEFT](https://github.com/huggingface/peft): Parameter-Efficient Fine-Tuning
- [DeepSpeed](https://github.com/microsoft/DeepSpeed): Distributed training optimization

## Citation

```bibtex
@inproceedings{du2025crab,
  title={Crab: A unified audio-visual scene understanding model with explicit cooperation},
  author={Du, Henghui and Li, Guangyao and Zhou, Chang and Zhang, Chunjie and Zhao, Alan and Hu, Di},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18804--18814},
  year={2025}
}

@article{cai2026crab,
  title={Crab $\^{}$\{$+$\}$ $: A Scalable and Unified Audio-Visual Scene Understanding Model with Explicit Cooperation},
  author={Cai, Dongnuan and Du, Henghui and Zhou, Chang and Chen, Xi and Guo, Dan and Zhang, Hongyuan and Li, Xuelong and Hu, Di},
  journal={arXiv preprint arXiv:2603.04128},
  year={2026}
}
```
