# Crab+: A Scalable and Unified Audio-Visual Scene Understanding Model with Explicit Cooperation

🚀🚀 Welcome to the repo of **Crab+**! If our project helps you, please give us a ⭐ on GitHub to support us. 🙏🙏

[![arXiv](https://img.shields.io/badge/arXiv-2603.04128-AD1C18.svg?logo=arxiv)](https://arxiv.org/abs/2603.04128)
[![HF Dataset](https://img.shields.io/badge/🤗-Crab_Plus-9C276A.svg)](https://huggingface.co/datasets/Jayson236/Crab_Plus)
[![HF Model](https://img.shields.io/badge/🤗-Qwen2.5--Omni--7B-9C276A.svg)](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)

**Crab+** is a scalable and unified audio-visual scene understanding model built upon [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) with custom **I-LoRA** (Interaction-aware LoRA) fine-tuning. It addresses the negative transfer problem in multi-task audio-visual learning through explicit cooperation from both data and model perspectives, achieving positive transfer in multi-tasks learning.

## 📰 News

* **[2026.03.04]** Release training and evaluation codes of Crab+.
* **[2026.03.04]** Crab+ paper is available on [arXiv](https://arxiv.org/abs/2603.04128).

## 🛠️ Requirements and Installation

Basic dependencies:

* Python == 3.10
* PyTorch == 2.5.1
* Transformers
* DeepSpeed
* PEFT

Install required packages:

```bash
git clone https://github.com/xxx/Crab-Qwen2.5-Omni.git
cd Crab-Qwen2.5-Omni
conda create -n crab python=3.10 -y
conda activate crab

# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

pip install -r requirements.txt
```

### Install SAM2 (optional, for segmentation tasks only)

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e ".[notebooks]"
cd ..

# Download SAM2 checkpoint
mkdir -p sam2/checkpoints
wget -P sam2/checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

## 📦 Data Preparation

### Download Dataset Annotations

The JSON annotation files are hosted on HuggingFace:

```bash
# Method 1: Using huggingface-cli
huggingface-cli download Jayson236/Crab_Plus AVUIE_2.zip --repo-type dataset --local-dir .
unzip AVUIE_2.zip

# Method 2: Using wget
wget https://huggingface.co/datasets/Jayson236/Crab_Plus/resolve/main/AVUIE_2.zip
unzip AVUIE_2.zip
```

After extraction, you should see the `AVUIE_2/` directory containing JSON annotation files for all tasks.

### Download Media Files

Each dataset requires its original audio/video/image files. The expected directory structure:

- **Video+Audio datasets**: `AVUIE_2/{task}/video/{filename}` and `AVUIE_2/{task}/audio/{filename}`
- **Image+Audio datasets** (ms3, s4, ref_avs, arig): `AVUIE_2/{task}/{relative_path}` (paths stored directly in JSON)

#### Video + Audio Datasets

```
AVUIE_2/
├── a2v/
│   ├── train.json, test.json
│   ├── video/          # *.mp4 files
│   └── audio/          # *.mp3 files
├── v2a/
│   ├── train.json, test.json
│   ├── video/          # *.mp4 files
│   └── audio/          # *.mp3 files
├── ks/
│   ├── train.json, test.json
│   ├── video/          # *.mp4 files
│   └── audio/          # *.wav files
├── ucf/
│   ├── train.json, test.json
│   ├── video/          # *.avi files (in subdirectories)
│   └── audio/          # *.wav files (in subdirectories)
├── meld/
│   ├── train.json, test.json
│   ├── video/          # *.mp4 files (in train/ or test/)
│   └── audio/          # *.mp3 files (in train/ or test/)
├── mer24/
│   ├── train.json
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
│   ├── train.json
│   ├── video/          # *.mp4 files
│   └── audio/          # *.mp3 files
├── avvp/
│   ├── train.json, test.json
│   ├── video/          # *.mp4 files
│   └── audio/          # *.mp3 files
└── avcap/
    ├── train.json
    ├── video/          # *.mp4 files
    └── audio/          # *.mp3 files
```

#### Image + Audio Datasets (Segmentation-related)

```
AVUIE_2/
├── s4/
│   ├── train.json, test.json
│   └── AVS/v1s/       # Per-clip directories:
│       └── {clip_id}/
│           ├── audio.wav
│           ├── frames/         # 0.jpg, 1.jpg, ...
│           ├── labels_rgb/     # Ground-truth masks
│           └── labels_semantic/
├── ms3/
│   ├── train.json, test.json
│   └── AVS/v1m/       # Same structure as s4
├── ref_avs/
│   ├── train.json, test.json
│   └── REFAVS/media/
│       └── {clip_id}/
│           ├── audio.wav
│           ├── frames/
│           └── gt_mask/
└── arig/
    ├── train.json, test.json
    └── AVS/v1s/        # Shares media files with s4
```

## ⚖️ Model Weights

### Base Model

Download [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) from HuggingFace:

```bash
huggingface-cli download Qwen/Qwen2.5-Omni-7B --local-dir /path/to/Qwen2.5-Omni-7B
```

If using a local path, update `QWEN_OMNI_PATH` in the shell scripts.

### Fine-tuned LoRA Weights

```bash
mkdir -p weight

# Method 1: Using huggingface-cli
huggingface-cli download Jayson236/Crab_Plus finetune_weights.bin --repo-type dataset --local-dir weight/

# Method 2: Using wget
wget -P weight/ https://huggingface.co/datasets/Jayson236/Crab_Plus/resolve/main/finetune_weights.bin
```

## 🗝️ Fine-tuning

Edit `scripts/finetune/finetune_omni.sh` to configure paths, then run:

```bash
bash scripts/finetune/finetune_omni.sh
```

Key training arguments:
- `NPROC_PER_NODE=2`: Number of GPUs per node
- `LOCAL_BATCH_SIZE=4`: Per-GPU batch size
- `--num_train_epochs 5`: Number of epochs
- `--lora_r 128`, `--lora_alpha 256`: LoRA rank and scaling
- `--deepspeed deepspeed/stage2.json`: DeepSpeed ZeRO Stage 2

## 🔍 Inference

Edit `scripts/finetune/inference_omni.sh` to configure paths, then run:

```bash
bash scripts/finetune/inference_omni.sh
```

### Segmentation (SAM2)

For segmentation tasks (S4, MS3, Ref-AVS), a two-stage pipeline is used:

1. **Stage 1**: Run Crab+ inference to generate predictions with bounding boxes / point coordinates
2. **Stage 2**: Feed predictions into SAM2 for mask generation

```bash
cd sam2

# S4 / MS3
bash ../seg/scripts/inference.sh

# Ref-AVS
bash ../seg/scripts/inference_ref.sh
```

## 🙏 Acknowledgement

This project is built upon the following open-source projects:

- [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni)
- [SAM2](https://github.com/facebookresearch/sam2)
- [PEFT](https://github.com/huggingface/peft)

## 📑 Citation

If you find Crab+ useful for your research and applications, please cite using this BibTeX:

```bibtex
@inproceedings{du2025crab,
  title={Crab: A unified audio-visual scene understanding model with explicit cooperation},
  author={Du, Henghui and Li, Guangyao and Zhou, Chang and Zhang, Chunjie and Zhao, Alan and Hu, Di},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18804--18814},
  year={2025}
}

@article{cai2026crab,
  title={Crab$^{+}$: A Scalable and Unified Audio-Visual Scene Understanding Model with Explicit Cooperation},
  author={Cai, Dongnuan and Du, Henghui and Zhou, Chang and Chen, Xi and Guo, Dan and Zhang, Hongyuan and Li, Xuelong and Hu, Di},
  journal={arXiv preprint arXiv:2603.04128},
  year={2026}
}
```

## 🔒 License

This project is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
