# Qwen 3.5 4B LoRA Training & Deployment Guide

This document provides complete instructions for training a Rust expert LoRA adapter on [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), with ChatML formatting, hardware-aware profiles, and deployment instructions for local and agent workflows.

## Model Overview

**Qwen 3.5 4B** is a highly capable, multilingual foundation model with exceptional coding and reasoning performance for its compact parameter class.

- **Base Model:** `Qwen/Qwen3.5-4B`
- **Architecture:** Causal LM with ChatML formatting
- **Context Length:** 32K context window
- **Precision Recommendation:** 16-bit `bfloat16` LoRA is strongly recommended. 4-bit QLoRA is not recommended for the Qwen 3.5 series to prevent degraded reasoning precision.
- **LoRA Configuration:** $r=16$, $\alpha=16$, dropout $0.0$ targeting all 7 projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Prompt Format:** Native ChatML template with Rust systems prompt

## 1. Quick Start: Training

### Step 1: Validate Setup (Dry-Run)

Validate dependencies, GPU memory, and dataset integrity before initiating training:

```bash
uv run python train_qwen.py --dry-run
```

This checks:

- CUDA availability and VRAM capacity
- Core dependencies (`transformers`, `peft`, `trl`, `torch`)
- Data format (`data/datasets/train.jsonl` with `instruction`, `input`, `output`)
- Inline configuration and hardware profiles

### Step 2: Start Training

Run with default settings (RTX 4070 profile in 16-bit bf16):

```bash
uv run python train_qwen.py
```

Outputs the trained adapter to `models/qwen3_5_4b_rust_lora`.

## 2. Hardware Profiles

Select a profile matching your available GPU hardware using `--hardware-profile`:

| Profile        | GPU             | VRAM        | Batch Size | Grad Accum | Max Seq Len | Mode                      |
| -------------- | --------------- | ----------- | ---------- | ---------- | ----------- | ------------------------- |
| `rtx4070`      | RTX 4070 / 3060 | 12GB        | 1          | 4          | 1024        | 16-bit bf16 (**Default**) |
| `rtx3090_4090` | RTX 3090 / 4090 | 24GB        | 1          | 4          | 2048        | 16-bit bf16               |
| `rtx6000`      | RTX 6000 Ada    | 48GB        | 2          | 2          | 4096        | 16-bit bf16               |
| `a100`         | NVIDIA A100     | 40GB / 80GB | 4          | 1          | 4096        | 16-bit bf16               |
| `dev`          | Any GPU         | Variable    | 1          | 1          | 512         | Minimal test profile      |

### Examples

```bash
# Default RTX 4070 profile
uv run python train_qwen.py

# Workstation RTX 6000 Ada profile (longer sequence length)
uv run python train_qwen.py --hardware-profile rtx6000

# High-throughput data center training on A100
uv run python train_qwen.py --hardware-profile a100

# Development quick test
uv run python train_qwen.py --hardware-profile dev
```

## 3. Hyperparameters & Configuration

Configurations are defined inline directly in `train_qwen.py`:

- **Model ID:** `Qwen/Qwen3.5-4B`
- **Precision:** `bfloat16` (or `float16` if bf16 unsupported)
- **Epochs:** `3`
- **Learning Rate:** `2e-4` with cosine decay and 10 warmup steps
- **LoRA Hyperparameters:**
  - Rank ($r$): `16`
  - Alpha ($\alpha$): `16`
  - Dropout: `0.0`
  - Target modules: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

## 4. CLI Reference

```
options:
  -h, --help            Show help message and exit
  --dry-run             Validate environment and dataset without training
  --hardware-profile    Hardware profile (rtx4070, rtx3090_4090, rtx6000, a100, dev)
  --model-path          Model identifier on HuggingFace Hub or local path
  --batch-size          Per-device training batch size
  --gradient-accumulation-steps
                        Number of gradient accumulation steps
  --max-seq-length      Maximum sequence length in tokens
  --num-epochs          Number of training epochs (default: 3)
  --learning-rate       Optimizer learning rate (default: 0.0002)
  --load-in-4bit        Force 4-bit quantization (QLoRA)
  --data-file           Path to training JSONL dataset
  --output-dir          Directory to save final LoRA weights
  --checkpoint-dir      Directory for intermediate training checkpoints
  --resume-from-checkpoint
                        Path to checkpoint directory to resume training
  --wandb-project       Weights & Biases project name for tracking
  --wandb-entity        Weights & Biases team/entity name
  --trust-remote-code   Enable execution of custom remote code
```

## 5. Output Artifacts

Training produces the following structure:

```
models/qwen3_5_4b_rust_lora/
├── adapter_config.json        # LoRA rank, alpha, target modules
├── adapter_model.safetensors  # Trained adapter weights
├── manifest.json              # Training metadata, hyperparameters, and duration
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── checkpoints/               # Periodic step checkpoints (if enabled)
    └── checkpoint-50/
```
