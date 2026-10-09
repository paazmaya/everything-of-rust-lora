# IBM Granite 4.2 LoRA Training & Deployment Guide

This document provides complete instructions for training a Rust expert LoRA adapter on [IBM Granite 4.2](https://huggingface.co/collections/ibm-granite/granite-42-language-models) (8B and 3B variants), merging and quantizing with GGUF, and deploying for coding agents or local inference.

## Model Overview

**IBM Granite 4.2** is a family of enterprise-ready, open foundation models built with state-of-the-art architecture for code intelligence, reasoning, and instruction following.

- **Supported Variants:**
  - `ibm-granite/granite-4.2-8b` (default, 8.2B parameters, 36 layers, 128K context)
  - `ibm-granite/granite-4.2-3b` (3.3B parameters, 32 layers, 128K context)
- **Context Length:** Up to 128K native context window
- **Quantization Support:** 4-bit QLoRA default (NF4 / Unsloth kernels), 16-bit bf16 on high-memory GPUs
- **LoRA Configuration:** $r=64$, $\alpha=128$, dropout $0.05$ targeting all 7 projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Prompt Format:** Standard Alpaca instruction format

## 1. Quick Start: Training

### Step 1: Validate Setup (Dry-Run)

Validate dependencies, GPU memory, and dataset integrity before initiating training:

```bash
uv run python train_granite.py --dry-run
```

This checks:

- CUDA availability and VRAM capacity
- Core dependencies (`transformers`, `peft`, `trl`, `torch`)
- Data format (`data/datasets/train.jsonl` with `instruction`, `input`, `output`)
- Inline configuration and hardware profiles

### Step 2: Start Training

Run with the default Granite 4.2 8B model and RTX 3090/4090 profile:

```bash
uv run python train_granite.py
```

Outputs the trained adapter to `models/granite_4.2_8b_rust_lora`.

## 2. Hardware Profiles

Select a profile matching your available GPU hardware using `--hardware-profile`:

| Profile        | GPU             | VRAM        | Batch Size | Grad Accum | Max Seq Len | Mode                      |
| -------------- | --------------- | ----------- | ---------- | ---------- | ----------- | ------------------------- |
| `rtx4070`      | RTX 4070 / 3060 | 12GB        | 1          | 8          | 2048        | 4-bit QLoRA               |
| `rtx3090_4090` | RTX 3090 / 4090 | 24GB        | 1          | 8          | 4096        | 4-bit QLoRA (**Default**) |
| `rtx6000`      | RTX 6000 Ada    | 48GB        | 2          | 4          | 4096        | 16-bit bf16               |
| `a100`         | NVIDIA A100     | 40GB / 80GB | 2          | 4          | 4096        | 16-bit bf16               |
| `dev`          | Any GPU         | Variable    | 1          | 1          | 512         | Minimal test profile      |

### Examples

```bash
# Train on 12GB GPU (RTX 4070)
uv run python train_granite.py --hardware-profile rtx4070

# Train with 3B variant on RTX 4070
uv run python train_granite.py --model-path ibm-granite/granite-4.2-3b --hardware-profile rtx4070

# Train on workstation with RTX 6000 Ada
uv run python train_granite.py --hardware-profile rtx6000
```

## 3. Variant Auto-Detection

The training script automatically detects whether an 8B or 3B model is specified from the model ID or directory path, and configures the default output directory accordingly:

```bash
# 8B model -> saves to models/granite_4.2_8b_rust_lora
uv run python train_granite.py --model-path ibm-granite/granite-4.2-8b

# 3B model -> saves to models/granite_4.2_3b_rust_lora
uv run python train_granite.py --model-path ibm-granite/granite-4.2-3b

# Local path detection
uv run python train_granite.py --model-path /models/granite-4.2-3b
```

## 4. Hyperparameters & Configuration

All configurations are defined inline directly in `train_granite.py`, eliminating external YAML dependencies:

- **Model ID:** `ibm-granite/granite-4.2-8b`
- **Epochs:** `3`
- **Learning Rate:** `2e-4` (cosine scheduler with 10 warmup steps)
- **Optimizer:** `adamw_8bit` with weight decay `0.01`
- **LoRA Hyperparameters:**
  - Rank ($r$): `64`
  - Alpha ($\alpha$): `128`
  - Dropout: `0.05`
  - Target modules: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

## 5. CLI Reference

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

## 6. Exporting to Ollama / GGUF

Once training is complete, export the adapter into a quantized GGUF file for local deployment:

```bash
uv run python scripts/10_export_ollama.py \
    --model-path models/granite_4.2_8b_rust_lora \
    --base-model ibm-granite/granite-4.2-8b \
    --model-name rust-granite-8b \
    --quant q4_k_m
```

This generates an Ollama Modelfile and packages the model for `ollama run rust-granite-8b`.

## 7. Output Artifacts

Training produces the following structure:

```
models/granite_4.2_8b_rust_lora/
├── adapter_config.json        # LoRA rank, alpha, target modules
├── adapter_model.safetensors  # Trained adapter weights
├── manifest.json              # Training metadata, hyperparameters, and duration
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── checkpoints/               # Periodic step checkpoints (if enabled)
    └── checkpoint-50/
```
