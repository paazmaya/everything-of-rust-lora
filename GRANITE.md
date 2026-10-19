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

# Resume interrupted training from the latest checkpoint:
uv run python train_granite.py --resume

# Resume from a specific checkpoint:
uv run python train_granite.py --resume-from-checkpoint models/granite_4.2_8b_rust_lora/checkpoints/checkpoint-50
```

### Resuming an Interrupted Run

If training is stopped or interrupted (e.g., via `Ctrl+C` or system sleep), simply add `--resume`:

```bash
uv run python train_granite.py --resume
```

The script automatically detects the latest checkpoint directory (`checkpoint-<step>`) and resumes optimizer, scheduler, and model states from that exact point.

To resume from a specific checkpoint:

```bash
uv run python train_granite.py --resume-from-checkpoint models/granite_4.2_8b_rust_lora/checkpoints/checkpoint-50
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
- **Epochs:** `1`
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
  --num-epochs          Number of training epochs (default: 1)
  --learning-rate       Optimizer learning rate (default: 0.0002)
  --load-in-4bit        Force 4-bit quantization (QLoRA)
  --data-file           Path to training JSONL dataset
  --output-dir          Directory to save final LoRA weights
  --checkpoint-dir      Directory for intermediate training checkpoints
  --save-steps          Number of update steps between saving checkpoints
  --save-total-limit    Maximum number of checkpoints to retain
  --save-strategy       Checkpoint save strategy (steps, epoch, no)
  --resume              Automatically resume training from latest checkpoint
  --resume-from-checkpoint
                        Path to checkpoint directory to resume training
  --wandb-project       Weights & Biases project name for tracking
  --wandb-entity        Weights & Biases team/entity name
  --trust-remote-code   Enable execution of custom remote code
```

## 6. Exporting to Unsloth Studio, Ollama & GGUF

### Unsloth Specific Optimizations for Granite 4.2 8B

1. **Deterministic Greedy Decoding:**
   As highlighted in Unsloth Granite documentation, Granite instruction models perform best with zero-temperature greedy generation:
   - `temperature=0.0, top_k=0, top_p=1.0`
   - Role Delimiters: `<|start_of_role|>system|user|assistant<|end_of_role|>` and `<|end_of_text|>`
2. **Post-Training SVD Rank Truncation:**
   Because Granite was trained with high rank ($r=64, \alpha=128$), you can optionally truncate the adapter down to $r=16$ using Singular Value Decomposition (SVD), shrinking the adapter by 75% with virtually zero loss in Rust code reasoning.
3. **Native Context Length:**
   Supports up to 131,072 tokens natively.

### Export Commands

```bash
# Export merged model for Unsloth Studio
uv run python scripts/11_export_unsloth.py \
    --model models/granite_rust_lora \
    --base ibm-granite/granite-4.2-8b \
    --mode merged-hf \
    --output-dir models/granite_4.2_8b_merged

# Export for Ollama
uv run python scripts/10_export_ollama.py \
    --model models/granite_rust_lora \
    --base ibm-granite/granite-4.2-8b \
    --name rust-granite-8b \
    --quant q4_k_m
```

This generates an Ollama Modelfile and packages the model for `ollama run rust-granite-8b`. For complete quantization and SVD compression details, see [QUANTIZIZE.md](QUANTIZIZE.md).

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
