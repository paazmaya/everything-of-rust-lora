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

# Resume interrupted training from the latest checkpoint:
uv run python train_qwen.py --resume

# Resume from a specific checkpoint:
uv run python train_qwen.py --resume-from-checkpoint models/qwen3_5_4b_rust_lora/checkpoints/checkpoint-50
```

### Resuming an Interrupted Run

If training is stopped or interrupted (e.g., via `Ctrl+C` or system sleep), simply add `--resume`:

```bash
uv run python train_qwen.py --resume
```

The script automatically detects the latest checkpoint directory (`checkpoint-<step>`) and resumes optimizer, scheduler, and model states from that exact point.

To resume from a specific checkpoint:

```bash
uv run python train_qwen.py --resume-from-checkpoint models/qwen3_5_4b_rust_lora/checkpoints/checkpoint-50
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

## 6. Exporting, Quantization & Unsloth Studio Deployment

### Unsloth Specific Optimizations for Qwen 3.5 4B

1. **Reasoning Activation in Compact Models:**
   On compact Qwen 3.5 variants (0.8B to 9B), reasoning is disabled by default. When running `llama-server` or importing into Unsloth Studio, enable thinking mode via template kwargs:
   ```bash
   --chat-template-kwargs '{"enable_thinking":true}'
   ```
2. **Precise Rust Coding Parameters:**
   - **Thinking Mode (Coding):** `temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0`
   - **Non-Thinking Mode:** `temperature=0.7, top_p=0.80, presence_penalty=1.5`
3. **Sensitive Tensor Protection:**
   The hybrid attention/SSM architecture of Qwen 3.5 has high sensitivity in `ssm_out` and `attn_gate`. Do not use aggressive 2-bit quantization on these layers.
4. **Recommended Quantization:** `UD-Q4_K_XL` (~3.2GB) balances speed and accuracy for 8GB VRAM or Apple Silicon devices.

### Export Commands

```bash
# Export merged model for Unsloth Studio
uv run python scripts/11_export_unsloth.py \
    --model models/qwen3_5_4b_rust_lora \
    --base Qwen/Qwen3.5-4B \
    --mode merged-hf \
    --output-dir models/qwen3_5_4b_merged

# Serve with llama-server
llama-server \
    --model models/qwen3_5_4b_merged/qwen3.5-4b-UD-Q4_K_XL.gguf \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
    --chat-template-kwargs '{"enable_thinking":true}' \
    --port 8001
```

For detailed multi-model quantization recipes and Dynamic 3.0 benchmarks, see [QUANTIZIZE.md](QUANTIZIZE.md).
