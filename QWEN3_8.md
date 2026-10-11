# Qwen 3.8 27B LoRA Training & Deployment Guide

This document provides comprehensive instructions for training a Rust systems programming LoRA adapter on [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). It covers the hybrid architectural design of the 27B model family, target module specifications for universal 27B compatibility, macOS Apple Silicon (MPS + memory offloading) configuration, and checkpoint resumption.

## Architecture Overview & Compatibility

### Why a Dedicated Pipeline is Required for Qwen 3.8 27B

The Qwen 3.8 27B model introduces substantial architectural advancements over compact 4B models:

| Architectural Metric        | Qwen 3.5 4B                 | Qwen 3.8 27B                     |
| :-------------------------- | :-------------------------- | :------------------------------- |
| **Total Layers**            | 32 layers                   | **64 layers**                    |
| **Hidden Dimension ($d$)**  | 2,560                       | **5,120**                        |
| **Intermediate Size (FFN)** | 9,216                       | **17,408**                       |
| **Attention Architecture**  | Hybrid (24 linear + 8 full) | **Hybrid (48 linear + 16 full)** |
| **Vocabulary Size**         | 248,320                     | **248,320**                      |
| **Context Window**          | 32,768 tokens               | **262,144 tokens (native)**      |

### Mathematical Incompatibility with 4B Adapters

LoRA modifies weight matrices $W \in \mathbb{R}^{d_{out} \times d_{in}}$ via low-rank decomposition:

$$\Delta W = \frac{\alpha}{r} \cdot (B \cdot A)$$

where $A \in \mathbb{R}^{r \times d_{in}}$ and $B \in \mathbb{R}^{d_{out} \times r}$.

- A LoRA adapter trained on a 4B model has $d_{in}=2560$ and 32 layer tensors.
- Qwen 3.8 27B requires $d_{in}=5120$ and 64 layer tensors.
- **Loading a 4B adapter into a 27B model fails unconditionally with tensor dimension mismatch errors.**

### Universal 27B Adapter Compatibility

To ensure the adapter can be loaded by **any** model using the Qwen 3.8 27B architecture (including instruct, base, and fine-tuned community variations), `train_qwen3_8.py` targets all 12 linear and attention projections:

1. **Full-Attention Projections (16 layers):** `q_proj`, `k_proj`, `v_proj`, `o_proj`
2. **Linear-Attention Projections (48 layers):** `in_proj_a`, `in_proj_b`, `in_proj_qkv`, `in_proj_z`, `out_proj`
3. **MLP Feed-Forward Blocks (all 64 layers):** `gate_proj`, `up_proj`, `down_proj`

## 1. Quick Start: Training on macOS

### Step 1: Preflight Validation (Dry-Run)

Validate your environment, Metal Performance Shaders (MPS), dependencies, and dataset integrity:

```bash
uv run python train_qwen3_8.py --dry-run
```

This verifies:

- Apple Silicon MPS acceleration or CUDA availability
- System unified memory capacity
- Core fine-tuning dependencies (`transformers`, `peft`, `trl`, `torch`)
- Dataset schema in `data/datasets/train.jsonl`

### Step 2: Start Training

Run with the default macOS profile (`mac_48gb`):

```bash
uv run python train_qwen3_8.py
```

This trains with:

- 16-bit `bfloat16` precision
- Gradient checkpointing enabled to save activation memory
- Offload folder `models/qwen3_8_offload` for layer weight staging
- Checkpointing every 25 steps to `models/qwen3_8_27b_rust_lora/checkpoints/`
- Output adapter saved to `models/qwen3_8_27b_rust_lora/`

OR

use [DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU](https://huggingface.co/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU/) as a base model.

Since the network might get unstable, download in more controlled fashion:

```sh
hf download DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU \
 --local-dir ./models/DavidAU__Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU \
 --include "*.json"

hf download DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU \
 --local-dir ./models/DavidAU__Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU \
 chat_template.jinja model-mtp-restored.safetensors

hf download DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU \
 --local-dir ./models/DavidAU__Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU \
  model-00001-of-00012.safetensors
# Then download the rest 11 safetensors files one by one
```

### Step 3: Resuming an Interrupted Run

If training is stopped or interrupted (e.g., via `Ctrl+C` or system sleep), simply add `--resume`:

```bash
uv run python train_qwen3_8.py --resume
```

The script automatically detects the latest checkpoint directory (`checkpoint-<step>`) and resumes optimizer, scheduler, and model states from that exact point.

To resume from a specific checkpoint:

```bash
uv run python train_qwen3_8.py --resume-from-checkpoint models/qwen3_8_27b_rust_lora/checkpoints/checkpoint-50
```

## 2. Hardware Profiles

`train_qwen3_8.py` provides built-in hardware profiles via `--hardware-profile`:

| Profile        | Platform / GPU    | Memory      | Batch Size | Grad Accum | Effective Batch | Max Seq Len | Mode           | Notes                     |
| :------------- | :---------------- | :---------- | :--------- | :--------- | :-------------- | :---------- | :------------- | :------------------------ |
| `mac_48gb`     | Apple Silicon Mac | 32–48GB RAM | 1          | 16         | 16              | 512         | bf16 + Offload | **Default on macOS**      |
| `mac_dev`      | Apple Silicon Mac | Any         | 1          | 1          | 1               | 256         | bf16 + Offload | Quick test / verification |
| `rtx3090_4090` | RTX 3090 / 4090   | 24GB VRAM   | 1          | 8          | 8               | 2048        | 4-bit QLoRA    | Default on Linux/Windows  |
| `rtx6000`      | RTX 6000 Ada      | 48GB VRAM   | 1          | 8          | 8               | 4096        | 16-bit bf16    | Workstations              |
| `a100_80gb`    | A100 / H100       | 80GB VRAM   | 2          | 4          | 8               | 4096        | 16-bit bf16    | High-throughput server    |

### Usage Examples

```bash
# Default Mac profile
uv run python train_qwen3_8.py

# Quick iteration test on Mac (256 sequence length)
uv run python train_qwen3_8.py --hardware-profile mac_dev

# Resuming interrupted run
uv run python train_qwen3_8.py --resume

# Training on an 80GB GPU node
uv run python train_qwen3_8.py --hardware-profile a100_80gb
```

## 3. macOS Optimization Details

Training a 27B model on unified memory (32GB–48GB) requires dedicated adjustments:

1. **Precision & Memory Footprint:**
   - At 16-bit (`bfloat16`), the base model weights alone require $\approx 54\text{ GB}$.
   - The `mac_48gb` profile enables `offload_folder="models/qwen3_8_offload"`, which streams inactive layer weights between disk and unified RAM.
2. **Optimizer Selection:**
   - Standard 8-bit Adam (`adamw_8bit`) relies on CUDA-specific assembly kernels that fail on macOS.
   - The training engine automatically selects `adamw_torch` on Apple Silicon, preventing crashes.
3. **Gradient Checkpointing:**
   - Reduces activation memory consumption by recomputing intermediate activations during the backward pass instead of storing all 64 layers in memory.
4. **Sequence Length:**
   - Set to 512 tokens by default on Mac to limit peak attention tensor allocations while fully accommodating Rust code functions and documentation snippets.

## 4. CLI Options Reference

```
options:
  -h, --help            Show help message and exit
  --dry-run             Validate environment and dataset without starting training
  --hardware-profile    Hardware profile (mac_48gb, mac_dev, rtx3090_4090, rtx6000, a100_80gb)
  --model-path          Model identifier on HuggingFace Hub or local path [default: Qwen/Qwen3.8-27B]
  --batch-size          Per-device training batch size
  --gradient-accumulation-steps
                        Number of gradient accumulation steps
  --max-seq-length      Maximum sequence length in tokens
  --num-epochs          Number of training epochs (default: 3)
  --learning-rate       Optimizer learning rate (default: 0.0002)
  --load-in-4bit        Force 4-bit quantization (QLoRA)
  --data-file           Path to training dataset JSONL file
  --output-dir          Directory to save final LoRA weights [default: models/qwen3_8_27b_rust_lora]
  --checkpoint-dir      Directory for intermediate checkpoints
  --resume              Automatically resume from the latest checkpoint
  --resume-from-checkpoint
                        Path to specific checkpoint directory (or 'latest')
  --offload-folder      Directory for CPU/disk weight offloading
  --wandb-project       Weights & Biases project name
  --wandb-entity        Weights & Biases team/entity name
  --trust-remote-code   Enable custom remote code execution
```

## 5. Output Artifacts & Deployment

Upon training completion, weights are stored in:

```
models/qwen3_8_27b_rust_lora/
├── adapter_config.json        # LoRA rank (16), alpha (16), target modules
├── adapter_model.safetensors  # Trained 27B LoRA adapter weights
├── manifest.json              # Hyperparameters, model metadata, and training duration
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── checkpoints/               # Step checkpoints
    ├── checkpoint-25/
    ├── checkpoint-50/
    └── ...
```

### Loading the Adapter in Python

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "Qwen/Qwen3.8-27B"
adapter_path = "models/qwen3_8_27b_rust_lora"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter_path)
```

### Merging, MTP GGUF Quantization & Unsloth Studio

To merge the adapter with either `Qwen/Qwen3.8-27B` or `DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU` and create an extreme 2-bit $IQ2\_XS$ quantization with Multi-Token Prediction (MTP) for Unsloth Studio:

- Export for Unsloth Studio: `uv run python scripts/11_export_unsloth.py --model models/qwen3_8_27b_rust_lora --base Qwen/Qwen3.8-27B --mode merged-hf`
- Complete step-by-step MTP GGUF and importance matrix guide: See [QUANTIZIZE.md](QUANTIZIZE.md#4-showcase-extreme-iq2_xs-merge-with-mtp-for-unsloth-studio).
