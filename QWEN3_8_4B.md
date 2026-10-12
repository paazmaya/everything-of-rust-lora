# Qwen 3.8 4B Distill LoRA Training & Deployment Guide

This document provides complete instructions for fine-tuning a specialized Rust systems programming LoRA adapter on [empero-ai/Qwen3.8-4B-Distill](https://huggingface.co/empero-ai/Qwen3.8-4B-Distill). It explains the hybrid linear + full attention architecture, why in-memory 4B training resolves the severe disk-offloading bottlenecks observed with 27B models on Apple Silicon, hardware-aware profiles, and deployment instructions.

## Architecture Overview & Performance Comparison

### Why 27B Offloading Was Slow vs In-Memory 4B Throughput

When fine-tuning Qwen 3.8 27B on Apple Silicon Macs, the 27B parameter footprint in 16-bit BF16 (~54 GB) exceeds the unified memory budget available to PyTorch Metal Performance Shaders (MPS). This forces base model layers onto disk offloading.

Each training step then requires swapping 64 layer weight tensors across SSD storage during forward and backward passes, leading to severe I/O latency:

$$\text{Step Latency (27B Offloaded)} \approx 1,278\text{ s/it } (\approx 21.3\text{ minutes/step})$$

In contrast, **Qwen 3.8 4B Distill** requires only ~8.0 GB for model weights in 16-bit BF16:

$$\text{Memory Overhead (4B BF16)} \approx 8.0\text{ GB (Weights)} + 2.5\text{ GB (LoRA + Activations)} \approx 10.5\text{ GB Unified RAM}$$

Because the model, gradients, optimizer states, and activations fit 100% inside unified RAM with zero disk offloading, step times drop from ~1,278 seconds to single-digit seconds (a ~500x speedup).

### Architectural Metric Comparison

| Architectural Metric        | Qwen 3.5 4B            | Qwen 3.8 4B Distill             | Qwen 3.8 27B                 |
| :-------------------------- | :--------------------- | :------------------------------ | :--------------------------- |
| **Base Model Identifier**   | `Qwen/Qwen3.5-4B`      | `empero-ai/Qwen3.8-4B-Distill`  | `Qwen/Qwen3.8-27B`           |
| **Total Layers**            | 32 layers              | **32 layers**                   | 64 layers                    |
| **Hidden Dimension ($d$)**  | 2,560                  | **2,560**                       | 5,120                        |
| **Intermediate Size (FFN)** | 9,216                  | **9,216**                       | 17,408                       |
| **Attention Architecture**  | 32 standard layers     | **Hybrid (24 linear + 8 full)** | Hybrid (48 linear + 16 full) |
| **Full Attention Layers**   | 32 layers              | **8 layers** (every 4th layer)  | 16 layers (every 4th layer)  |
| **Linear Attention Layers** | 0 layers               | **24 layers**                   | 48 layers                    |
| **Target Projections**      | 7 standard projections | **All 12 hybrid projections**   | All 12 hybrid projections    |
| **Context Window**          | 32,768 tokens          | **262,144 tokens (native)**     | 262,144 tokens (native)      |
| **Disk Offloading Needed**  | No                     | **No (0 offload required)**     | Yes (on unified RAM < 64GB)  |

### Universal Hybrid Target Projections

Because `empero-ai/Qwen3.8-4B-Distill` utilizes the Qwen 3.8 hybrid attention design, standard 7-target LoRA scripts (which only target `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) completely miss 75% of the attention layers (24 out of 32 layers).

[train_qwen3_8_4b.py](train_qwen3_8_4b.py) targets all 12 linear and attention projections:

1. **Full-Attention Projections (8 layers):** `q_proj`, `k_proj`, `v_proj`, `o_proj`
2. **Linear-Attention Projections (24 layers):** `in_proj_a`, `in_proj_b`, `in_proj_qkv`, `in_proj_z`, `out_proj`
3. **MLP Feed-Forward Blocks (all 32 layers):** `gate_proj`, `up_proj`, `down_proj`

## 1. Quick Start: Training

```sh
hf download empero-ai/Qwen3.8-4B-Distill \
  --local-dir ./models/empero-ai__Qwen3.8-4B-Distill
```

### Step 1: Preflight Validation (Dry-Run)

Validate your local environment, Metal Performance Shaders (MPS) or CUDA availability, dependencies, and dataset integrity before initiating training:

```bash
uv run python train_qwen3_8_4b.py --dry-run
```

This verifies:

- Apple Silicon MPS acceleration or CUDA GPU availability
- System memory capacity (minimum 8.0 GB for Mac, 12.0 GB for CUDA)
- Core fine-tuning dependencies (`transformers`, `peft`, `trl`, `torch`)
- Training dataset schema in [data/datasets/train.jsonl](data/datasets/train.jsonl)

### Step 2: Start Training

Run with the default profile (`mac_native` on Apple Silicon, `rtx4070` on CUDA):

```bash
uv run python train_qwen3_8_4b.py
```

This executes training with:

- 16-bit `bfloat16` precision
- In-memory execution without disk offload
- Gradient checkpointing enabled to minimize activation memory
- Effective batch size of 8 (`batch_size=1`, `gradient_accumulation_steps=8`)
- Checkpointing every 25 steps to [models/qwen3_8_4b_rust_lora/checkpoints/](models/qwen3_8_4b_rust_lora/checkpoints/)
- Final adapter saved to [models/qwen3_8_4b_rust_lora](models/qwen3_8_4b_rust_lora)

### Step 3: Resuming an Interrupted Run

If training is stopped or interrupted (e.g., via `Ctrl+C` or system sleep), resume seamlessly with `--resume`:

```bash
uv run python train_qwen3_8_4b.py --resume
```

The script automatically locates the latest valid checkpoint directory (`checkpoint-<step>`) and resumes optimizer, scheduler, and RNG states.

To resume from a specific checkpoint:

```bash
uv run python train_qwen3_8_4b.py --resume-from-checkpoint models/qwen3_8_4b_rust_lora/checkpoints/checkpoint-50
```

## 2. Hardware Profiles

Select a profile tailored to your available hardware using `--hardware-profile`:

| Profile        | Target Platform    | Memory    | Batch | Grad Accum | Context | Description                                         |
| :------------- | :----------------- | :-------- | :---- | :--------- | :------ | :-------------------------------------------------- |
| `mac_native`   | Apple Silicon Mac  | 16GB–48GB | 1     | 8          | 2048    | 100% in-memory 16-bit BF16 LoRA (**macOS default**) |
| `mac_fast`     | Apple Silicon Mac  | 16GB–48GB | 1     | 4          | 1024    | Fast iteration cycle (1024 tokens)                  |
| `rtx4070`      | RTX 4070 / 3060    | 12GB      | 1     | 4          | 2048    | 16-bit BF16 LoRA (**CUDA default**)                 |
| `rtx3090_4090` | RTX 3090 / 4090    | 24GB      | 2     | 4          | 4096    | 16-bit BF16 with 4K context window                  |
| `a100`         | NVIDIA A100 / H100 | 40GB–80GB | 4     | 2          | 8192    | High-throughput data center profile                 |
| `dev`          | Any hardware       | Minimal   | 1     | 1          | 512     | Fast test profile (512 tokens, 5 save steps)        |

### Usage Examples

```bash
# Default Apple Silicon Mac profile
uv run python train_qwen3_8_4b.py

# Faster iteration profile on Mac
uv run python train_qwen3_8_4b.py --hardware-profile mac_fast

# CUDA workstation profile (RTX 4090)
uv run python train_qwen3_8_4b.py --hardware-profile rtx3090_4090

# Quick development verification
uv run python train_qwen3_8_4b.py --hardware-profile dev

# Train with a locally cloned model repository
uv run python train_qwen3_8_4b.py --model-path ./models/empero-ai__Qwen3.8-4B-Distill
```

## 3. Hyperparameters & Configuration

Configurations are managed in [train_qwen3_8_4b.py](train_qwen3_8_4b.py):

- **Model ID:** `empero-ai/Qwen3.8-4B-Distill`
- **Output Directory:** `models/qwen3_8_4b_rust_lora`
- **Precision:** `bfloat16` (or `float32` fallback if MPS unsupported)
- **LoRA Parameters:** $r=16$, $\alpha=16$, dropout $0.0$, bias `"none"`
- **Learning Rate:** $2 \times 10^{-4}$ with cosine scheduler and 10 warmup steps
- **Epochs:** 3 epochs over [data/datasets/train.jsonl](data/datasets/train.jsonl)
- **Prompt Format:** Native ChatML format (`<|im_start|>`, `<|im_end|>`) with Rust system prompt

## 4. Exporting & Deployment

Once training is complete, the LoRA adapter can be merged and exported for Ollama, llama.cpp, or vLLM inference.

### Ollama Deployment

Generate an Ollama Modelfile and register the fine-tuned model:

```bash
uv run python scripts/09_create_modelfile.py \
  --model models/qwen3_8_4b_rust_lora \
  --name rust-qwen3-8-4b \
  --base qwen3_8

uv run python scripts/10_export_ollama.py \
  --model models/qwen3_8_4b_rust_lora \
  --name rust-qwen3-8-4b \
  --base empero-ai/Qwen3.8-4B-Distill
```

Test inference in Ollama:

```bash
ollama run rust-qwen3-8-4b "Write a thread-safe lock-free ring buffer in Rust using atomic operations."
```

### Merging and GGUF Conversion (llama.cpp)

Merge adapter weights into the base model and convert to GGUF format:

```bash
# 1. Merge adapter into base weights
uv run python scripts/11_export_unsloth.py \
  --model models/qwen3_8_4b_rust_lora \
  --base empero-ai/Qwen3.8-4B-Distill \
  --output-dir models/qwen3_8_4b_merged

# 2. Convert to GGUF
python3 external/llama.cpp/convert_hf_to_gguf.py models/qwen3_8_4b_merged \
  --outfile models/qwen3_8_4b_merged/qwen3.8-4b-f16.gguf \
  --outtype f16

# 3. Quantize to Q4_K_M for fast local inference
llama-quantize \
  models/qwen3_8_4b_merged/qwen3.8-4b-f16.gguf \
  models/qwen3_8_4b_merged/qwen3.8-4b-UD-Q4_K_XL.gguf \
  Q4_K_M
```

## 5. Troubleshooting & Verification

### Verifying MPS Acceleration

Run the dry-run command:

```bash
uv run python train_qwen3_8_4b.py --dry-run
```

Output should display:

```text
Apple Silicon Metal Performance Shaders (MPS) available.
Preflight checks passed: Environment and dataset are valid.
```

### Adapter Parameter Count

When model loading completes, the trainable parameter summary should reflect all 12 target projections:

```text
trainable params: ~32,000,000 || all params: ~4,200,000,000 || trainable%: ~0.76%
```

Both `linear_attn` and `self_attn` blocks will contain active `lora_A` and `lora_B` tensors on the active compute device (`mps`).
