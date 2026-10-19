# Amazon ALoDLM-8B LoRA Training & Deployment Guide

This document provides complete instructions for fine-tuning a specialized Rust systems programming LoRA adapter on [amazon/ALoDLM-8B](https://huggingface.co/amazon/ALoDLM-8B). It explains the Adaptively Looped Diffusion Language Model architecture, how standard causal LoRA fine-tuning aligns the underlying Qwen3 backbone for systems and embedded Rust, hardware-aware profiles, manual model downloading, and deployment workflows.

## Architecture Overview

**Amazon ALoDLM-8B** is a diffusion language model developed by Amazon Science. It generates text and code by repeatedly executing shared transformer layers, retaining latent states for unresolved tokens, and adaptively allocating compute via a learned exit gate.

- **Base Architecture:** `Qwen3ForCausalLM` backbone (`model_type: "qwen3"`)
- **Model Scale:** 8.19B parameters, 36 hidden layers, hidden dimension $4096$, intermediate size $12288$, 32 attention heads, 8 key-value heads (Grouped Query Attention)
- **Attention Type:** All 36 layers utilize standard full attention (no hybrid linear-attention blocks)
- **Context Length:** 40,960 tokens native context window (`max_position_embeddings: 40960`)
- **Precision:** Native 16-bit `bfloat16` weights
- **Chat Template:** Standard ChatML syntax (`<|im_start|>` and `<|im_end|>`) via `Qwen2Tokenizer`
- **Recurrence Mechanism:** Inference-time adaptive looping between layers 10 and 26 (`loop_start: 10`, `loop_end: 26`, `max_depth: 4`), paired with `exit_gate.pt` and `alodlm_config.json`

### LoRA Adapter Design

Because the base model weights represent a standard 36-layer Qwen3 transformer backbone, LoRA fine-tuning in [train_alodlm.py](train_alodlm.py) targets all 7 attention and feed-forward projection matrices across every layer:

- **Self-Attention Projections:** `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **MLP Feed-Forward Projections:** `gate_proj`, `up_proj`, `down_proj`

This ensures that representations refined during both standard causal decoding and recurrent diffusion passes are informed by Rust systems and embedded development knowledge.

## 1. Manual Model Download

To download the base model files manually into your local workspace directory prior to training:

```bash
hf download amazon/ALoDLM-8B \
  --local-dir ./models/amazon__ALoDLM-8B
```

Or using Git LFS:

```bash
git clone https://huggingface.co/amazon/ALoDLM-8B ./models/amazon__ALoDLM-8B
```

Once downloaded locally, point `--model-path` to the local directory:

```bash
uv run python train_alodlm.py --model-path ./models/amazon__ALoDLM-8B
```

## 2. Quick Start: Training

### Step 1: Preflight Validation (Dry-Run)

Validate your environment, GPU/MPS hardware availability, Python dependencies, and dataset integrity without downloading or loading base model weights:

```bash
uv run python train_alodlm.py --dry-run
```

This verifies:

- Apple Silicon Metal Performance Shaders (MPS) or NVIDIA CUDA availability
- System memory capacity (minimum 16.0 GB on Mac, 12.0 GB on CUDA)
- Core dependencies (`transformers`, `peft`, `trl`, `torch`)
- Training dataset schema and formatting in [data/datasets/train.jsonl](data/datasets/train.jsonl)

### Step 2: Start Training

Run with the default profile (`mac_48gb` on Apple Silicon macOS, `rtx3090_4090` on Linux/CUDA):

```bash
uv run python train_alodlm.py
```

> **12 GB GPUs (RTX 4070 / 4070 Ti / 3060):** on Windows the default profile is `rtx4070` (1024-token sequences, 4-bit QLoRA). On Linux the default `rtx3090_4090` profile assumes 24 GB VRAM and fails with `Unsloth: No or negligible GPU memory available for fused cross entropy.` on 12 GB cards, so pass `--hardware-profile rtx4070` explicitly there. Close other GPU-heavy apps (browsers, Docker Desktop, etc.) to free VRAM.

Outputs the trained LoRA adapter to `models/alodlm_8b_rust_lora`.

## 3. Hardware Profiles

The training script provides pre-tuned hardware profiles selectable via `--hardware-profile`:

| Profile        | Hardware / Device            | Memory      | Batch Size | Grad Accum | Max Seq Len | Mode / Precision           |
| -------------- | ---------------------------- | ----------- | ---------- | ---------- | ----------- | -------------------------- |
| `mac_48gb`     | Apple Silicon Mac (M-Series) | 32GB–48GB   | 1          | 8          | 1024        | 16-bit bf16 + Checkpoint   |
| `mac_fast`     | Apple Silicon Mac (M-Series) | 16GB–36GB   | 1          | 4          | 512         | 16-bit bf16 fast test      |
| `rtx4070`      | NVIDIA RTX 4070 / 3060       | 12GB VRAM   | 1          | 8          | 1024        | 4-bit QLoRA                |
| `rtx3090_4090` | NVIDIA RTX 3090 / 4090       | 24GB VRAM   | 1          | 8          | 2048        | 4-bit QLoRA (**CUDA Def**) |
| `rtx6000`      | NVIDIA RTX 6000 Ada          | 48GB VRAM   | 2          | 4          | 4096        | 16-bit bf16 LoRA           |
| `a100`         | NVIDIA A100 / H100           | 40GB / 80GB | 2          | 4          | 4096        | 16-bit bf16 LoRA           |
| `dev`          | Development / Testing        | Any         | 1          | 1          | 512         | 4-bit QLoRA minimal        |

### Profile Usage Examples

```bash
# Recommended for Apple Silicon Mac (fastest in-memory training, 512 seq length):
uv run python train_alodlm.py --hardware-profile mac_fast --num-epochs 1

# Train on Apple Silicon Mac with 1024 seq length:
uv run python train_alodlm.py --hardware-profile mac_48gb

# Train on 24GB VRAM GPU (RTX 3090/4090) with 4-bit QLoRA:
uv run python train_alodlm.py --hardware-profile rtx3090_4090

# Train on 12GB VRAM GPU (RTX 4070):
uv run python train_alodlm.py --hardware-profile rtx4070

# Train on high-memory workstation (RTX 6000 Ada or A100):
uv run python train_alodlm.py --hardware-profile rtx6000

# Quick development smoke test:
uv run python train_alodlm.py --hardware-profile dev
```

### Apple Silicon Performance & Memory Optimization

On Apple Silicon (M-Series), ALoDLM-8B in 16-bit BF16 consumes ~16.4 GB of model weights. With 32GB–48GB unified memory, the model fits **100% in-memory** with zero SSD disk offload:

1. **In-Memory MPS Device Mapping:** The engine automatically bypasses Accelerate's conservative `device_map="auto"` heuristic on macOS and loads the model directly to `mps:0`. This avoids SSD layer swapping which otherwise causes ~488s/step latency and NaN gradients.
2. **Use `mac_fast` Profile:** Drops sequence length to 512 tokens and gradient accumulation to 4. Due to quadratic attention scaling ($O(L^2)$), this speeds up each training step by ~40%–50%.
3. **Set Epochs to 1 or 2:** Training across all 42,197 examples for 1 epoch provides 5,275 optimizer updates (at gradient accumulation 8) or 10,550 updates (at accumulation 4), which is sufficient for domain alignment while completing in hours instead of days.
4. **PyTorch MPS Allocator Setting:** The engine automatically sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to permit allocations across all available unified memory without artificial 70% threshold throttling.

## 4. Checkpoint Resumption & Reliability

Training automatically saves periodic checkpoints (every 25–50 steps depending on profile) into `models/alodlm_8b_rust_lora/checkpoints/`.

### Resume from Latest Checkpoint

If a training run is interrupted by system reboot, power outage, or `Ctrl+C`:

```bash
uv run python train_alodlm.py --resume
```

The script locates the most recent valid checkpoint containing `trainer_state.json` and resumes training without repeating earlier epochs.

### Resume from Specific Checkpoint

```bash
uv run python train_alodlm.py --resume-from-checkpoint models/alodlm_8b_rust_lora/checkpoints/checkpoint-50
```

## 5. CLI Options Reference

The script supports all standardized CLI flags implemented across the training framework:

```text
--dry-run                     Run environment, hardware, dependency, and data preflight checks without training.
--hardware-profile PROFILE    Select a predefined hardware execution profile.
--model-path PATH             HuggingFace hub repo ID or local model path [default: amazon/ALoDLM-8B].
--output-dir DIR              Directory where final LoRA adapter will be saved [default: models/alodlm_8b_rust_lora].
--checkpoint-dir DIR          Directory for intermediate checkpoints [default: <output-dir>/checkpoints].
--data-file PATH              Path to instruction dataset [default: data/datasets/train.jsonl].
--batch-size INT              Per-device training batch size.
--gradient-accumulation-steps INT  Number of gradient accumulation steps.
--max-seq-length INT          Maximum sequence length in tokens.
--num-epochs INT              Number of training epochs [default: 3].
--learning-rate FLOAT         Optimizer learning rate [default: 2e-4].
--load-in-4bit                Force 4-bit NF4 quantization (QLoRA).
--resume                      Automatically detect and resume from the latest valid checkpoint.
--resume-from-checkpoint PATH Resume training from an exact checkpoint directory.
--save-steps INT              Interval of training steps between checkpoint saves.
--save-total-limit INT        Maximum number of retained checkpoints [default: 3].
--save-strategy {steps,epoch,no} Checkpoint saving strategy [default: steps].
--offload-folder DIR          Directory for offloading parameters to disk/CPU if memory is constrained.
--trust-remote-code           Allow custom model architectures to execute code from HuggingFace Hub.
--wandb-project NAME          Weights & Biases project name for experiment tracking.
--wandb-entity NAME           Weights & Biases team/entity.
```

## 6. Deployment & Serving

Once training finishes, the adapter in `models/alodlm_8b_rust_lora` can be used in two ways:

### Mode A: Standard Causal Generation (vLLM / Transformers)

Merge the LoRA adapter with the base Qwen3 backbone:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "models/amazon__ALoDLM-8B", torch_dtype="auto", device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "models/alodlm_8b_rust_lora")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("models/alodlm_8b_rust_merged")

tokenizer = AutoTokenizer.from_pretrained("models/amazon__ALoDLM-8B")
tokenizer.save_pretrained("models/alodlm_8b_rust_merged")
```

The merged model can be served with vLLM, Ollama (via GGUF export), or Hugging Face pipelines.

### Mode B: Recurrent Diffusion Generation (`alodlm-generate`)

To run inference with Amazon's recurrent latent refinement and exit gate, copy the merged weights back into the ALoDLM layout containing `exit_gate.pt` and `alodlm_config.json`:

```bash
alodlm-generate \
  --model models/alodlm_8b_rust_merged \
  --device cuda \
  --precision bf16 \
  --mode entropy \
  --q 0.5 \
  --tau 0.4 \
  --prompt "Write an asynchronous Rust actor pattern using tokio channels."
```

## Testing

```powershell
uv run python scripts/eval_version_knowledge.py --benchmark-only
uv run python scripts/eval_version_knowledge.py --model-path H:\amazon__ALoDLM-8B\ --batch-size 4 --max-new-tokens 256 --output-file eval_base_v2.json
uv run python scripts/eval_version_knowledge.py --model-path H:\amazon__ALoDLM-8B\ --adapter-path .\models\alodlm_8b_rust_lora\checkpoints\checkpoint-7500\ --batch-size 4 --max-new-tokens 256 --output-file eval_checkpoint7500_v2.json
uv run python scripts/eval_version_knowledge.py --compare-files eval_base_v2.json eval_checkpoint7500_v2.json --output-file eval_comparison_v2.json
```

The evaluator uses the training ChatML format and system prompt explicitly: the
ALoDLM tokenizer may have no `chat_template`. Both runs must use identical settings.
If answers hit the token limit, inspect the saved finish reasons and rerun **both**
models with the same larger `--max-new-tokens` and new output paths.

For an existing report, no weights are needed to rescore unchanged questions:

```powershell
uv run python scripts/eval_version_knowledge.py --rescore-file eval_checkpoint7500.json --output-file eval_checkpoint7500_rescored.json
```

Version 2 replaces substring recall with required-fact rubrics and separate partial
coverage. Automated passes do not certify every additional claim; inspect responses
for hallucinations. Five clarified questions differ from the original benchmark
and need fresh inference. Historical reports without generation metadata cannot
establish truncation or a base-versus-adapter improvement/regression. Check the
checkpoint's own `trainer_state.json` to confirm its epoch count; the directory's
step number alone is insufficient. Regenerated local splits are not evidence of
the dataset used by this checkpoint.
