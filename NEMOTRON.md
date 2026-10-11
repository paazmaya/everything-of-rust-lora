# Nemotron 3 Nano 4B LoRA Training Guide

This document provides comprehensive guidance for training a LoRA adapter on NVIDIA Nemotron 3 Nano 4B using the production-grade training script with validation, checkpointing, and hardware-aware configuration.

## Quick Start

### Validate Configuration (Recommended)

Before training, validate that your setup is correct without starting training:

```bash
uv run python train_nemotron3_nano.py --dry-run
```

This checks:

- ✅ CUDA and GPU availability
- ✅ All dependencies (unsloth, transformers, trl, peft, mamba-ssm)
- ✅ GPU memory requirements
- ✅ Training data exists and has correct schema
- ✅ Configuration files are valid

### Default Training

```bash
uv run python train_nemotron3_nano.py
```

Uses RTX 4070 hardware profile with default settings:

- Batch size: 1 per device
- Gradient accumulation: 4
- Max sequence length: 1024
- Learning rate: 2e-4
- Epochs: 3
- VRAM: ~12GB

## Hardware Profiles

The script includes predefined profiles optimized for different GPUs. Select via `--hardware-profile`:

| Profile     | GPU Type         | VRAM     | Batch Size | Grad Accum | Max Seq Len | Use Case                      |
| ----------- | ---------------- | -------- | ---------- | ---------- | ----------- | ----------------------------- |
| `rtx4070`   | NVIDIA RTX 4070  | 12GB     | 1          | 4          | 1024        | 💻 **Default** - Consumer GPU |
| `rtx6000`   | NVIDIA RTX 6000  | 24GB     | 2          | 2          | 2048        | 🏢 Workstation                |
| `a100`      | NVIDIA A100      | 40GB     | 4          | 1          | 4096        | 🔬 Data center                |
| `a100_80gb` | NVIDIA A100 80GB | 80GB     | 8          | 1          | 4096        | 🚀 High-end                   |
| `dev`       | Any GPU          | Variable | 1          | 1          | 512         | 🐛 Development                |

### Use a Hardware Profile

```bash
# RTX 6000 (24GB)
uv run python train_nemotron3_nano.py --hardware-profile rtx6000

# A100 (40GB)
uv run python train_nemotron3_nano.py --hardware-profile a100

# Development (minimal resources)
uv run python train_nemotron3_nano.py --hardware-profile dev
```

### Override Profile Settings

Use CLI arguments to override hardware profile defaults:

```bash
# Use rtx4070 profile but with larger batch size
uv run python train_nemotron3_nano.py --hardware-profile rtx4070 --batch-size 2

# Use A100 profile but with custom learning rate
uv run python train_nemotron3_nano.py --hardware-profile a100 --learning-rate 1e-4

# Mix and match
uv run python train_nemotron3_nano.py \
  --hardware-profile rtx6000 \
  --max-seq-length 4096 \
  --num-epochs 5
```

## Inline Configuration & Hardware Profiles

The training script defines all defaults and hardware profiles inline in [train_nemotron3_nano.py](train_nemotron3_nano.py), eliminating external YAML configuration files.

### Key Configuration Constants

**Model Configuration**

- `DEFAULT_MODEL_NAME`: `NVIDIA/Nemotron-3-Nano-4B`
- `trust_remote_code`: Enabled via `--trust-remote-code` for models with custom architecture

**Training Hyperparameters**

- `max_seq_length`: Tokens per sample (default `1024`)
- `num_epochs`: Training epochs (default `3`)
- `learning_rate`: Adam learning rate (default `2e-4`)
- `warmup_steps`: Linear warmup steps (default `10`)
- `weight_decay`: L2 regularization (default `0.01`)
- `lr_scheduler_type`: Cosine annealing

**LoRA Settings**

- `r`: Rank (`16`, optimal for Nemotron 4B)
- `lora_alpha`: Scaling factor (`16`)
- `target_modules`: All 7 projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)

**Hardware Profiles**

- `rtx4070`: Batch size 1, grad accum 4, max seq 1024 (**Default**)
- `rtx6000`: Batch size 2, grad accum 2, max seq 2048
- `a100`: Batch size 4, grad accum 1, max seq 4096
- `a100_80gb`: Batch size 8, grad accum 1, max seq 4096
- `dev`: Batch size 1, grad accum 1, max seq 512 (minimal VRAM)

**Output Configuration**

- `output_dir`: Final LoRA adapter directory (`models/nemotron3_nano_rust_lora`)
- `checkpoint_dir`: Intermediate checkpoints (`models/nemotron3_nano_rust_lora/checkpoints`)
- `save_strategy`: Steps (every 50 steps, keeping last 3 checkpoints)

**Experiment Tracking**

- `wandb_project`: Optional W&B project (pass via `--wandb-project`)
- `manifest.json`: Saved automatically to output directory with full training metadata

## Training Examples

### Example 1: Default Training (RTX 4070)

```bash
uv run python train_nemotron3_nano.py
```

✅ Checks CUDA, GPU memory, dependencies, data schema
✅ Trains for 3 epochs with batch size 1, gradient accumulation 4
✅ Saves checkpoints every 50 steps
✅ Final model saved to `models/nemotron3_nano_rust_lora/`
✅ Training metadata saved to `models/nemotron3_nano_rust_lora/manifest.json`

### Example 2: Training with W&B Tracking

```bash
# First, authenticate W&B
wandb login

# Then train with tracking
uv run python train_nemotron3_nano.py \
  --wandb-project rust-lora \
  --wandb-entity your-team-name
```

Enables detailed experiment tracking in Weights & Biases.

### Example 3: Resume from Checkpoint

If training was interrupted (e.g., via `Ctrl+C` or system sleep), resume from the latest checkpoint:

```bash
uv run python train_nemotron3_nano.py --resume
```

Or resume from a specific checkpoint:

```bash
uv run python train_nemotron3_nano.py \
  --resume-from-checkpoint models/nemotron3_nano_rust_lora/checkpoints/checkpoint-50
```

### Example 4: Custom Hyperparameters

```bash
uv run python train_nemotron3_nano.py \
  --hardware-profile rtx6000 \
  --batch-size 4 \
  --max-seq-length 2048 \
  --num-epochs 5 \
  --learning-rate 1e-4 \
  --warmup-steps 50 \
  --verbose
```

### Example 5: Local Model with Remote Code

For local Nemotron models or custom variants:

```bash
uv run python train_nemotron3_nano.py \
  --model-path /path/to/local/model \
  --trust-remote-code
```

⚠️ **Note**: If the model uses `mamba-ssm`, install it first:

```bash
pip install mamba-ssm
```

### Example 6: Quick Test (Dev Profile)

For testing on limited hardware:

```bash
uv run python train_nemotron3_nano.py \
  --hardware-profile dev \
  --data-file data/datasets/train_mini.jsonl
```

Uses minimal batch size (1), seq length (512), and gradient accumulation (1) for fast iteration.

## CLI Arguments Reference

```
Model Arguments:
  --model-path MODEL_PATH              HuggingFace ID or local path
  --trust-remote-code                  Enable remote code execution

Hardware Arguments:
  --hardware-profile PROFILE           rtx4070|rtx6000|a100|a100_80gb|dev

Training Arguments:
  --batch-size BATCH_SIZE              Per-device batch size
  --gradient-accumulation-steps STEPS  Gradient accumulation for effective batch
  --max-seq-length LENGTH              Maximum sequence length
  --num-epochs EPOCHS                  Training epochs
  --learning-rate RATE                 Adam learning rate
  --warmup-steps STEPS                 Linear warmup steps

Data Arguments:
  --data-file PATH                     Path to training data (JSONL)

Output Arguments:
  --output-dir DIR                     LoRA adapter output directory
  --checkpoint-dir DIR                 Intermediate checkpoints directory
  --save-steps STEPS                   Steps between saving checkpoints
  --save-total-limit LIMIT             Maximum checkpoints to retain
  --save-strategy STRATEGY             Save strategy (steps, epoch, no)
  --resume                             Automatically resume from latest checkpoint
  --resume-from-checkpoint PATH        Resume from specific checkpoint

Tracking Arguments:
  --wandb-project PROJECT              W&B project name (enables tracking)
  --wandb-entity ENTITY                W&B team/entity name

Utility Arguments:
  --dry-run                            Validate without training
  --help                               Show help message
```

## Data Format

Training data must be JSONL (JSON Lines) format. Each line is a JSON object with:

```json
{
  "instruction": "Explain async/await in Rust",
  "input": "",
  "output": "Async/await is a programming pattern..."
}
```

### Required Fields

- `instruction` (string): Task description or question
- `input` (string): Optional context (can be empty string)
- `output` (string): Expected model response

### Example Data File

```bash
{"instruction": "What is Rust?", "input": "", "output": "Rust is a systems programming language..."}
{"instruction": "Explain async Rust", "input": "ESP32", "output": "On ESP32, async Rust allows..."}
{"instruction": "How do I...", "input": "context here", "output": "Here's how to..."}
```

### Validation

The script validates data before training:

- ✅ File exists at specified path
- ✅ Valid JSONL format
- ✅ Contains all required fields
- ✅ Not empty

If validation fails, clear error messages guide you to fix the issue.

## Output and Artifacts

### After Training Completes

```
models/nemotron3_nano_rust_lora/
├── adapter_config.json          # LoRA configuration
├── adapter_model.bin            # LoRA weights
├── pytorch_model.bin            # LoRA weights (backup)
├── special_tokens_map.json      # Tokenizer mappings
├── tokenizer.json               # BPE tokenizer
├── tokenizer.model              # Tokenizer model
├── tokenizer_config.json        # Tokenizer config
├── manifest.json                # Training metadata
└── checkpoints/
    ├── checkpoint-50/
    ├── checkpoint-100/
    └── checkpoint-150/          # Latest checkpoint
```

### manifest.json Contents

```json
{
  "timestamp": "2026-09-07T12:34:56.789123",
  "model": "NVIDIA/Nemotron-3-Nano-4B",
  "hardware_profile": "rtx4070",
  "variant": "3-Nano",
  "config": {
    "training": {
      "num_epochs": 3,
      "learning_rate": 0.0002,
      "batch_size": 1,
      "gradient_accumulation_steps": 4,
      "max_seq_length": 1024
    },
    "lora": {
      "r": 16,
      "lora_alpha": 16
    }
  }
}
```

This manifest enables artifact tracking and reproducibility (aligned with NVIDIA Nemotron Lightning best practices).

## Checkpointing

Checkpoints are saved every 50 steps during training. Use `--resume-from-checkpoint` to continue from any checkpoint.

### Why Checkpointing Matters

- **Interruptions**: If training is interrupted, resume without losing progress
- **Experimentation**: Save intermediate models to compare
- **Recovery**: Fallback if later steps degrade performance
- **Reproducibility**: Each checkpoint includes exact training state

### Checkpoint File Structure

```
checkpoints/checkpoint-50/
├── optimizer.pt
├── scheduler.pt
├── training_args.bin
├── trainer_state.json
└── ...
```

## Troubleshooting

### CUDA Not Available

```
ERROR: CUDA is not available. GPU training is required for this script.
```

**Solution**: This script requires GPU training. If on macOS:

- Use a cloud provider (Colab, Unsloth Studio, Lambda Labs)
- Use WSL2 with GPU support on Windows
- Install NVIDIA GPU drivers on Linux

### Out of Memory (OOM)

```
RuntimeError: CUDA out of memory
```

**Solutions** (in order of impact):

1. Use smaller batch size: `--batch-size 1`
2. Reduce sequence length: `--max-seq-length 512`
3. Use dev profile: `--hardware-profile dev`
4. Use smaller model: No standard 4B model is smaller than Nemotron 3 Nano

### Missing Dependencies

```
ERROR: Missing dependencies: mamba-ssm
```

**Solution**:

```bash
pip install mamba-ssm
```

This is only needed if using `--trust-remote-code` with certain model variants.

### Data File Not Found

```
ERROR: Training data not found at: data/datasets/train.jsonl
```

**Solution**: Ensure training data exists at the specified path. Create it using:

```bash
uv run python scripts/08_create_dataset.py
```

### Data Schema Validation Failed

```
ERROR: Missing fields in data: ['instruction']. Required: ['instruction', 'input', 'output']
```

**Solution**: Ensure each JSONL line contains all required fields as JSON objects.

## Model Variants

The script auto-detects model variants:

```bash
# Nemotron 3 Nano (base)
uv run python train_nemotron3_nano.py  # Uses default NVIDIA/Nemotron-3-Nano-4B

# Nemotron 3 Nano Instruct
uv run python train_nemotron3_nano.py --model-path NVIDIA/Nemotron-3-Nano-4B-Instruct

# Nemotron 3.5 Lightning (not recommended, requires H100 GPUs)
uv run python train_nemotron3_nano.py --model-path NVIDIA/Nemotron-3.5-Lightning-30B
```

The `manifest.json` will show which variant was trained.

## Performance Notes

### Expected Training Time

On RTX 4070 (12GB):

- **Small dataset** (1-5k samples): 30-60 minutes for 3 epochs
- **Medium dataset** (10-50k samples): 2-6 hours for 3 epochs
- **Large dataset** (100k+ samples): 12+ hours for 3 epochs

On A100 (40GB):

- **Approximately 3-4x faster** than RTX 4070
- Can use larger batch sizes and sequence lengths

### Memory Usage

LoRA training with this script uses approximately:

- **Base model**: ~8GB (4B model, 16-bit precision)
- **Gradients + optimizer state**: ~2-3GB
- **Batch + cache**: ~1-2GB
- **Total**: ~12GB for RTX 4070 profile

Reducing batch size, sequence length, or using smaller hardware profile reduces memory usage.

## Experiment Tracking with W&B

Enable Weights & Biases for advanced experiment tracking:

```bash
# Authenticate (one-time)
wandb login

# Train with tracking
uv run python train_nemotron3_nano.py \
  --wandb-project rust-lora \
  --wandb-entity your-team
```

W&B tracks:

- Training/validation loss curves
- Learning rate schedules
- GPU memory usage
- Training duration
- Hyperparameters
- Model checkpoints
- System metrics

Useful for comparing multiple training runs and reproducibility.

## Next Steps: Export to Unsloth Studio & Ollama

### Unsloth Specific Optimizations for Nemotron 3 Nano

1. **NoPE Positional Architecture (1M Context):**
   Nemotron 3 Nano is trained with **NoPE (No Positional Embeddings)**. You can freely expand context from 16K up to 1,000,000 tokens using `--ctx-size` without needing RoPE scaling or YaRN.
2. **Special Reasoning Tokens:**
   Nemotron 3 uses `<think>` (token ID 12) and `</think>` (token ID 13). When deploying in llama.cpp, include `--special` (and `--verbose-prompt` if needed) to ensure the thinking tags render properly.
3. **Coding & Tool-Calling Presets:**
   - Coding: `temperature=0.6, top_p=0.95`
   - General chat: `temperature=1.0, top_p=1.0`
   - Stop tokens: `<|im_end|>`, `<extra_id_1>`, `<extra_id_2>`
4. **Quantization Footprint:**
   - 4-bit (`UD-Q4_K_XL`): ~3GB RAM/VRAM
   - 8-bit (`Q8_0`): ~5GB RAM/VRAM

### Export Commands

```bash
# Export merged model for Unsloth Studio
uv run python scripts/11_export_unsloth.py \
  --model models/nemotron3_nano_rust_lora \
  --base NVIDIA/Nemotron-3-Nano-4B \
  --mode merged-hf \
  --output-dir models/nemotron3_nano_merged

# Serve in llama-server with special reasoning tokens
llama-server \
  --model models/nemotron3_nano_merged/nemotron3_nano_Q8_0.gguf \
  --special \
  --ctx-size 32768 \
  --temp 0.6 --top-p 0.95 \
  --port 8001

# Or export to Ollama
uv run python scripts/09_create_modelfile.py \
  --model models/nemotron3_nano_rust_lora \
  --name rust-nemotron \
  --base nemotron

uv run python scripts/10_export_ollama.py \
  --model models/nemotron3_nano_rust_lora \
  --name rust-nemotron \
  --base NVIDIA/Nemotron-3-Nano-4B
```

For complete quantization analysis and Dynamic 3.0 benchmarks, see [QUANTIZIZE.md](QUANTIZIZE.md).

## References

- **Unsloth Documentation**: https://unsloth.ai/docs/models/nemotron-3
- **Nemotron 3 Nano Card**: https://huggingface.co/NVIDIA/Nemotron-3-Nano-4B
- **Nemotron 3.5 Lightning**: https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/lightning35/README.md
- **NVIDIA NeMo**: https://github.com/NVIDIA-NeMo/Nemotron

## Questions?

Refer to the in-script docstrings:

```bash
uv run python train_nemotron3_nano.py --help
```

Or check the source code comments in `train_nemotron3_nano.py` for detailed implementation notes.
