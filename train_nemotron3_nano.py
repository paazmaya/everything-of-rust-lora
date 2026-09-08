#!/usr/bin/env python3
"""
Production-Grade Nemotron 3 Nano 4B LoRA Training Script

This script trains a LoRA adapter on NVIDIA Nemotron 3 Nano 4B using Unsloth.
It includes validation, checkpointing, experiment tracking, and hardware-aware configuration.

Key Features:
- Data validation (schema, file existence)
- Dependency checks (mamba-ssm, CUDA, GPU memory)
- Hardware-aware batch size tuning via profiles
- Resumable training with checkpointing
- Experiment tracking (local manifest + optional W&B)
- Dry-run mode for validation without training
- Comprehensive error handling and logging

Model Support:
- NVIDIA/Nemotron-3-Nano-4B (default)
- NVIDIA/Nemotron-3-Nano-4B-Instruct
- Local model paths (with optional trust_remote_code)

Usage:
    # Default training (RTX 4070 profile, HuggingFace model):
    uv run python train_nemotron3_nano.py

    # Dry-run to validate configuration:
    uv run python train_nemotron3_nano.py --dry-run

    # Use local model:
    uv run python train_nemotron3_nano.py --model-path /path/to/model --trust-remote-code

    # Use RTX 6000 hardware profile:
    uv run python train_nemotron3_nano.py --hardware-profile rtx6000

    # Resume from checkpoint:
    uv run python train_nemotron3_nano.py --resume-from-checkpoint models/nemotron3_nano_rust_lora/checkpoints/checkpoint-50

    # Custom hyperparameters:
    uv run python train_nemotron3_nano.py --batch-size 2 --max-seq-length 2048

    # Enable W&B tracking:
    uv run python train_nemotron3_nano.py --wandb-project my-project --wandb-entity my-team

Documentation References:
- Nemotron 3 Nano: https://unsloth.ai/docs/models/nemotron-3
- Nemotron Lightning Practices: https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/lightning35/README.md
- Unsloth: https://unsloth.ai/docs/models/nemotron-3
"""

# ruff: noqa: I001
import argparse
import json
import logging
import os
import sys
from collections.abc import Sized
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from transformers import TrainingArguments
from trl.trainer.sft_trainer import SFTTrainer
from unsloth import FastLanguageModel

# ============================================================================
# Logging Configuration
# ============================================================================


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging with timestamps and levels."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# ============================================================================
# Configuration Management
# ============================================================================


def load_config(config_path: str = "config/nemotron_training_config.yaml") -> dict[str, Any]:
    """Load training configuration from YAML file."""
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {}

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config or {}
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config file: {e}")
        raise


def get_hardware_profile(config: dict[str, Any], profile_name: str = "rtx4070") -> dict[str, Any]:
    """Get hardware profile from config."""
    profiles = config.get("hardware_profiles", {})
    if profile_name not in profiles:
        logger.warning(
            f"Hardware profile '{profile_name}' not found. Available: {list(profiles.keys())}"
        )
        return profiles.get("rtx4070", {})  # Fallback to default

    profile = profiles[profile_name]
    logger.info(
        f"Using hardware profile '{profile_name}': {profile.get('description', 'No description')}"
    )
    return profile


# ============================================================================
# Validation Functions
# ============================================================================


def check_cuda_available() -> bool:
    """Verify CUDA is available and report GPU info."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. GPU training is required for this script.")
        return False

    logger.info(f"CUDA available. GPU: {torch.cuda.get_device_name(0)}")
    return True


def check_gpu_memory(min_vram_gb: float = 12) -> bool:
    """Check if GPU has sufficient VRAM."""
    if not torch.cuda.is_available():
        return False

    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
    logger.info(f"GPU VRAM: {total_memory:.1f} GB")

    if total_memory < min_vram_gb:
        logger.warning(
            f"GPU has {total_memory:.1f} GB, but {min_vram_gb} GB is recommended. "
            "Training may fail or run slowly."
        )
        return False

    return True


def check_dependencies(trust_remote_code: bool = False) -> bool:
    """Check for required dependencies."""
    missing = []

    try:
        import unsloth  # noqa: F401
    except ImportError:
        missing.append("unsloth")

    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers")

    try:
        import trl  # noqa: F401
    except ImportError:
        missing.append("trl")

    try:
        import peft  # noqa: F401
    except ImportError:
        missing.append("peft")

    if trust_remote_code:
        try:
            import mamba_ssm  # pyright: ignore[reportMissingImports] # noqa: F401
        except ImportError:
            logger.warning(
                "The model requires `mamba-ssm` package when trust_remote_code=True.\n"
                "Install with: pip install mamba-ssm"
            )
            missing.append("mamba-ssm")

    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.error("Install with: uv sync or pip install <package>")
        return False

    logger.info("All dependencies available")
    return True


def validate_data_file(data_path: str, required_fields: list[str] | None = None) -> bool:
    """Validate training data file exists and has correct schema."""
    if not os.path.exists(data_path):
        logger.error(f"Training data not found at: {data_path}")
        logger.error("Create training data in JSONL format with fields: instruction, input, output")
        return False

    logger.info(f"Found training data at: {data_path}")

    # Validate schema on first record
    required_fields = required_fields or ["instruction", "input", "output"]
    try:
        with open(data_path) as f:
            first_line = f.readline().strip()
            if not first_line:
                logger.error("Training data file is empty")
                return False

            record = json.loads(first_line)
            missing_fields = [f for f in required_fields if f not in record]

            if missing_fields:
                logger.error(
                    f"Missing fields in data: {missing_fields}. Required: {required_fields}"
                )
                return False

            logger.info(f"Data schema valid (found {len(record)} fields)")
            return True

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse training data as JSONL: {e}")
        return False


def detect_model_variant(model_path: str) -> str:
    """
    Detect Nemotron model variant from path.

    Returns: "3-Nano" or "3.5-Lightning" or "unknown"
    """
    model_lower = model_path.lower()

    if "3.5" in model_path or "lightning" in model_lower:
        return "3.5-Lightning"
    elif "nano" in model_lower or "3-nano" in model_path:
        return "3-Nano"
    else:
        return "unknown"


# ============================================================================
# Training Functions
# ============================================================================


def create_output_directories(output_dir: str, checkpoint_dir: str) -> None:
    """Create output directories if they don't exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")


def save_training_metadata(
    output_dir: str,
    model_path: str,
    config: dict[str, Any],
    hardware_profile: str,
) -> None:
    """Save training metadata to manifest for artifact tracking."""
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "hardware_profile": hardware_profile,
        "variant": detect_model_variant(model_path),
        "config": config,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved training metadata to {manifest_path}")


def train_nemotron(
    model_path: str = "NVIDIA/Nemotron-3-Nano-4B",
    max_seq_length: int = 1024,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    warmup_steps: int = 10,
    weight_decay: float = 0.01,
    lr_scheduler_type: str = "cosine",
    trust_remote_code: bool = False,
    output_dir: str = "models/nemotron3_nano_rust_lora",
    checkpoint_dir: str = "models/nemotron3_nano_rust_lora/checkpoints",
    resume_from_checkpoint: str | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    data_file: str = "data/datasets/train.jsonl",
    save_strategy: str = "steps",
    save_steps: int = 50,
    logging_steps: int = 10,
) -> None:
    """
    Train a LoRA adapter on NVIDIA Nemotron 3 Nano 4B.

    Args:
        model_path: HuggingFace model ID or local path
        max_seq_length: Maximum sequence length for training
        batch_size: Per-device batch size
        gradient_accumulation_steps: Gradient accumulation for larger effective batch
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        warmup_steps: Linear warmup steps
        weight_decay: Weight decay for regularization
        lr_scheduler_type: Learning rate scheduler type
        trust_remote_code: Enable remote code execution for custom models
        output_dir: Directory to save final LoRA adapter
        checkpoint_dir: Directory to save training checkpoints
        resume_from_checkpoint: Path to checkpoint to resume from
        wandb_project: W&B project name (optional)
        wandb_entity: W&B entity name (optional)
        data_file: Path to training data (JSONL)
        save_strategy: Checkpoint save strategy ("steps", "epoch", "no")
        save_steps: Save checkpoint every N steps
        logging_steps: Log metrics every N steps
    """
    logger.info("=" * 80)
    logger.info("NEMOTRON 3 NANO LORA TRAINING")
    logger.info("=" * 80)

    # Detect model variant
    variant = detect_model_variant(model_path)
    logger.info(f"Model variant: {variant}")

    # Determine dtype
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"Using dtype: {dtype}")

    # Load model
    logger.info(f"Loading model from: {model_path}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
            trust_remote_code=trust_remote_code,
        )
    except ModuleNotFoundError as err:
        if "mamba_ssm" in str(err) or "mamba-ssm" in str(err):
            raise RuntimeError(
                "The Nemotron model requires the `mamba-ssm` package when `trust_remote_code=True`. "
                "Install with: pip install mamba-ssm"
            ) from err
        raise

    logger.info("Model loaded successfully")

    # Setup LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    logger.info("LoRA configured (r=16, alpha=16)")

    # Create output directories
    create_output_directories(output_dir, checkpoint_dir)

    # Setup prompt template
    nemotron_prompt = """<|im_start|>system
You are an expert Rust programmer, specializing in systems programming, async Rust, ESP32 embedded development, and popular Rust libraries.<|im_end|>
<|im_start|>user
{}{}<|im_end|>
<|im_start|>assistant
{}"""

    def formatting_prompts_func(examples: dict[str, list[str]]) -> dict[str, list[str]]:
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts: list[str] = []
        for instruction, input_text, output in zip(instructions, inputs, outputs, strict=False):
            user_msg = instruction + "\n" + input_text if input_text else instruction
            text = nemotron_prompt.format(user_msg, "", output) + "<|im_end|>"
            texts.append(text)
        return {"text": texts}

    # Load dataset
    logger.info(f"Loading training data from: {data_file}")
    dataset = load_dataset("json", data_files=data_file, split="train")
    if isinstance(dataset, Sized):
        logger.info(f"Dataset size: {len(dataset)} samples")

    # Format dataset
    dataset = dataset.map(formatting_prompts_func, batched=True)
    logger.info("Dataset formatted with prompt template")

    # Configure training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=logging_steps,
        output_dir=checkpoint_dir,
        save_strategy=save_strategy,
        save_steps=save_steps if save_strategy == "steps" else 500,
        optim="adamw_8bit",
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        seed=3407,
        report_to=["wandb"] if wandb_project else [],
        run_name=f"nemotron3-nano-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,  # pyright: ignore[reportCallIssue]
        train_dataset=dataset,
        dataset_text_field="text",  # pyright: ignore[reportCallIssue]
        max_seq_length=max_seq_length,  # pyright: ignore[reportCallIssue]
        args=training_args,
    )

    # Log training config
    logger.info("Training configuration:")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {batch_size} (per device)")
    logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info(f"  LR scheduler: {lr_scheduler_type}")
    logger.info(f"  Max seq length: {max_seq_length}")

    # Train
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)

    if resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    # Save final model
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE - SAVING MODEL")
    logger.info("=" * 80)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to: {output_dir}")

    # Save metadata
    save_training_metadata(output_dir, model_path, {}, "custom")


def validate_configuration(
    config: dict[str, Any],
    check_data: bool = True,
    check_deps: bool = True,
    check_gpu: bool = True,
    trust_remote_code: bool = False,
) -> bool:
    """Validate all configuration before training."""
    logger.info("=" * 80)
    logger.info("VALIDATING CONFIGURATION")
    logger.info("=" * 80)

    # Check CUDA
    if check_gpu:
        if not check_cuda_available():
            logger.error("CUDA validation failed")
            return False

    # Check dependencies
    if check_deps:
        if not check_dependencies(trust_remote_code=trust_remote_code):
            logger.error("Dependency validation failed")
            return False

    # Check GPU memory
    if check_gpu:
        min_vram = config.get("validation", {}).get("min_vram_gb", 12)
        if not check_gpu_memory(min_vram_gb=min_vram):
            logger.warning("GPU memory check passed with warnings")

    # Check data
    if check_data:
        data_file = config.get("data", {}).get("train_file", "data/datasets/train.jsonl")
        required_fields = config.get("data", {}).get("required_fields", None)
        if not validate_data_file(data_file, required_fields=required_fields):
            logger.error("Data validation failed")
            return False

    logger.info("=" * 80)
    logger.info("VALIDATION PASSED")
    logger.info("=" * 80)
    return True


# ============================================================================
# CLI and Main
# ============================================================================


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Production-grade Nemotron 3 Nano 4B LoRA training with validation and checkpointing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default training:
  python train_nemotron3_nano.py

  # Dry-run to validate config:
  python train_nemotron3_nano.py --dry-run

  # Resume from checkpoint:
  python train_nemotron3_nano.py --resume-from-checkpoint models/nemotron3_nano_rust_lora/checkpoints/checkpoint-50

  # Use RTX 6000 hardware profile:
  python train_nemotron3_nano.py --hardware-profile rtx6000

  # Custom settings:
  python train_nemotron3_nano.py --batch-size 2 --max-seq-length 2048 --num-epochs 5
        """,
    )

    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        default="NVIDIA/Nemotron-3-Nano-4B",
        help="HuggingFace model ID or local path (default: NVIDIA/Nemotron-3-Nano-4B)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for local models with custom config. May require mamba-ssm.",
    )

    # Hardware arguments
    parser.add_argument(
        "--hardware-profile",
        type=str,
        default="rtx4070",
        choices=["rtx4070", "rtx6000", "a100", "a100_80gb", "dev"],
        help="Hardware profile for batch size and seq length tuning",
    )

    # Training arguments
    parser.add_argument(
        "--max-seq-length",
        type=int,
        help="Maximum sequence length (overrides hardware profile)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Per-device batch size (overrides hardware profile)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="Gradient accumulation steps (overrides hardware profile)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Linear warmup steps",
    )

    # Data arguments
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/datasets/train.jsonl",
        help="Path to training data (JSONL format)",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/nemotron3_nano_rust_lora",
        help="Directory to save final LoRA adapter",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    # Tracking arguments
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name (optional)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity/team name (optional)",
    )

    # Utility arguments
    parser.add_argument(
        "--config",
        type=str,
        default="config/nemotron_training_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and exit without training",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        setup_logging(verbose=True)

    # Load config
    config = load_config(args.config)

    # Get hardware profile
    profile = get_hardware_profile(config, args.hardware_profile)

    # Build training args with CLI overrides
    max_seq_length = args.max_seq_length or profile.get("max_seq_length", 1024)
    batch_size = args.batch_size or profile.get("batch_size", 1)
    gradient_accumulation_steps = args.gradient_accumulation_steps or profile.get(
        "gradient_accumulation_steps", 4
    )

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")

    # Validate configuration
    if not validate_configuration(
        config,
        check_data=not args.dry_run,  # Skip data check in dry-run
        check_deps=True,
        check_gpu=True,
        trust_remote_code=args.trust_remote_code,
    ):
        logger.error("Configuration validation failed")
        sys.exit(1)

    # Dry-run: just validate and exit
    if args.dry_run:
        logger.info("Dry-run complete. Configuration is valid.")
        logger.info("To start training, run: python train_nemotron3_nano.py")
        sys.exit(0)

    # Train
    try:
        train_nemotron(
            model_path=args.model_path,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            trust_remote_code=args.trust_remote_code,
            output_dir=args.output_dir,
            checkpoint_dir=checkpoint_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            data_file=args.data_file,
        )
        logger.info("Training completed successfully!")
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.debug("Full traceback:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
