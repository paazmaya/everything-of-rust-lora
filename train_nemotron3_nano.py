#!/usr/bin/env python3
"""
Production-Grade NVIDIA Nemotron 3 Nano 4B LoRA Training Script

Trains a LoRA adapter on NVIDIA Nemotron 3 Nano 4B for Rust systems programming.
Features inline configuration, hardware-aware profiles, and preflight dry-run validation.

Usage:
    # Dry-run validation:
    uv run python train_nemotron3_nano.py --dry-run

    # Train using default RTX 4070 profile:
    uv run python train_nemotron3_nano.py

    # Train on workstation with RTX 6000:
    uv run python train_nemotron3_nano.py --hardware-profile rtx6000

    # Train with local model and custom batch size:
    uv run python train_nemotron3_nano.py --model-path /path/to/nemotron --batch-size 2
"""

import sys
from typing import Any

from lib import (
    DEFAULT_TRAIN_FILE,
    TrainingConfig,
    apply_args_to_config,
    build_arg_parser,
    load_and_prepare_dataset,
    load_model_and_tokenizer,
    run_preflight_checks,
    run_sft_training,
    setup_logging,
)

# ============================================================================
# Inline Configuration & Hardware Profiles
# ============================================================================

DEFAULT_MODEL_NAME = "NVIDIA/Nemotron-3-Nano-4B"
DEFAULT_OUTPUT_DIR = "models/nemotron3_nano_rust_lora"
DEFAULT_HARDWARE_PROFILE = "rtx4070"

# Predefined hardware profiles (migrated inline from nemotron_training_config.yaml)
HARDWARE_PROFILES: dict[str, dict[str, Any]] = {
    "rtx4070": {
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 1024,
        "load_in_4bit": False,
        "description": "NVIDIA RTX 4070 (12GB VRAM) - recommended default",
    },
    "rtx6000": {
        "batch_size": 2,
        "gradient_accumulation_steps": 2,
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "description": "NVIDIA RTX 6000 (24GB VRAM) - larger batches",
    },
    "a100": {
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "max_seq_length": 4096,
        "load_in_4bit": False,
        "description": "NVIDIA A100 (40GB VRAM) - high-end",
    },
    "a100_80gb": {
        "batch_size": 8,
        "gradient_accumulation_steps": 1,
        "max_seq_length": 4096,
        "load_in_4bit": False,
        "description": "NVIDIA A100 80GB - flagship",
    },
    "dev": {
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_seq_length": 512,
        "load_in_4bit": True,
        "description": "Development profile - minimal VRAM, fast iteration",
    },
}

NEMOTRON_PROMPT = """<|im_start|>system
You are an expert Rust programmer, specializing in systems programming, async Rust, ESP32 embedded development, and popular Rust libraries.<|im_end|>
<|im_start|>user
{}{}<|im_end|>
<|im_start|>assistant
{}"""


def detect_model_variant(model_path: str) -> str:
    """Detect Nemotron model variant from path or HuggingFace ID."""
    model_lower = model_path.lower()
    if "3.5" in model_path or "lightning" in model_lower:
        return "3.5-Lightning"
    elif "nano" in model_lower or "3-nano" in model_path:
        return "3-Nano"
    return "unknown"


def format_nemotron_prompts(examples: dict[str, list[str]]) -> dict[str, list[str]]:
    """Format instruction examples into NeMo ChatML format."""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts: list[str] = []
    for instruction, input_text, output in zip(instructions, inputs, outputs, strict=False):
        user_msg = f"{instruction}\n{input_text}" if input_text else instruction
        text = NEMOTRON_PROMPT.format(user_msg, "", output) + "<|im_end|>"
        texts.append(text)
    return {"text": texts}


def train_nemotron(config: TrainingConfig) -> None:
    """Execute LoRA training workflow for Nemotron."""
    logger = setup_logging(name="rust_lora.nemotron")
    config.model_family = "Nemotron"
    config.variant = detect_model_variant(config.model_path)

    # 1. Preflight validation
    if not run_preflight_checks(config, min_vram_gb=12.0, logger=logger):
        logger.warning("Preflight checks reported issues; proceeding with caution.")

    # 2. Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config, logger=logger)

    # 3. Load and format dataset
    dataset = load_and_prepare_dataset(
        data_file=config.data_file,
        prompt_formatter_fn=format_nemotron_prompts,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        logger=logger,
    )

    # 4. Train and save model
    run_sft_training(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=config,
        logger=logger,
    )


def main() -> None:
    """CLI entry point for Nemotron training."""
    parser = build_arg_parser(
        description="Train LoRA adapter on NVIDIA Nemotron 3 Nano 4B for Rust",
        default_model=DEFAULT_MODEL_NAME,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        hardware_profiles=HARDWARE_PROFILES,
        default_profile=DEFAULT_HARDWARE_PROFILE,
        default_max_seq_length=1024,
        default_batch_size=1,
        default_grad_accum=4,
    )
    args = parser.parse_args()
    logger = setup_logging(name="rust_lora.nemotron")

    config = TrainingConfig(
        model_path=args.model_path or DEFAULT_MODEL_NAME,
        model_family="Nemotron",
        variant=detect_model_variant(args.model_path or DEFAULT_MODEL_NAME),
        output_dir=args.output_dir or DEFAULT_OUTPUT_DIR,
        data_file=args.data_file or DEFAULT_TRAIN_FILE,
        trust_remote_code=args.trust_remote_code,
    )
    config = apply_args_to_config(config, args, hardware_profiles=HARDWARE_PROFILES)

    if args.dry_run:
        logger.info("Executing dry-run validation for Nemotron...")
        passed = run_preflight_checks(config, min_vram_gb=12.0, logger=logger)
        sys.exit(0 if passed else 1)

    try:
        train_nemotron(config)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Training failed with unexpected exception")
        sys.exit(1)


if __name__ == "__main__":
    main()
