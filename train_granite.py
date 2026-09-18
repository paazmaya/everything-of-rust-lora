#!/usr/bin/env python3
"""
Production-Grade IBM Granite 4.2 LoRA Training Script

Trains a LoRA adapter on IBM Granite 4.2 (8B or 3B variant) for Rust systems programming.
Features inline configuration, automatic variant detection, and preflight dry-run validation.

Usage:
    # Dry-run validation:
    uv run python train_granite.py --dry-run

    # Train using default Granite 4.2 8B:
    uv run python train_granite.py

    # Train with Granite 4.2 3B variant:
    uv run python train_granite.py --model-path ibm-granite/granite-4.2-3b

    # Train with local model directory:
    uv run python train_granite.py --model-path /path/to/granite-4.2-8b
"""

import re
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

DEFAULT_MODEL_NAME = "ibm-granite/granite-4.2-8b"
DEFAULT_HARDWARE_PROFILE = "rtx3090_4090"

HARDWARE_PROFILES: dict[str, dict[str, Any]] = {
    "rtx4070": {
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 2048,
        "load_in_4bit": True,
        "description": "NVIDIA RTX 4070 (12GB VRAM) - 4-bit QLoRA",
    },
    "rtx3090_4090": {
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 4096,
        "load_in_4bit": True,
        "description": "NVIDIA RTX 3090/4090 (24GB VRAM) - 4-bit QLoRA default",
    },
    "rtx6000": {
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 4096,
        "load_in_4bit": False,
        "description": "NVIDIA RTX 6000 Ada (48GB VRAM) - 16-bit bf16 LoRA",
    },
    "a100": {
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 4096,
        "load_in_4bit": False,
        "description": "NVIDIA A100 40GB/80GB - 16-bit bf16 LoRA",
    },
    "dev": {
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_seq_length": 512,
        "load_in_4bit": True,
        "description": "Development profile - fast iteration and minimal VRAM",
    },
}

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def detect_model_variant(model_path: str) -> str:
    """Detect the model variant (8B or 3B) from the model ID or path."""
    if re.search(r"3b", model_path, re.IGNORECASE):
        return "3B"
    elif re.search(r"8b", model_path, re.IGNORECASE):
        return "8B"
    return "8B"


def format_granite_prompts(examples: dict[str, list[str]]) -> dict[str, list[str]]:
    """Format instruction examples into Alpaca format for Granite."""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts: list[str] = []
    for instruction, input_text, output in zip(instructions, inputs, outputs, strict=False):
        text = ALPACA_PROMPT.format(instruction, input_text, output)
        texts.append(text)
    return {"text": texts}


def train_granite(config: TrainingConfig) -> None:
    """Execute LoRA training workflow for IBM Granite 4.2."""
    logger = setup_logging(name="rust_lora.granite")
    variant = detect_model_variant(config.model_path)
    config.model_family = "Granite"
    config.variant = variant

    # 1. Preflight validation
    if not run_preflight_checks(config, min_vram_gb=12.0, logger=logger):
        logger.warning("Preflight checks reported issues; proceeding with caution.")

    # 2. Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config, logger=logger)

    # 3. Load and format dataset
    dataset = load_and_prepare_dataset(
        data_file=config.data_file,
        prompt_formatter_fn=format_granite_prompts,
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
    """CLI entry point for Granite training."""
    parser = build_arg_parser(
        description="Train LoRA adapter on IBM Granite 4.2 (8B or 3B) for Rust",
        default_model=DEFAULT_MODEL_NAME,
        default_output_dir="models/granite_4.2_8b_rust_lora",
        hardware_profiles=HARDWARE_PROFILES,
        default_profile=DEFAULT_HARDWARE_PROFILE,
        default_max_seq_length=4096,
        default_batch_size=1,
        default_grad_accum=8,
        default_load_in_4bit=True,
    )
    args = parser.parse_args()
    logger = setup_logging(name="rust_lora.granite")

    variant = detect_model_variant(args.model_path or DEFAULT_MODEL_NAME)
    default_dir = f"models/granite_4.2_{variant.lower()}_rust_lora"
    output_dir = (
        args.output_dir if args.output_dir != "models/granite_4.2_8b_rust_lora" else default_dir
    )

    config = TrainingConfig(
        model_path=args.model_path or DEFAULT_MODEL_NAME,
        model_family="Granite",
        variant=variant,
        max_seq_length=4096,
        batch_size=1,
        gradient_accumulation_steps=8,
        load_in_4bit=True,
        load_in_16bit=False,
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        output_dir=output_dir,
        data_file=args.data_file or DEFAULT_TRAIN_FILE,
        trust_remote_code=args.trust_remote_code,
    )
    config = apply_args_to_config(config, args, hardware_profiles=HARDWARE_PROFILES)

    if args.dry_run:
        logger.info(f"Executing dry-run validation for Granite 4.2 {variant}...")
        passed = run_preflight_checks(config, min_vram_gb=12.0, logger=logger)
        sys.exit(0 if passed else 1)

    try:
        train_granite(config)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Training failed with unexpected exception")
        sys.exit(1)


if __name__ == "__main__":
    main()
