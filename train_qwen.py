#!/usr/bin/env python3
"""
Production-Grade Qwen 3.5 4B LoRA Training Script

Trains a LoRA adapter on Qwen/Qwen3.5-4B for Rust systems and embedded development.
Features inline configuration, hardware-aware profiles, and preflight dry-run validation.

Note: Qwen 3.5 4B is best fine-tuned in 16-bit bf16 precision; 4-bit QLoRA is not recommended.

Usage:
    # Dry-run validation:
    uv run python train_qwen.py --dry-run

    # Train using default RTX 4070 profile:
    uv run python train_qwen.py

    # Train on workstation with RTX 6000:
    uv run python train_qwen.py --hardware-profile rtx6000

    # Train with local model directory:
    uv run python train_qwen.py --model-path /path/to/Qwen3.5-4B
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

DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-4B"
DEFAULT_OUTPUT_DIR = "models/qwen3_5_4b_rust_lora"
DEFAULT_HARDWARE_PROFILE = "rtx4070"

HARDWARE_PROFILES: dict[str, dict[str, Any]] = {
    "rtx4070": {
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 1024,
        "load_in_4bit": False,
        "description": "NVIDIA RTX 4070 (12GB VRAM) - 16-bit bf16 LoRA (recommended default)",
    },
    "rtx3090_4090": {
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "description": "NVIDIA RTX 3090/4090 (24GB VRAM) - 16-bit bf16 LoRA",
    },
    "rtx6000": {
        "batch_size": 2,
        "gradient_accumulation_steps": 2,
        "max_seq_length": 4096,
        "load_in_4bit": False,
        "description": "NVIDIA RTX 6000 Ada (48GB VRAM) - larger context and batches",
    },
    "a100": {
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "max_seq_length": 4096,
        "load_in_4bit": False,
        "description": "NVIDIA A100 (40GB VRAM) - high-throughput",
    },
    "dev": {
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_seq_length": 512,
        "load_in_4bit": True,
        "description": "Development profile - fast iteration and minimal VRAM",
    },
}

QWEN_PROMPT = """<|im_start|>system
You are an expert Rust programmer, specializing in systems programming, async Rust, ESP32 embedded development, and popular Rust libraries.<|im_end|>
<|im_start|>user
{}{}<|im_end|>
<|im_start|>assistant
{}"""


def format_qwen_prompts(examples: dict[str, list[str]]) -> dict[str, list[str]]:
    """Format instruction examples into ChatML format for Qwen."""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts: list[str] = []
    for instruction, input_text, output in zip(instructions, inputs, outputs, strict=False):
        user_msg = f"{instruction}\n{input_text}" if input_text else instruction
        text = QWEN_PROMPT.format(user_msg, "", output) + "<|im_end|>"
        texts.append(text)
    return {"text": texts}


def train_qwen(config: TrainingConfig) -> None:
    """Execute LoRA training workflow for Qwen 3.5."""
    logger = setup_logging(name="rust_lora.qwen")
    config.model_family = "Qwen"

    # 1. Preflight validation
    if not run_preflight_checks(config, min_vram_gb=12.0, logger=logger):
        logger.warning("Preflight checks reported issues; proceeding with caution.")

    # 2. Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config, logger=logger)

    # 3. Load and format dataset
    dataset = load_and_prepare_dataset(
        data_file=config.data_file,
        prompt_formatter_fn=format_qwen_prompts,
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
    """CLI entry point for Qwen training."""
    parser = build_arg_parser(
        description="Train LoRA adapter on Qwen 3.5 4B for Rust",
        default_model=DEFAULT_MODEL_NAME,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        hardware_profiles=HARDWARE_PROFILES,
        default_profile=DEFAULT_HARDWARE_PROFILE,
        default_max_seq_length=1024,
        default_batch_size=1,
        default_grad_accum=4,
        default_load_in_4bit=False,
    )
    args = parser.parse_args()
    logger = setup_logging(name="rust_lora.qwen")

    config = TrainingConfig(
        model_path=args.model_path or DEFAULT_MODEL_NAME,
        model_family="Qwen",
        max_seq_length=1024,
        batch_size=1,
        gradient_accumulation_steps=4,
        load_in_4bit=False,
        load_in_16bit=True,
        output_dir=args.output_dir or DEFAULT_OUTPUT_DIR,
        data_file=args.data_file or DEFAULT_TRAIN_FILE,
        trust_remote_code=args.trust_remote_code,
    )
    config = apply_args_to_config(config, args, hardware_profiles=HARDWARE_PROFILES)

    if args.dry_run:
        logger.info("Executing dry-run validation for Qwen 3.5...")
        passed = run_preflight_checks(config, min_vram_gb=12.0, logger=logger)
        sys.exit(0 if passed else 1)

    try:
        train_qwen(config)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Training failed with unexpected exception")
        sys.exit(1)


if __name__ == "__main__":
    main()
