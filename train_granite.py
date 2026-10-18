#!/usr/bin/env python3
"""
Production-Grade IBM Granite 4.2 LoRA Training Script

Trains a LoRA adapter on IBM Granite 4.2 (8B or 3B variant) for Rust systems programming.
Features inline configuration, automatic variant detection, checkpoint resumption, and preflight dry-run validation.

Usage:
    # Dry-run validation:
    uv run python train_granite.py --dry-run

    # Train using default Granite 4.2 8B:
    uv run python train_granite.py

    # Resume interrupted training from the latest checkpoint:
    uv run python train_granite.py --resume

    # Resume from a specific checkpoint:
    uv run python train_granite.py --resume-from-checkpoint models/granite_4.2_8b_rust_lora/checkpoints/checkpoint-50

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
    build_resume_epilog,
    execute_training_pipeline,
    format_alpaca_prompts,
    run_training_cli,
    setup_logging,
)

# ============================================================================
# Inline Configuration & Hardware Profiles
# ============================================================================

DEFAULT_MODEL_NAME = "ibm-granite/granite-4.2-8b"
DEFAULT_OUTPUT_DIR = "models/granite_4.2_8b_rust_lora"
# Windows workstation has a 12GB RTX 4070 Ti
DEFAULT_HARDWARE_PROFILE = "rtx4070" if sys.platform == "win32" else "rtx3090_4090"

HARDWARE_PROFILES: dict[str, dict[str, Any]] = {
    "rtx4070": {
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 2048,
        "load_in_4bit": True,
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA RTX 4070 (12GB VRAM) - 4-bit QLoRA",
    },
    "rtx3090_4090": {
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 4096,
        "load_in_4bit": True,
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA RTX 3090/4090 (24GB VRAM) - 4-bit QLoRA default",
    },
    "rtx6000": {
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 4096,
        "load_in_4bit": False,
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA RTX 6000 Ada (48GB VRAM) - 16-bit bf16 LoRA",
    },
    "a100": {
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 4096,
        "load_in_4bit": False,
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA A100 40GB/80GB - 16-bit bf16 LoRA",
    },
    "dev": {
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_seq_length": 512,
        "load_in_4bit": True,
        "save_steps": 5,
        "save_total_limit": 2,
        "description": "Development profile - fast iteration and minimal VRAM",
    },
}


def detect_model_variant(model_path: str) -> str:
    """Detect the model variant (8B or 3B) from the model ID or path."""
    if re.search(r"3b", model_path, re.IGNORECASE):
        return "3B"
    elif re.search(r"8b", model_path, re.IGNORECASE):
        return "8B"
    return "8B"


def train_granite(config: TrainingConfig) -> None:
    """Execute LoRA training workflow for IBM Granite 4.2 with resumability."""
    logger = setup_logging(name="rust_lora.granite")
    variant = detect_model_variant(config.model_path)
    config.model_family = "Granite"
    config.variant = variant

    execute_training_pipeline(
        config=config,
        prompt_formatter_fn=format_alpaca_prompts,
        min_vram_gb=12.0,
        logger=logger,
    )


def main() -> None:
    """CLI entry point for Granite training."""
    parser = build_arg_parser(
        description="Train LoRA adapter on IBM Granite 4.2 (8B or 3B) for Rust",
        default_model=DEFAULT_MODEL_NAME,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        hardware_profiles=HARDWARE_PROFILES,
        default_profile=DEFAULT_HARDWARE_PROFILE,
        default_max_seq_length=4096,
        default_batch_size=1,
        default_grad_accum=8,
        default_load_in_4bit=True,
        epilog=build_resume_epilog(
            "train_granite.py",
            """
  # Train with Granite 4.2 3B variant:
  uv run python train_granite.py --model-path ibm-granite/granite-4.2-3b
            """,
        ),
    )
    args = parser.parse_args()
    logger = setup_logging(name="rust_lora.granite")

    variant = detect_model_variant(args.model_path or DEFAULT_MODEL_NAME)
    default_dir = f"models/granite_4.2_{variant.lower()}_rust_lora"
    output_dir = args.output_dir if args.output_dir != DEFAULT_OUTPUT_DIR else default_dir

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

    run_training_cli(
        config=config,
        args=args,
        train_fn=train_granite,
        min_vram_gb=12.0,
        logger=logger,
    )


if __name__ == "__main__":
    main()
