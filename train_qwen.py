#!/usr/bin/env python3
"""
Production-Grade Qwen 3.5 4B LoRA Training Script

Trains a LoRA adapter on Qwen/Qwen3.5-4B for Rust systems and embedded development.
Features inline configuration, hardware-aware profiles, checkpoint resumption, and preflight dry-run validation.

Note: Qwen 3.5 4B is best fine-tuned in 16-bit bf16 precision; 4-bit QLoRA is not recommended.

Usage:
    # Dry-run validation:
    uv run python train_qwen.py --dry-run

    # Train using default RTX 4070 profile:
    uv run python train_qwen.py

    # Resume interrupted training from the latest checkpoint:
    uv run python train_qwen.py --resume

    # Resume from a specific checkpoint:
    uv run python train_qwen.py --resume-from-checkpoint models/qwen3_5_4b_rust_lora/checkpoints/checkpoint-50

    # Train on workstation with RTX 6000:
    uv run python train_qwen.py --hardware-profile rtx6000

    # Train with local model directory:
    uv run python train_qwen.py --model-path /path/to/Qwen3.5-4B
"""

from typing import Any

from lib import (
    DEFAULT_TRAIN_FILE,
    TrainingConfig,
    apply_args_to_config,
    build_arg_parser,
    build_resume_epilog,
    execute_training_pipeline,
    format_chatml_prompts,
    run_training_cli,
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
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA RTX 4070 (12GB VRAM) - 16-bit bf16 LoRA (recommended default)",
    },
    "rtx3090_4090": {
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA RTX 3090/4090 (24GB VRAM) - 16-bit bf16 LoRA",
    },
    "rtx6000": {
        "batch_size": 2,
        "gradient_accumulation_steps": 2,
        "max_seq_length": 4096,
        "load_in_4bit": False,
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA RTX 6000 Ada (48GB VRAM) - larger context and batches",
    },
    "a100": {
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "max_seq_length": 4096,
        "load_in_4bit": False,
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA A100 (40GB VRAM) - high-throughput",
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


def train_qwen(config: TrainingConfig) -> None:
    """Execute LoRA training workflow for Qwen 3.5 with resumability."""
    logger = setup_logging(name="rust_lora.qwen")
    config.model_family = "Qwen"

    execute_training_pipeline(
        config=config,
        prompt_formatter_fn=format_chatml_prompts,
        min_vram_gb=12.0,
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
        epilog=build_resume_epilog("train_qwen.py"),
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

    run_training_cli(
        config=config,
        args=args,
        train_fn=train_qwen,
        min_vram_gb=12.0,
        logger=logger,
    )


if __name__ == "__main__":
    main()
