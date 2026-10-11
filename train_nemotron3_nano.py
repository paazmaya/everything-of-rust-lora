#!/usr/bin/env python3
"""
Production-Grade NVIDIA Nemotron 3 Nano 4B LoRA Training Script

Trains a LoRA adapter on NVIDIA Nemotron 3 Nano 4B for Rust systems programming.
Features inline configuration, hardware-aware profiles, checkpoint resumption, and preflight dry-run validation.

Usage:
    # Dry-run validation:
    uv run python train_nemotron3_nano.py --dry-run

    # Train using default RTX 4070 profile:
    uv run python train_nemotron3_nano.py

    # Resume interrupted training from the latest checkpoint:
    uv run python train_nemotron3_nano.py --resume

    # Resume from a specific checkpoint:
    uv run python train_nemotron3_nano.py --resume-from-checkpoint models/nemotron3_nano_rust_lora/checkpoints/checkpoint-50

    # Train on workstation with RTX 6000:
    uv run python train_nemotron3_nano.py --hardware-profile rtx6000

    # Train with local model and custom batch size:
    uv run python train_nemotron3_nano.py --model-path /path/to/nemotron --batch-size 2
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
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA RTX 4070 (12GB VRAM) - recommended default",
    },
    "rtx6000": {
        "batch_size": 2,
        "gradient_accumulation_steps": 2,
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA RTX 6000 (24GB VRAM) - larger batches",
    },
    "a100": {
        "batch_size": 4,
        "gradient_accumulation_steps": 1,
        "max_seq_length": 4096,
        "load_in_4bit": False,
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA A100 (40GB VRAM) - high-end",
    },
    "a100_80gb": {
        "batch_size": 8,
        "gradient_accumulation_steps": 1,
        "max_seq_length": 4096,
        "load_in_4bit": False,
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA A100 80GB - flagship",
    },
    "dev": {
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_seq_length": 512,
        "load_in_4bit": True,
        "save_steps": 5,
        "save_total_limit": 2,
        "description": "Development profile - minimal VRAM, fast iteration",
    },
}


def detect_model_variant(model_path: str) -> str:
    """Detect Nemotron model variant from path or HuggingFace ID."""
    model_lower = model_path.lower()
    if "3.5" in model_path or "lightning" in model_lower:
        return "3.5-Lightning"
    elif "nano" in model_lower or "3-nano" in model_path:
        return "3-Nano"
    return "unknown"


def train_nemotron(config: TrainingConfig) -> None:
    """Execute LoRA training workflow for Nemotron with resumability."""
    logger = setup_logging(name="rust_lora.nemotron")
    config.model_family = "Nemotron"
    config.variant = detect_model_variant(config.model_path)

    execute_training_pipeline(
        config=config,
        prompt_formatter_fn=format_chatml_prompts,
        min_vram_gb=12.0,
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
        epilog=build_resume_epilog("train_nemotron3_nano.py"),
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

    run_training_cli(
        config=config,
        args=args,
        train_fn=train_nemotron,
        min_vram_gb=12.0,
        logger=logger,
    )


if __name__ == "__main__":
    main()
