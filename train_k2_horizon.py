#!/usr/bin/env python3
"""
Production-Grade IFM K2-Horizon-3.7B LoRA Training Script

Trains a LoRA adapter on IFM/K2-Horizon-3.7B (dense 3.7B, 512K context) for Rust development.
Features inline configuration, hardware-aware profiles, checkpoint resumption, and preflight dry-run validation.

Usage:
    # Dry-run validation:
    uv run python train_k2_horizon.py --dry-run

    # Train using default RTX 3090/4090 profile:
    uv run python train_k2_horizon.py

    # Resume interrupted training from the latest checkpoint:
    uv run python train_k2_horizon.py --resume

    # Resume from a specific checkpoint:
    uv run python train_k2_horizon.py --resume-from-checkpoint models/k2_horizon_rust_lora/checkpoints/checkpoint-50

    # Train on 12GB GPU with 4-bit QLoRA:
    uv run python train_k2_horizon.py --hardware-profile rtx4070

    # Train with local model directory:
    uv run python train_k2_horizon.py --model-path /path/to/K2-Horizon-3.7B
"""

from typing import Any

from lib import (
    DEFAULT_RUST_SYSTEM_PROMPT,
    DEFAULT_TRAIN_FILE,
    TrainingConfig,
    apply_args_to_config,
    build_arg_parser,
    build_resume_epilog,
    execute_training_pipeline,
    run_training_cli,
    setup_logging,
)

# ============================================================================
# Inline Configuration & Hardware Profiles
# ============================================================================

DEFAULT_MODEL_NAME = "IFM/K2-Horizon-3.7B"
DEFAULT_OUTPUT_DIR = "models/k2_horizon_rust_lora"
DEFAULT_HARDWARE_PROFILE = "rtx3090_4090"
TRUST_REMOTE_CODE = True

HARDWARE_PROFILES: dict[str, dict[str, Any]] = {
    "rtx4070": {
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 1024,
        "load_in_4bit": True,
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA RTX 4070 (12GB VRAM) - 4-bit QLoRA default",
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
        "batch_size": 2,
        "gradient_accumulation_steps": 2,
        "max_seq_length": 4096,
        "load_in_4bit": False,
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA A100 40GB - data center",
    },
    "a100_80gb": {
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "max_seq_length": 8192,
        "load_in_4bit": False,
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA A100/H100 80GB - long context training",
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

# Chat template delimiters
IM_START = "<|ifm|im_start|>"
IM_END = "<|ifm|im_end|>"

K2_PROMPT_TEMPLATE = (
    f"{IM_START}system\n"
    f"{DEFAULT_RUST_SYSTEM_PROMPT}{IM_END}\n"
    f"{IM_START}user\n"
    "{user_message}"
    f"{IM_END}\n"
    f"{IM_START}assistant\n"
    "<ifm|think>\n"
    f"</ifm|think>{{output_message}}{IM_END}"
)


def format_k2_prompts(examples: dict[str, list[str]]) -> dict[str, list[str]]:
    """Format instruction examples into K2-Horizon ChatML format."""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts: list[str] = []
    for instruction, input_text, output in zip(instructions, inputs, outputs, strict=False):
        user_msg = f"{instruction}\n{input_text}" if input_text else instruction
        text = K2_PROMPT_TEMPLATE.format(
            user_message=user_msg,
            output_message=output,
        )
        texts.append(text)
    return {"text": texts}


def train_k2_horizon(config: TrainingConfig) -> None:
    """Execute LoRA training workflow for IFM K2-Horizon with resumability."""
    logger = setup_logging(name="rust_lora.k2_horizon")
    config.model_family = "K2-Horizon"
    config.trust_remote_code = True

    execute_training_pipeline(
        config=config,
        prompt_formatter_fn=format_k2_prompts,
        min_vram_gb=12.0,
        logger=logger,
    )


def main() -> None:
    """CLI entry point for K2-Horizon training."""
    parser = build_arg_parser(
        description="Train LoRA adapter on IFM K2-Horizon-3.7B for Rust development",
        default_model=DEFAULT_MODEL_NAME,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        hardware_profiles=HARDWARE_PROFILES,
        default_profile=DEFAULT_HARDWARE_PROFILE,
        default_max_seq_length=2048,
        default_batch_size=1,
        default_grad_accum=4,
        epilog=build_resume_epilog("train_k2_horizon.py"),
    )
    args = parser.parse_args()
    logger = setup_logging(name="rust_lora.k2_horizon")

    config = TrainingConfig(
        model_path=args.model_path or DEFAULT_MODEL_NAME,
        model_family="K2-Horizon",
        output_dir=args.output_dir or DEFAULT_OUTPUT_DIR,
        data_file=args.data_file or DEFAULT_TRAIN_FILE,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    config = apply_args_to_config(config, args, hardware_profiles=HARDWARE_PROFILES)

    run_training_cli(
        config=config,
        args=args,
        train_fn=train_k2_horizon,
        min_vram_gb=12.0,
        logger=logger,
    )


if __name__ == "__main__":
    main()
