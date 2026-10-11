#!/usr/bin/env python3
"""
Production-Grade Qwen 3.8 27B LoRA Training Script

Trains a LoRA adapter specifically designed for the Qwen 3.8 27B architecture
(64 layers, hidden size 5120, hybrid linear + full attention).

The resulting LoRA adapter targets all linear-attention projections,
full-attention projections, and feed-forward MLP projections, ensuring
complete compatibility with any base model sharing the Qwen 3.8 27B architecture.

Features:
    - Default hardware profiles tailored for macOS Apple Silicon (MPS + offload)
    - Full checkpointing and automatic resumption (--resume)
    - Hybrid attention LoRA targeting (linear_attn + self_attn + mlp)
    - Preflight dry-run validation (--dry-run)

Usage:
    # Dry-run validation on Mac:
    uv run python train_qwen3_8.py --dry-run

    # Train using default Mac profile (mac_48gb on macOS):
    uv run python train_qwen3_8.py

    # Resume an interrupted training session:
    uv run python train_qwen3_8.py --resume

    # Train with fast development profile:
    uv run python train_qwen3_8.py --hardware-profile mac_dev

    # Train on Linux / CUDA workstation with RTX 6000 / A100:
    uv run python train_qwen3_8.py --hardware-profile rtx6000
"""

import sys
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
# Model Constants & Target Architecture Modules
# ============================================================================

DEFAULT_MODEL_NAME = "Qwen/Qwen3.8-27B"
DEFAULT_OUTPUT_DIR = "models/qwen3_8_27b_rust_lora"
DEFAULT_OFFLOAD_FOLDER = "models/qwen3_8_offload"

# Select default profile based on OS
IS_MACOS = sys.platform == "darwin"
DEFAULT_HARDWARE_PROFILE = "mac_48gb" if IS_MACOS else "rtx3090_4090"

# Qwen 3.8 27B Hybrid Architecture Targets:
# - Full-attention layers: q_proj, k_proj, v_proj, o_proj
# - Linear-attention layers: in_proj_a, in_proj_b, in_proj_qkv, in_proj_z, out_proj
# - MLP feed-forward layers: gate_proj, up_proj, down_proj
QWEN3_8_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_a",
    "in_proj_b",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

HARDWARE_PROFILES: dict[str, dict[str, Any]] = {
    "mac_48gb": {
        "batch_size": 1,
        "gradient_accumulation_steps": 16,
        "max_seq_length": 512,
        "load_in_4bit": False,
        "gradient_checkpointing": True,
        "offload_folder": DEFAULT_OFFLOAD_FOLDER,
        "save_steps": 25,
        "save_total_limit": 3,
        "description": "Apple Silicon Mac (32GB-48GB Unified RAM) - bf16 LoRA with offload & checkpointing",
    },
    "mac_dev": {
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_seq_length": 256,
        "load_in_4bit": False,
        "gradient_checkpointing": True,
        "offload_folder": DEFAULT_OFFLOAD_FOLDER,
        "save_steps": 5,
        "save_total_limit": 2,
        "description": "Apple Silicon Mac quick iteration/test profile (256 tokens)",
    },
    "rtx3090_4090": {
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 2048,
        "load_in_4bit": True,
        "gradient_checkpointing": True,
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA RTX 3090/4090 (24GB VRAM) - 4-bit QLoRA",
    },
    "rtx6000": {
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 4096,
        "load_in_4bit": False,
        "gradient_checkpointing": True,
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA RTX 6000 Ada (48GB VRAM) - 16-bit bf16 LoRA",
    },
    "a100_80gb": {
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 4096,
        "load_in_4bit": False,
        "gradient_checkpointing": False,
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA A100/H100 (80GB VRAM) - high-throughput 16-bit bf16",
    },
}


def train_qwen3_8(config: TrainingConfig) -> None:
    """Execute LoRA training workflow for Qwen 3.8 27B with resumability."""
    logger = setup_logging(name="rust_lora.qwen3_8")
    config.model_family = "Qwen3.8-27B"
    min_ram = 32.0 if IS_MACOS else 24.0

    execute_training_pipeline(
        config=config,
        prompt_formatter_fn=format_chatml_prompts,
        min_vram_gb=min_ram,
        allow_mps=True,
        logger=logger,
    )


def create_qwen3_8_config(args: Any) -> TrainingConfig:
    """Instantiate and configure TrainingConfig for Qwen 3.8 27B."""
    default_seq_len = 512 if IS_MACOS else 2048
    default_grad_accum = 16 if IS_MACOS else 8

    config = TrainingConfig(
        model_path=args.model_path or DEFAULT_MODEL_NAME,
        model_family="Qwen3.8-27B",
        max_seq_length=default_seq_len,
        batch_size=1,
        gradient_accumulation_steps=default_grad_accum,
        load_in_4bit=False if IS_MACOS else True,
        load_in_16bit=True if IS_MACOS else False,
        output_dir=args.output_dir or DEFAULT_OUTPUT_DIR,
        data_file=args.data_file or DEFAULT_TRAIN_FILE,
        target_modules=list(QWEN3_8_TARGET_MODULES),
        trust_remote_code=args.trust_remote_code,
        gradient_checkpointing=True if IS_MACOS else False,
        offload_folder=DEFAULT_OFFLOAD_FOLDER if IS_MACOS else None,
        save_steps=25 if IS_MACOS else 50,
        save_total_limit=3,
    )
    return apply_args_to_config(config, args, hardware_profiles=HARDWARE_PROFILES)


def main() -> None:
    """CLI entry point for Qwen 3.8 27B training."""
    default_seq_len = 512 if IS_MACOS else 2048
    default_grad_accum = 16 if IS_MACOS else 8

    parser = build_arg_parser(
        description="Train LoRA adapter on Qwen 3.8 27B for Rust Systems Development",
        default_model=DEFAULT_MODEL_NAME,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        hardware_profiles=HARDWARE_PROFILES,
        default_profile=DEFAULT_HARDWARE_PROFILE,
        default_max_seq_length=default_seq_len,
        default_batch_size=1,
        default_grad_accum=default_grad_accum,
        default_load_in_4bit=False if IS_MACOS else True,
        epilog=build_resume_epilog(
            "train_qwen3_8.py",
            """
  # Train on macOS with default memory-safe profile:
  uv run python train_qwen3_8.py

  # Train on 80GB GPU server:
  uv run python train_qwen3_8.py --hardware-profile a100_80gb
            """,
        ),
    )
    args = parser.parse_args()
    logger = setup_logging(name="rust_lora.qwen3_8")
    config = create_qwen3_8_config(args)
    min_ram = 32.0 if IS_MACOS else 24.0

    run_training_cli(
        config=config,
        args=args,
        train_fn=train_qwen3_8,
        min_vram_gb=min_ram,
        allow_mps=True,
        logger=logger,
    )


if __name__ == "__main__":
    main()
