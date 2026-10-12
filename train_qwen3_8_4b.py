#!/usr/bin/env python3
"""
Production-Grade Qwen 3.8 4B Distill LoRA Training Script

Trains a LoRA adapter on empero-ai/Qwen3.8-4B-Distill for Rust systems and embedded development.
Targets all 12 linear-attention, full-attention, and MLP feed-forward projection matrices
across the 32-layer hybrid architecture.

Features:
    - 100% in-memory execution for Apple Silicon (zero SSD offload thrashing)
    - Full hybrid attention LoRA targeting (linear_attn + self_attn + mlp)
    - Hardware-aware profiles for Apple Silicon Mac (MPS) and NVIDIA GPUs (CUDA)
    - Resumable training from checkpoints (--resume)
    - Preflight dry-run validation (--dry-run)

Usage:
    # Dry-run validation:
    uv run python train_qwen3_8_4b.py --dry-run

    # Train using default profile (mac_native on Apple Silicon, rtx4070 on CUDA):
    uv run python train_qwen3_8_4b.py

    # Resume interrupted training from the latest checkpoint:
    uv run python train_qwen3_8_4b.py --resume

    # Resume from a specific checkpoint:
    uv run python train_qwen3_8_4b.py --resume-from-checkpoint models/qwen3_8_4b_rust_lora/checkpoints/checkpoint-50

    # Fast local development test:
    uv run python train_qwen3_8_4b.py --hardware-profile dev

    # Train with local model directory:
    uv run python train_qwen3_8_4b.py --model-path /path/to/Qwen3.8-4B-Distill
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

DEFAULT_MODEL_NAME = "empero-ai/Qwen3.8-4B-Distill"
DEFAULT_OUTPUT_DIR = "models/qwen3_8_4b_rust_lora"

IS_MACOS = sys.platform == "darwin"
DEFAULT_HARDWARE_PROFILE = "mac_native" if IS_MACOS else "rtx4070"

# Qwen 3.8 4B Hybrid Architecture Targets:
# - Full-attention layers (8 layers): q_proj, k_proj, v_proj, o_proj
# - Linear-attention layers (24 layers): in_proj_a, in_proj_b, in_proj_qkv, in_proj_z, out_proj
# - MLP feed-forward layers (all 32 layers): gate_proj, up_proj, down_proj
QWEN3_8_4B_TARGET_MODULES = [
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
    "mac_native": {
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "gradient_checkpointing": True,
        "offload_folder": None,
        "save_steps": 25,
        "save_total_limit": 3,
        "description": "Apple Silicon Mac (16GB-48GB Unified RAM) - 100% in-memory 16-bit bf16 LoRA (zero offload)",
    },
    "mac_fast": {
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 1024,
        "load_in_4bit": False,
        "gradient_checkpointing": True,
        "offload_folder": None,
        "save_steps": 25,
        "save_total_limit": 3,
        "description": "Apple Silicon Mac fast iteration profile (1024 tokens, 16-bit bf16)",
    },
    "rtx4070": {
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "gradient_checkpointing": True,
        "offload_folder": None,
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA RTX 4070 / 3060 (12GB VRAM) - 16-bit bf16 LoRA",
    },
    "rtx3090_4090": {
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 4096,
        "load_in_4bit": False,
        "gradient_checkpointing": True,
        "offload_folder": None,
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA RTX 3090 / 4090 (24GB VRAM) - 16-bit bf16 LoRA with extended context",
    },
    "a100": {
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "max_seq_length": 8192,
        "load_in_4bit": False,
        "gradient_checkpointing": False,
        "offload_folder": None,
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA A100 / H100 (40GB-80GB VRAM) - high-throughput 16-bit bf16",
    },
    "dev": {
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_seq_length": 512,
        "load_in_4bit": False,
        "gradient_checkpointing": True,
        "offload_folder": None,
        "save_steps": 5,
        "save_total_limit": 2,
        "description": "Development test profile - fast iteration and minimal memory footprint",
    },
}


def train_qwen3_8_4b(config: TrainingConfig) -> None:
    """Execute LoRA training workflow for Qwen 3.8 4B Distill with resumability."""
    logger = setup_logging(name="rust_lora.qwen3_8_4b")
    config.model_family = "Qwen3.8-4B"
    min_ram = 8.0 if IS_MACOS else 12.0

    execute_training_pipeline(
        config=config,
        prompt_formatter_fn=format_chatml_prompts,
        min_vram_gb=min_ram,
        allow_mps=True,
        logger=logger,
    )


def create_qwen3_8_4b_config(args: Any) -> TrainingConfig:
    """Instantiate and configure TrainingConfig for Qwen 3.8 4B Distill."""
    default_seq_len = 2048
    default_grad_accum = 8 if IS_MACOS else 4

    config = TrainingConfig(
        model_path=args.model_path or DEFAULT_MODEL_NAME,
        model_family="Qwen3.8-4B",
        max_seq_length=default_seq_len,
        batch_size=1,
        gradient_accumulation_steps=default_grad_accum,
        load_in_4bit=False,
        load_in_16bit=True,
        output_dir=args.output_dir or DEFAULT_OUTPUT_DIR,
        data_file=args.data_file or DEFAULT_TRAIN_FILE,
        target_modules=list(QWEN3_8_4B_TARGET_MODULES),
        trust_remote_code=args.trust_remote_code,
        gradient_checkpointing=True,
        offload_folder=None,
        save_steps=25 if IS_MACOS else 50,
        save_total_limit=3,
    )
    return apply_args_to_config(config, args, hardware_profiles=HARDWARE_PROFILES)


def main() -> None:
    """CLI entry point for Qwen 3.8 4B Distill training."""
    default_seq_len = 2048
    default_grad_accum = 8 if IS_MACOS else 4

    parser = build_arg_parser(
        description="Train LoRA adapter on Qwen 3.8 4B Distill for Rust Systems Development",
        default_model=DEFAULT_MODEL_NAME,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        hardware_profiles=HARDWARE_PROFILES,
        default_profile=DEFAULT_HARDWARE_PROFILE,
        default_max_seq_length=default_seq_len,
        default_batch_size=1,
        default_grad_accum=default_grad_accum,
        default_load_in_4bit=False,
        epilog=build_resume_epilog(
            "train_qwen3_8_4b.py",
            """
  # Train on macOS Apple Silicon (native in-memory, zero disk offload):
  uv run python train_qwen3_8_4b.py

  # Train on CUDA workstation (e.g. RTX 4070 or RTX 4090):
  uv run python train_qwen3_8_4b.py --hardware-profile rtx4070

  # Fast development iteration:
  uv run python train_qwen3_8_4b.py --hardware-profile dev
            """,
        ),
    )
    args = parser.parse_args()
    logger = setup_logging(name="rust_lora.qwen3_8_4b")
    config = create_qwen3_8_4b_config(args)
    min_ram = 8.0 if IS_MACOS else 12.0

    run_training_cli(
        config=config,
        args=args,
        train_fn=train_qwen3_8_4b,
        min_vram_gb=min_ram,
        allow_mps=True,
        logger=logger,
    )


if __name__ == "__main__":
    main()
