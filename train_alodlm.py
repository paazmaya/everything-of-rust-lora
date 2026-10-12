#!/usr/bin/env python3
"""
Production-Grade Amazon ALoDLM-8B LoRA Training Script

Trains a LoRA adapter on amazon/ALoDLM-8B for Rust systems and embedded development.
Targets all full-attention projection matrices (q_proj, k_proj, v_proj, o_proj) and
MLP feed-forward projection matrices (gate_proj, up_proj, down_proj) across the
36-layer Qwen3 backbone architecture.

Features:
    - Standard ChatML prompt formatting matching ALoDLM chat template
    - Hardware-aware profiles for Apple Silicon Mac (MPS) and NVIDIA GPUs (CUDA)
    - Checkpoint resumption (--resume and --resume-from-checkpoint)
    - Preflight dry-run validation (--dry-run)
    - Reusable TrainingConfig and pipeline from lib/

Usage:
    # Dry-run validation (no weight download required):
    uv run python train_alodlm.py --dry-run

    # Train using default profile (mac_48gb on macOS, rtx3090_4090 on CUDA):
    uv run python train_alodlm.py

    # Resume interrupted training from the latest checkpoint:
    uv run python train_alodlm.py --resume

    # Resume from a specific checkpoint:
    uv run python train_alodlm.py --resume-from-checkpoint models/alodlm_8b_rust_lora/checkpoints/checkpoint-50

    # Fast local development test:
    uv run python train_alodlm.py --hardware-profile dev

    # Train with local model directory:
    uv run python train_alodlm.py --model-path /path/to/ALoDLM-8B
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

DEFAULT_MODEL_NAME = "amazon/ALoDLM-8B"
DEFAULT_OUTPUT_DIR = "models/alodlm_8b_rust_lora"

IS_MACOS = sys.platform == "darwin"
DEFAULT_HARDWARE_PROFILE = "mac_48gb" if IS_MACOS else "rtx3090_4090"

# ALoDLM-8B uses a 36-layer Qwen3 causal backbone with full attention across all layers.
# Target projections:
# - Full-attention projections: q_proj, k_proj, v_proj, o_proj
# - MLP feed-forward projections: gate_proj, up_proj, down_proj
ALODLM_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

HARDWARE_PROFILES: dict[str, dict[str, Any]] = {
    "mac_48gb": {
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 1024,
        "load_in_4bit": False,
        "gradient_checkpointing": True,
        "offload_folder": None,
        "save_steps": 25,
        "save_total_limit": 3,
        "description": "Apple Silicon Mac (32GB-48GB Unified RAM) - 16-bit bf16 LoRA with gradient checkpointing",
    },
    "mac_fast": {
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 512,
        "load_in_4bit": False,
        "gradient_checkpointing": True,
        "offload_folder": None,
        "save_steps": 25,
        "save_total_limit": 3,
        "description": "Apple Silicon Mac fast iteration profile (512 tokens, 16-bit bf16)",
    },
    "rtx4070": {
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 1024,
        "load_in_4bit": True,
        "gradient_checkpointing": True,
        "offload_folder": None,
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA RTX 4070 / 3060 (12GB VRAM) - 4-bit QLoRA",
    },
    "rtx3090_4090": {
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_seq_length": 2048,
        "load_in_4bit": True,
        "gradient_checkpointing": True,
        "offload_folder": None,
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA RTX 3090 / 4090 (24GB VRAM) - 4-bit QLoRA default",
    },
    "rtx6000": {
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 4096,
        "load_in_4bit": False,
        "gradient_checkpointing": True,
        "offload_folder": None,
        "save_steps": 50,
        "save_total_limit": 3,
        "description": "NVIDIA RTX 6000 Ada (48GB VRAM) - 16-bit bf16 LoRA",
    },
    "a100": {
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 4096,
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
        "load_in_4bit": True,
        "gradient_checkpointing": True,
        "offload_folder": None,
        "save_steps": 5,
        "save_total_limit": 2,
        "description": "Development test profile - fast iteration and minimal memory footprint",
    },
}


def train_alodlm(config: TrainingConfig) -> None:
    """Execute LoRA training workflow for Amazon ALoDLM-8B with resumability."""
    logger = setup_logging(name="rust_lora.alodlm")
    config.model_family = "ALoDLM"
    config.variant = "8B"
    min_ram = 16.0 if IS_MACOS else 12.0

    execute_training_pipeline(
        config=config,
        prompt_formatter_fn=format_chatml_prompts,
        min_vram_gb=min_ram,
        allow_mps=True,
        logger=logger,
    )


def create_alodlm_config(args: Any) -> TrainingConfig:
    """Instantiate and configure TrainingConfig for Amazon ALoDLM-8B."""
    default_seq_len = 1024 if IS_MACOS else 2048
    default_grad_accum = 8

    config = TrainingConfig(
        model_path=args.model_path or DEFAULT_MODEL_NAME,
        model_family="ALoDLM",
        variant="8B",
        max_seq_length=default_seq_len,
        batch_size=1,
        gradient_accumulation_steps=default_grad_accum,
        load_in_4bit=not IS_MACOS,
        load_in_16bit=IS_MACOS,
        output_dir=args.output_dir or DEFAULT_OUTPUT_DIR,
        data_file=args.data_file or DEFAULT_TRAIN_FILE,
        target_modules=list(ALODLM_TARGET_MODULES),
        trust_remote_code=args.trust_remote_code,
        gradient_checkpointing=True,
        offload_folder=None,
        save_steps=25 if IS_MACOS else 50,
        save_total_limit=3,
    )
    return apply_args_to_config(config, args, hardware_profiles=HARDWARE_PROFILES)


def main() -> None:
    """CLI entry point for Amazon ALoDLM-8B training."""
    default_seq_len = 1024 if IS_MACOS else 2048
    default_grad_accum = 8

    parser = build_arg_parser(
        description="Train LoRA adapter on Amazon ALoDLM-8B for Rust Systems Development",
        default_model=DEFAULT_MODEL_NAME,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        hardware_profiles=HARDWARE_PROFILES,
        default_profile=DEFAULT_HARDWARE_PROFILE,
        default_max_seq_length=default_seq_len,
        default_batch_size=1,
        default_grad_accum=default_grad_accum,
        default_load_in_4bit=not IS_MACOS,
        epilog=build_resume_epilog(
            "train_alodlm.py",
            """
  # Train on macOS Apple Silicon (16-bit bf16 with gradient checkpointing):
  uv run python train_alodlm.py

  # Train on CUDA workstation (e.g. RTX 3090/4090 or RTX 4070):
  uv run python train_alodlm.py --hardware-profile rtx3090_4090

  # Fast development iteration:
  uv run python train_alodlm.py --hardware-profile dev
            """,
        ),
    )
    args = parser.parse_args()
    logger = setup_logging(name="rust_lora.alodlm")
    config = create_alodlm_config(args)
    min_ram = 16.0 if IS_MACOS else 12.0

    run_training_cli(
        config=config,
        args=args,
        train_fn=train_alodlm,
        min_vram_gb=min_ram,
        allow_mps=True,
        logger=logger,
    )


if __name__ == "__main__":
    main()
