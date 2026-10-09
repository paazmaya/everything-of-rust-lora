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
    load_and_prepare_dataset,
    load_model_and_tokenizer,
    run_preflight_checks,
    run_sft_training,
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

QWEN_PROMPT = """<|im_start|>system
You are an expert Rust programmer, specializing in systems programming, async Rust, ESP32 embedded development, and popular Rust libraries.<|im_end|>
<|im_start|>user
{}{}<|im_end|>
<|im_start|>assistant
{}"""


def format_qwen_prompts(examples: dict[str, list[str]]) -> dict[str, list[str]]:
    """Format instruction examples into ChatML format for Qwen 3.8."""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts: list[str] = []
    for instruction, input_text, output in zip(instructions, inputs, outputs, strict=False):
        user_msg = f"{instruction}\n{input_text}" if input_text else instruction
        text = QWEN_PROMPT.format(user_msg, "", output) + "<|im_end|>"
        texts.append(text)
    return {"text": texts}


def train_qwen3_8(config: TrainingConfig) -> None:
    """Execute LoRA training workflow for Qwen 3.8 27B."""
    logger = setup_logging(name="rust_lora.qwen3_8")
    config.model_family = "Qwen3.8-27B"

    # 1. Preflight validation
    min_ram = 32.0 if IS_MACOS else 24.0
    if not run_preflight_checks(config, min_vram_gb=min_ram, logger=logger, allow_mps=True):
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

    # 4. Train and save model (with resumption support)
    run_sft_training(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        config=config,
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
        epilog="""
Examples:
  # Dry-run validation check:
  uv run python train_qwen3_8.py --dry-run

  # Train on macOS with default memory-safe profile:
  uv run python train_qwen3_8.py

  # Resume interrupted training from the latest checkpoint:
  uv run python train_qwen3_8.py --resume

  # Train on 80GB GPU server:
  uv run python train_qwen3_8.py --hardware-profile a100_80gb
        """,
    )
    args = parser.parse_args()
    logger = setup_logging(name="rust_lora.qwen3_8")
    config = create_qwen3_8_config(args)

    if args.dry_run:
        logger.info("Executing dry-run validation for Qwen 3.8 27B...")
        min_ram = 32.0 if IS_MACOS else 24.0
        passed = run_preflight_checks(config, min_vram_gb=min_ram, logger=logger, allow_mps=True)
        sys.exit(0 if passed else 1)

    try:
        train_qwen3_8(config)
    except KeyboardInterrupt:
        logger.info(
            "Training interrupted by user. Run with --resume to continue from last checkpoint."
        )
        sys.exit(130)
    except Exception:
        logger.exception("Training failed with unexpected exception")
        sys.exit(1)


if __name__ == "__main__":
    main()
