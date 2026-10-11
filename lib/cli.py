"""
Shared CLI argument parser and argument binding helpers.
"""

import argparse
import logging
import sys
from collections.abc import Callable
from typing import Any

from .config import DEFAULT_TRAIN_FILE, TrainingConfig, apply_hardware_profile
from .validation import run_preflight_checks


def build_resume_epilog(script_name: str, extra_examples: str = "") -> str:
    """Generate standardized epilog containing resume and dry-run instructions."""
    base_examples = f"""
Examples:
  # Dry-run preflight check:
  uv run python {script_name} --dry-run

  # Standard training run:
  uv run python {script_name}

  # Resume interrupted training from the latest checkpoint:
  uv run python {script_name} --resume

  # Resume from a specific checkpoint:
  uv run python {script_name} --resume-from-checkpoint models/.../checkpoint-50
"""
    if extra_examples:
        base_examples += f"\n{extra_examples.strip()}\n"
    return base_examples


def build_arg_parser(
    description: str,
    default_model: str,
    default_output_dir: str,
    hardware_profiles: dict[str, dict[str, Any]] | None = None,
    default_profile: str | None = None,
    default_max_seq_length: int = 1024,
    default_batch_size: int = 1,
    default_grad_accum: int = 4,
    default_num_epochs: int = 3,
    default_learning_rate: float = 2e-4,
    default_load_in_4bit: bool = False,
    epilog: str | None = None,
) -> argparse.ArgumentParser:
    """Construct a standardized CLI argument parser for model training."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate environment, dependencies, and dataset without starting training.",
    )

    profile_help = "Hardware profile to configure batch sizes and sequence length"
    if hardware_profiles:
        profile_help += f" (available: {', '.join(hardware_profiles.keys())})"
    if default_profile:
        profile_help += f" [default: {default_profile}]"

    parser.add_argument(
        "--hardware-profile",
        type=str,
        default=default_profile,
        help=profile_help,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=default_model,
        help=f"Model identifier on HuggingFace Hub or local directory path [default: {default_model}]",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Per-device training batch size [default: {default_batch_size}]",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help=f"Number of gradient accumulation steps [default: {default_grad_accum}]",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help=f"Maximum sequence length in tokens [default: {default_max_seq_length}]",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=default_num_epochs,
        help=f"Number of training epochs [default: {default_num_epochs}]",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=default_learning_rate,
        help=f"Optimizer learning rate [default: {default_learning_rate}]",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=default_load_in_4bit,
        help="Force 4-bit quantization (QLoRA).",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=DEFAULT_TRAIN_FILE,
        help=f"Path to training dataset JSONL file [default: {DEFAULT_TRAIN_FILE}]",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output_dir,
        help=f"Directory to save final LoRA weights and tokenizer [default: {default_output_dir}]",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to store training checkpoints (defaults to output_dir/checkpoints).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Number of update steps between saving checkpoints [default: 50].",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to retain in checkpoint directory [default: 3].",
    )
    parser.add_argument(
        "--save-strategy",
        type=str,
        default=None,
        choices=["steps", "epoch", "no"],
        help="Checkpoint save strategy ('steps', 'epoch', or 'no') [default: steps].",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Automatically resume training from the latest checkpoint found in checkpoint directory.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to specific checkpoint directory (or 'latest') to resume training from.",
    )
    parser.add_argument(
        "--offload-folder",
        type=str,
        default=None,
        help="Directory for Accelerate CPU/disk offloading of model weights when VRAM is constrained.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name for experiment tracking.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases team/entity name.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable execution of custom remote code from HuggingFace repository.",
    )
    return parser


def apply_args_to_config(
    config: TrainingConfig,
    args: argparse.Namespace,
    hardware_profiles: dict[str, dict[str, Any]] | None = None,
) -> TrainingConfig:
    """Update TrainingConfig instance with parsed CLI arguments and selected profile."""
    profile_name = getattr(args, "hardware_profile", None)
    if hardware_profiles and profile_name:
        config = apply_hardware_profile(config, hardware_profiles, profile_name)

    fields = [
        "model_path",
        "batch_size",
        "gradient_accumulation_steps",
        "max_seq_length",
        "num_epochs",
        "learning_rate",
        "data_file",
        "output_dir",
        "checkpoint_dir",
        "save_steps",
        "save_total_limit",
        "save_strategy",
        "resume_from_checkpoint",
        "offload_folder",
        "wandb_project",
        "wandb_entity",
    ]
    for field_name in fields:
        val = getattr(args, field_name, None)
        if val is not None:
            setattr(config, field_name, val)

    if getattr(args, "checkpoint_dir", None) is None:
        config.checkpoint_dir = f"{config.output_dir}/checkpoints"

    if getattr(args, "resume", False):
        config.resume = True

    if getattr(args, "load_in_4bit", False):
        config.load_in_4bit = True
        config.load_in_16bit = False

    if getattr(args, "trust_remote_code", False):
        config.trust_remote_code = True

    return config


def run_training_cli(
    config: TrainingConfig,
    args: argparse.Namespace,
    train_fn: Callable[[TrainingConfig], None],
    min_vram_gb: float = 12.0,
    allow_mps: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    """Standardized CLI runner handling --dry-run, interruptions, and exceptions."""
    family_slug = (
        getattr(config, "model_family", "model").lower().replace(" ", "_").replace("-", "_")
    )
    log = logger or logging.getLogger(f"rust_lora.{family_slug}")

    if getattr(args, "dry_run", False):
        log.info(f"Executing dry-run validation for {config.model_family}...")
        passed = run_preflight_checks(
            config, min_vram_gb=min_vram_gb, logger=log, allow_mps=allow_mps
        )
        sys.exit(0 if passed else 1)

    try:
        train_fn(config)
    except KeyboardInterrupt:
        log.info(
            "Training interrupted by user. Run with --resume to continue from last checkpoint."
        )
        sys.exit(130)
    except Exception:
        log.exception("Training failed with unexpected exception")
        sys.exit(1)
