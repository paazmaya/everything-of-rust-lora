#!/usr/bin/env python3
"""
Production-Grade IFM K2-Horizon-3.7B LoRA Training Script

This script trains a LoRA adapter on IFM/K2-Horizon-3.7B
(https://huggingface.co/IFM/K2-Horizon-3.7B).

Key Features:
- Designed for K2-Horizon-3.7B (dense 3.7B parameters, 512K native context window)
- Uses K2-Horizon chat template delimiters (<|ifm|im_start|>, <|ifm|im_end|>, <ifm|think>)
- Hardware-aware profiles (rtx4070, rtx3090_4090, a100, dev)
- Seamless dual-engine loading: uses Unsloth FastLanguageModel when available,
  with automatic fallback to Transformers + PEFT
- Dry-run mode for configuration and dependency validation
- Resumable training with checkpointing and artifact manifest generation

Usage:
    # Validate setup without training:
    uv run python train_k2_horizon.py --dry-run

    # Train using default RTX 3090/4090 profile (bfloat16):
    uv run python train_k2_horizon.py

    # Train on 12GB GPU (RTX 4070 with 4-bit QLoRA):
    uv run python train_k2_horizon.py --hardware-profile rtx4070

    # Train with local model path:
    uv run python train_k2_horizon.py --model-path /path/to/K2-Horizon-3.7B

Documentation References:
- HF Model Card: https://huggingface.co/IFM/K2-Horizon-3.7B
- SGLang Recipe: https://docs.sglang.io/cookbook/autoregressive/IFM/K2-Horizon
"""

import argparse
import json
import logging
import os
import sys
from collections.abc import Sized
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl.trainer.sft_trainer import SFTTrainer

# ============================================================================
# Default Configuration Constants
# ============================================================================

# Model Configuration
DEFAULT_MODEL_NAME = "IFM/K2-Horizon-3.7B"
TRUST_REMOTE_CODE = True
DEFAULT_LOAD_IN_4BIT = False
DEFAULT_LOAD_IN_16BIT = True

# Training Hyperparameters
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_NUM_EPOCHS = 3
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_WARMUP_STEPS = 10
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_LR_SCHEDULER_TYPE = "cosine"
DEFAULT_LOGGING_STEPS = 10
DEFAULT_SEED = 3407

# LoRA Configuration
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.0
DEFAULT_LORA_BIAS = "none"
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Chat Template & Delimiters
BOS_TOKEN = "<|ifm|begin_of_text|>"
EOS_TOKEN = "<|ifm|endoftext|>"
IM_START = "<|ifm|im_start|>"
IM_END = "<|ifm|im_end|>"
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert Rust programmer, specializing in systems programming, "
    "async Rust, ESP32 embedded development, and popular Rust libraries."
)

# Data & Output Configuration
DEFAULT_TRAIN_FILE = "data/datasets/train.jsonl"
REQUIRED_DATA_FIELDS = ["instruction", "input", "output"]
DEFAULT_OUTPUT_DIR = "models/k2_horizon_rust_lora"
DEFAULT_CHECKPOINT_DIR = "models/k2_horizon_rust_lora/checkpoints"
DEFAULT_SAVE_STRATEGY = "steps"
DEFAULT_SAVE_STEPS = 50

# Predefined Hardware Profiles
HARDWARE_PROFILES: dict[str, dict[str, Any]] = {
    "rtx4070": {
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 1024,
        "load_in_4bit": True,
        "description": "NVIDIA RTX 4070 (12GB VRAM) - 4-bit QLoRA default",
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
        "batch_size": 2,
        "gradient_accumulation_steps": 2,
        "max_seq_length": 4096,
        "load_in_4bit": False,
        "description": "NVIDIA A100 40GB - data center",
    },
    "a100_80gb": {
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "max_seq_length": 8192,
        "load_in_4bit": False,
        "description": "NVIDIA A100/H100 80GB - long context training",
    },
    "dev": {
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_seq_length": 512,
        "load_in_4bit": True,
        "description": "Development profile - fast iteration and minimal VRAM",
    },
}

DEFAULT_HARDWARE_PROFILE = "rtx3090_4090"

# ============================================================================
# Logging Configuration
# ============================================================================


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging with timestamps and log levels."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# ============================================================================
# Profile Helpers
# ============================================================================


def get_hardware_profile(profile_name: str = DEFAULT_HARDWARE_PROFILE) -> dict[str, Any]:
    """Get hardware profile settings from predefined constants."""
    if profile_name not in HARDWARE_PROFILES:
        logger.warning(
            f"Hardware profile '{profile_name}' not found. Available: {list(HARDWARE_PROFILES.keys())}"
        )
        return HARDWARE_PROFILES[DEFAULT_HARDWARE_PROFILE]

    profile = HARDWARE_PROFILES[profile_name]
    logger.info(
        f"Using hardware profile '{profile_name}': {profile.get('description', 'No description')}"
    )
    return profile


# ============================================================================
# Validation Functions
# ============================================================================


def check_cuda_available() -> bool:
    """Verify CUDA availability and report active GPU."""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. GPU acceleration is required for training.")
        return False

    logger.info(f"CUDA available. Device: {torch.cuda.get_device_name(0)}")
    return True


def check_gpu_memory(min_vram_gb: float = 12.0) -> bool:
    """Check if GPU has sufficient VRAM."""
    if not torch.cuda.is_available():
        return False

    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU VRAM: {total_memory:.1f} GB")

    if total_memory < min_vram_gb:
        logger.warning(
            f"GPU has {total_memory:.1f} GB, but at least {min_vram_gb:.1f} GB is recommended. "
            "Consider using 4-bit quantization with --load-in-4bit or --hardware-profile rtx4070."
        )
        return False

    return True


def check_dependencies() -> bool:
    """Check for required runtime dependencies."""
    missing = []

    try:
        import transformers  # noqa: F401
    except ImportError:
        missing.append("transformers")

    try:
        import trl  # noqa: F401
    except ImportError:
        missing.append("trl")

    try:
        import peft  # noqa: F401
    except ImportError:
        missing.append("peft")

    if missing:
        logger.error(f"Missing required dependencies: {', '.join(missing)}")
        logger.error("Install with: uv sync or pip install <package>")
        return False

    logger.info("All essential dependencies (transformers, trl, peft) are available")
    return True


def validate_data_file(data_path: str, required_fields: list[str] | None = None) -> bool:
    """Validate that training data exists and conforms to Alpaca JSONL schema."""
    if not os.path.exists(data_path):
        logger.error(f"Training data not found at: {data_path}")
        logger.error("Create training data in JSONL format with fields: instruction, input, output")
        return False

    logger.info(f"Found training data at: {data_path}")

    required_fields = required_fields or REQUIRED_DATA_FIELDS
    try:
        with open(data_path) as f:
            first_line = f.readline().strip()
            if not first_line:
                logger.error("Training data file is empty")
                return False

            record = json.loads(first_line)
            missing_fields = [f for f in required_fields if f not in record]

            if missing_fields:
                logger.error(
                    f"Missing fields in data: {missing_fields}. Required: {required_fields}"
                )
                return False

            logger.info(f"Data schema valid (sample contains keys: {list(record.keys())})")
            return True

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse training data as JSONL: {e}")
        return False


# ============================================================================
# Output & Manifest Management
# ============================================================================


def create_output_directories(output_dir: str, checkpoint_dir: str) -> None:
    """Create output directories if they do not exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")


def save_training_metadata(
    output_dir: str,
    model_path: str,
    config: dict[str, Any],
    hardware_profile: str,
) -> None:
    """Save training metadata manifest for artifact tracking."""
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "model_family": "K2-Horizon",
        "parameters": "3.7B",
        "hardware_profile": hardware_profile,
        "config": config,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved training metadata to {manifest_path}")


# ============================================================================
# Model Loading Helper (Unsloth with HF/PEFT Fallback)
# ============================================================================


def load_model_and_tokenizer(
    model_path: str = DEFAULT_MODEL_NAME,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    dtype: torch.dtype | None = None,
    load_in_4bit: bool = DEFAULT_LOAD_IN_4BIT,
    load_in_16bit: bool = DEFAULT_LOAD_IN_16BIT,
    trust_remote_code: bool = TRUST_REMOTE_CODE,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    target_modules: list[str] | None = None,
):
    """
    Load K2-Horizon-3.7B and configure LoRA.

    Attempts loading with Unsloth FastLanguageModel first for optimized kernel speeds.
    If FastLanguageModel does not support custom architecture or remote code,
    smoothly falls back to Hugging Face AutoModelForCausalLM + PEFT LoraConfig.
    """
    target_modules = target_modules or DEFAULT_TARGET_MODULES
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Try Unsloth FastLanguageModel first
    try:
        from unsloth import FastLanguageModel

        logger.info("Attempting model load via Unsloth FastLanguageModel...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            load_in_16bit=load_in_16bit,
            full_finetuning=False,
            trust_remote_code=trust_remote_code,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        logger.info("Successfully loaded and configured LoRA via Unsloth")
        return model, tokenizer

    except Exception as exc:
        logger.info(
            f"Unsloth load deferred ({exc}). Falling back to standard Transformers + PEFT pipeline..."
        )

    # Standard Transformers + PEFT fallback
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading tokenizer from {model_path} (trust_remote_code={trust_remote_code})...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
        "device_map": "auto",
    }

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    logger.info(f"Loading base model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    logger.info("Successfully configured LoRA with PEFT")
    return model, tokenizer


# ============================================================================
# Main Training Function & Configuration Dataclass
# ============================================================================


@dataclass
class K2TrainingParams:
    """Hyperparameters and runtime settings for K2-Horizon LoRA training."""

    model_path: str = DEFAULT_MODEL_NAME
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = DEFAULT_NUM_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    lr_scheduler_type: str = DEFAULT_LR_SCHEDULER_TYPE
    load_in_4bit: bool = DEFAULT_LOAD_IN_4BIT
    trust_remote_code: bool = TRUST_REMOTE_CODE
    output_dir: str = DEFAULT_OUTPUT_DIR
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    resume_from_checkpoint: str | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None
    data_file: str = DEFAULT_TRAIN_FILE
    save_strategy: str = DEFAULT_SAVE_STRATEGY
    save_steps: int = DEFAULT_SAVE_STEPS
    logging_steps: int = DEFAULT_LOGGING_STEPS


def prepare_training_dataset(data_file: str, tokenizer: Any, max_seq_length: int):
    """Load and format the Rust instruction dataset into K2-Horizon ChatML format."""
    k2_prompt_template = (
        f"{IM_START}system\n"
        f"{DEFAULT_SYSTEM_PROMPT}{IM_END}\n"
        f"{IM_START}user\n"
        "{user_message}"
        f"{IM_END}\n"
        f"{IM_START}assistant\n"
        "<ifm|think>\n"
        f"</ifm|think>{{output_message}}{IM_END}"
    )

    def formatting_prompts_func(examples: dict[str, list[str]]) -> dict[str, list[str]]:
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts: list[str] = []
        for instruction, input_text, output in zip(instructions, inputs, outputs, strict=False):
            user_msg = f"{instruction}\n{input_text}" if input_text else instruction
            text = k2_prompt_template.format(
                user_message=user_msg,
                output_message=output,
            )
            texts.append(text)
        return {"text": texts}

    logger.info(f"Loading dataset from: {data_file}")
    dataset = load_dataset("json", data_files=data_file, split="train")
    if isinstance(dataset, Sized):
        logger.info(f"Loaded {len(dataset)} training examples")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    sample_text = dataset[0]["text"]  # pyright: ignore[reportIndexIssue]
    sample_tokens = len(tokenizer.encode(sample_text))
    logger.info(f"Sample token length: {sample_tokens} tokens (max limit: {max_seq_length})")
    return dataset


def train_k2_horizon(params: K2TrainingParams | None = None, **kwargs: Any) -> None:
    """
    Train a LoRA adapter on IFM K2-Horizon-3.7B.

    Args:
        params: K2TrainingParams dataclass instance containing hyperparameters.
        **kwargs: Overrides for any field in K2TrainingParams.
    """
    if params is None:
        params = K2TrainingParams(**kwargs)
    elif kwargs:
        for k, v in kwargs.items():
            if hasattr(params, k):
                setattr(params, k, v)

    logger.info("=" * 80)
    logger.info("IFM K2-HORIZON-3.7B LORA TRAINING")
    logger.info("=" * 80)

    # Determine optimal precision
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"Using dtype: {dtype} (4-bit: {params.load_in_4bit})")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_path=params.model_path,
        max_seq_length=params.max_seq_length,
        dtype=dtype,
        load_in_4bit=params.load_in_4bit,
        load_in_16bit=not params.load_in_4bit,
        trust_remote_code=params.trust_remote_code,
    )

    create_output_directories(params.output_dir, params.checkpoint_dir)
    dataset = prepare_training_dataset(params.data_file, tokenizer, params.max_seq_length)

    training_args = TrainingArguments(
        per_device_train_batch_size=params.batch_size,
        gradient_accumulation_steps=params.gradient_accumulation_steps,
        warmup_steps=params.warmup_steps,
        num_train_epochs=params.num_epochs,
        learning_rate=params.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=params.logging_steps,
        optim="adamw_8bit",
        weight_decay=params.weight_decay,
        lr_scheduler_type=params.lr_scheduler_type,
        seed=3407,
        output_dir=params.checkpoint_dir,
        save_strategy=params.save_strategy,
        save_steps=params.save_steps,
        save_total_limit=3,
        report_to="wandb" if params.wandb_project else "none",
    )

    if params.wandb_project:
        os.environ["WANDB_PROJECT"] = params.wandb_project
        if params.wandb_entity:
            os.environ["WANDB_ENTITY"] = params.wandb_entity

    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,  # pyright: ignore[reportCallIssue]
        train_dataset=dataset,  # pyright: ignore[reportArgumentType]
        dataset_text_field="text",  # pyright: ignore[reportCallIssue]
        max_seq_length=params.max_seq_length,  # pyright: ignore[reportCallIssue]
        dataset_num_proc=2,  # pyright: ignore[reportCallIssue]
        packing=False,  # pyright: ignore[reportCallIssue]
        args=training_args,
    )

    logger.info("Starting training...")
    start_time = datetime.now()
    trainer_stats = trainer.train(resume_from_checkpoint=params.resume_from_checkpoint)
    duration = datetime.now() - start_time

    logger.info("=" * 80)
    logger.info(f"Training completed in {duration}")
    logger.info(f"Final training loss: {trainer_stats.training_loss:.4f}")
    logger.info("=" * 80)

    # Save final model and tokenizer
    logger.info(f"Saving LoRA adapter to {params.output_dir}...")
    model.save_pretrained(params.output_dir)
    tokenizer.save_pretrained(params.output_dir)
    logger.info(f"Successfully saved adapter to {params.output_dir}")


# ============================================================================
# CLI Argument Parser
# ============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LoRA adapter on IFM K2-Horizon-3.7B for Rust knowledge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--hardware-profile",
        type=str,
        default=None,
        help=f"Hardware profile name ({', '.join(HARDWARE_PROFILES.keys())})",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=f"HuggingFace model ID or local directory path (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help=f"Maximum sequence length (default: {DEFAULT_MAX_SEQ_LENGTH})",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help=f"Number of training epochs (default: {DEFAULT_NUM_EPOCHS})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Force 4-bit quantization (QLoRA)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Directory to save final LoRA adapter (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help=f"Directory to save training checkpoints (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help=f"Path to training data (JSONL format) (default: {DEFAULT_TRAIN_FILE})",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases team/entity name",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration, dependencies, and data without starting training",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose DEBUG logging",
    )

    return parser.parse_args()


# ============================================================================
# Main Entry Point
# ============================================================================


def resolve_training_params(args: argparse.Namespace) -> tuple[K2TrainingParams, str]:
    """Resolve training configuration with CLI > Profile > Default constants precedence."""
    profile_name = args.hardware_profile or DEFAULT_HARDWARE_PROFILE
    hw_profile = get_hardware_profile(profile_name)

    params = K2TrainingParams(
        model_path=args.model_path or DEFAULT_MODEL_NAME,
        batch_size=(
            args.batch_size if args.batch_size is not None else hw_profile.get("batch_size", 1)
        ),
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
            if args.gradient_accumulation_steps is not None
            else hw_profile.get("gradient_accumulation_steps", 4)
        ),
        max_seq_length=(
            args.max_seq_length
            if args.max_seq_length is not None
            else hw_profile.get("max_seq_length", DEFAULT_MAX_SEQ_LENGTH)
        ),
        load_in_4bit=(
            args.load_in_4bit
            if args.load_in_4bit
            else hw_profile.get("load_in_4bit", DEFAULT_LOAD_IN_4BIT)
        ),
        num_epochs=args.num_epochs or DEFAULT_NUM_EPOCHS,
        learning_rate=args.learning_rate or DEFAULT_LEARNING_RATE,
        warmup_steps=DEFAULT_WARMUP_STEPS,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        lr_scheduler_type=DEFAULT_LR_SCHEDULER_TYPE,
        trust_remote_code=TRUST_REMOTE_CODE,
        output_dir=args.output_dir or DEFAULT_OUTPUT_DIR,
        checkpoint_dir=args.checkpoint_dir or DEFAULT_CHECKPOINT_DIR,
        resume_from_checkpoint=args.resume_from_checkpoint,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        data_file=args.data_file or DEFAULT_TRAIN_FILE,
        save_strategy=DEFAULT_SAVE_STRATEGY,
        save_steps=DEFAULT_SAVE_STEPS,
        logging_steps=DEFAULT_LOGGING_STEPS,
    )
    return params, profile_name


def _report_dry_run_results(deps_ok: bool, data_ok: bool, cuda_ok: bool, vram_ok: bool) -> None:
    """Print the dry-run validation report and exit with appropriate status code."""
    logger.info("DRY-RUN VALIDATION REPORT:")
    logger.info(f"  Dependencies: {'PASSED' if deps_ok else 'FAILED'}")
    logger.info(f"  Training Data: {'PASSED' if data_ok else 'FAILED'}")
    logger.info(f"  CUDA GPU:     {'PASSED' if cuda_ok else 'FAILED'}")
    logger.info(f"  GPU Memory:   {'PASSED' if vram_ok else 'WARNING'}")
    if deps_ok and data_ok:
        logger.info("Dry-run validation successful! Ready for training.")
        sys.exit(0)
    logger.error("Dry-run validation failed. Please check errors above.")
    sys.exit(1)


def execute_preflight_checks(data_file: str, is_dry_run: bool) -> None:
    """Run environment, dependency, and dataset validation."""
    deps_ok = check_dependencies()
    data_ok = validate_data_file(data_file)
    cuda_ok = check_cuda_available()
    vram_ok = check_gpu_memory(min_vram_gb=12.0) if cuda_ok else False

    if is_dry_run:
        _report_dry_run_results(deps_ok, data_ok, cuda_ok, vram_ok)

    if not (deps_ok and data_ok):
        logger.error("Pre-training validation checks failed. Aborting.")
        sys.exit(1)

    if not cuda_ok:
        logger.error("CUDA is required for training. Aborting.")
        sys.exit(1)


def main():
    """Main execution flow with validation and training."""
    args = parse_args()
    global logger
    logger = setup_logging(args.verbose)

    params, profile_name = resolve_training_params(args)

    logger.info("=" * 80)
    logger.info("Resolved Configuration:")
    logger.info(f"  Model Path:          {params.model_path}")
    logger.info(f"  Hardware Profile:    {profile_name}")
    logger.info(f"  Batch Size:          {params.batch_size}")
    logger.info(
        f"  Gradient Accum:      {params.gradient_accumulation_steps} (Effective: {params.batch_size * params.gradient_accumulation_steps})"
    )
    logger.info(f"  Max Sequence Length: {params.max_seq_length}")
    logger.info(f"  4-bit Quantization:  {params.load_in_4bit}")
    logger.info(f"  Epochs:              {params.num_epochs}")
    logger.info(f"  Learning Rate:       {params.learning_rate}")
    logger.info(f"  Output Dir:          {params.output_dir}")
    logger.info(f"  Data File:           {params.data_file}")
    logger.info("=" * 80)

    execute_preflight_checks(params.data_file, args.dry_run)

    # Save manifest
    save_training_metadata(
        output_dir=params.output_dir,
        model_path=params.model_path,
        config={
            "batch_size": params.batch_size,
            "gradient_accumulation_steps": params.gradient_accumulation_steps,
            "max_seq_length": params.max_seq_length,
            "load_in_4bit": params.load_in_4bit,
            "num_epochs": params.num_epochs,
            "learning_rate": params.learning_rate,
        },
        hardware_profile=profile_name,
    )

    # Run training
    train_k2_horizon(params)


if __name__ == "__main__":
    main()
