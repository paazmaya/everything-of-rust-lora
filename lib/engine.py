"""
Execution engine for dual-backend model loading and SFT training.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import TrainingArguments
from trl.trainer.sft_trainer import SFTTrainer

from .config import TrainingConfig


def setup_logging(name: str = "rust_lora", verbose: bool = False) -> logging.Logger:
    """Configure standard logging with timestamps and levels."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)


def get_optimal_dtype() -> torch.dtype:
    """Determine the optimal floating point precision for current hardware (CUDA, MPS, CPU)."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Apple Silicon MPS supports bfloat16 in PyTorch 2.x
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def get_latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Find the latest checkpoint directory matching checkpoint-<step> in checkpoint_dir."""
    if not os.path.isdir(checkpoint_dir):
        return None
    checkpoints: list[tuple[int, str]] = []
    for entry in os.scandir(checkpoint_dir):
        if entry.is_dir() and entry.name.startswith("checkpoint-"):
            try:
                step = int(entry.name.split("-")[1])
                checkpoints.append((step, entry.path))
            except (IndexError, ValueError):
                continue
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def create_output_directories(
    output_dir: str, checkpoint_dir: str, logger: logging.Logger | None = None
) -> None:
    """Create output and checkpoint directories if they do not exist."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if logger:
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Checkpoint directory: {checkpoint_dir}")


def save_training_metadata(
    output_dir: str,
    config: TrainingConfig,
    extra_metadata: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> str:
    """Save training metadata manifest for artifact tracking."""
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "model_path": config.model_path,
        "model_family": config.model_family,
        "variant": config.variant,
        "max_seq_length": config.max_seq_length,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "load_in_4bit": config.load_in_4bit,
        "target_modules": config.target_modules,
    }
    if extra_metadata:
        manifest.update(extra_metadata)

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    if logger:
        logger.info(f"Saved training metadata to {manifest_path}")
    return manifest_path


def load_model_and_tokenizer(
    config: TrainingConfig,
    logger: logging.Logger | None = None,
) -> tuple[Any, Any]:
    """
    Load base model and configure LoRA adapter.

    Attempts loading with Unsloth FastLanguageModel first for optimized kernels.
    If FastLanguageModel fails (e.g. unsupported architecture, remote code differences),
    it seamlessly falls back to Hugging Face AutoModelForCausalLM + PEFT.
    """
    log = logger or logging.getLogger("rust_lora.engine")
    dtype = get_optimal_dtype()
    log.info(f"Target precision: {dtype} (4-bit: {config.load_in_4bit})")

    # Attempt 1: Unsloth FastLanguageModel
    try:
        from unsloth import FastLanguageModel

        log.info(f"Loading model via Unsloth FastLanguageModel: {config.model_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_path,
            max_seq_length=config.max_seq_length,
            dtype=dtype,
            load_in_4bit=config.load_in_4bit,
            load_in_16bit=config.load_in_16bit and not config.load_in_4bit,
            full_finetuning=False,
            trust_remote_code=config.trust_remote_code,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=config.target_modules,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            use_gradient_checkpointing="unsloth",
            random_state=config.seed,
        )
        log.info(
            f"Successfully loaded and configured LoRA via Unsloth (r={config.lora_r}, alpha={config.lora_alpha})"
        )
        return model, tokenizer

    except Exception as exc:
        log.info(
            f"Unsloth loading deferred ({exc}). Falling back to Transformers + PEFT pipeline..."
        )

    # Attempt 2: Standard Transformers + PEFT fallback
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(
        f"Loading tokenizer from {config.model_path} (trust_remote_code={config.trust_remote_code})..."
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
        "torch_dtype": dtype,
        "device_map": "auto",
    }

    if config.offload_folder:
        model_kwargs["offload_folder"] = config.offload_folder
        Path(config.offload_folder).mkdir(parents=True, exist_ok=True)
        log.info(f"Enabled model disk/CPU offloading to: {config.offload_folder}")

    if config.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    log.info(f"Loading base model via AutoModelForCausalLM from {config.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(config.model_path, **model_kwargs)

    if getattr(config, "gradient_checkpointing", False):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            log.info("Enabled gradient checkpointing for memory optimization.")

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    log.info("Successfully configured LoRA with PEFT")
    return model, tokenizer


def build_training_arguments(
    config: TrainingConfig,
    checkpoint_dir: str,
    report_to: list[str],
) -> TrainingArguments:
    """Construct TrainingArguments with dynamic platform optimization."""
    is_cuda = torch.cuda.is_available()
    is_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    use_bf16 = (is_cuda and torch.cuda.is_bf16_supported()) or is_mps
    use_fp16 = is_cuda and not use_bf16
    optimizer_name = "adamw_8bit" if is_cuda else "adamw_torch"

    return TrainingArguments(
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=config.logging_steps,
        output_dir=checkpoint_dir,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps if config.save_strategy == "steps" else 500,
        save_total_limit=config.save_total_limit,
        optim=optimizer_name,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        gradient_checkpointing=getattr(config, "gradient_checkpointing", False),
        seed=config.seed,
        report_to=report_to,
    )


def resolve_checkpoint_resumption(
    config: TrainingConfig,
    checkpoint_dir: str,
    logger: logging.Logger,
) -> str | None:
    """Determine the checkpoint path to resume from, if requested."""
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint.lower() == "latest":
            latest = get_latest_checkpoint(checkpoint_dir)
            if not latest:
                logger.warning(
                    f"No existing checkpoints found in {checkpoint_dir} to resume from. Starting from scratch."
                )
            return latest
        return config.resume_from_checkpoint

    if getattr(config, "resume", False):
        latest = get_latest_checkpoint(checkpoint_dir)
        if not latest:
            logger.warning(
                f"No existing checkpoints found in {checkpoint_dir} to resume from. Starting from scratch."
            )
        return latest

    return None


def run_sft_training(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    config: TrainingConfig,
    logger: logging.Logger | None = None,
) -> None:
    """Run supervised fine-tuning using SFTTrainer and save trained weights."""
    log = logger or logging.getLogger("rust_lora.engine")
    create_output_directories(
        config.output_dir, config.checkpoint_dir or f"{config.output_dir}/checkpoints", logger=log
    )

    if config.wandb_project:
        os.environ["WANDB_PROJECT"] = config.wandb_project
        if config.wandb_entity:
            os.environ["WANDB_ENTITY"] = config.wandb_entity

    report_to = ["wandb"] if config.wandb_project else []
    active_checkpoint_dir = config.checkpoint_dir or f"{config.output_dir}/checkpoints"

    training_args = build_training_arguments(
        config=config,
        checkpoint_dir=active_checkpoint_dir,
        report_to=report_to,
    )

    log.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        dataset_num_proc=config.dataset_num_proc,
        args=training_args,
    )

    log.info("=" * 80)
    log.info(f"STARTING {config.model_family.upper()} LORA TRAINING")
    log.info("=" * 80)
    log.info(f"  Model: {config.model_path}")
    log.info(f"  Epochs: {config.num_epochs}")
    log.info(f"  Batch size: {config.batch_size} (per device)")
    log.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    log.info(f"  Learning rate: {config.learning_rate}")
    log.info(f"  Max sequence length: {config.max_seq_length}")
    log.info(f"  4-bit quantization: {config.load_in_4bit}")

    resume_target = resolve_checkpoint_resumption(
        config=config,
        checkpoint_dir=active_checkpoint_dir,
        logger=log,
    )

    start_time = datetime.now()
    if resume_target:
        log.info(f"Resuming from checkpoint: {resume_target}")
        trainer_stats = trainer.train(resume_from_checkpoint=resume_target)
    else:
        trainer_stats = trainer.train()
    duration = datetime.now() - start_time

    log.info("=" * 80)
    log.info(f"TRAINING COMPLETE - Duration: {duration}")
    if hasattr(trainer_stats, "training_loss"):
        log.info(f"Final training loss: {trainer_stats.training_loss:.4f}")
    log.info("=" * 80)

    log.info(f"Saving final adapter and tokenizer to: {config.output_dir}")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    save_training_metadata(
        output_dir=config.output_dir,
        config=config,
        extra_metadata={"training_duration_seconds": duration.total_seconds()},
        logger=log,
    )
    log.info(f"Training workflow finished successfully! Saved to {config.output_dir}")
