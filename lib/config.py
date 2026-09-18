"""
Central configuration data structures and profile helpers for model training.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

DEFAULT_TRAIN_FILE = "data/datasets/train.jsonl"
DEFAULT_REQUIRED_FIELDS = ["instruction", "input", "output"]
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class TrainingConfig:
    """Unified configuration for model LoRA fine-tuning."""

    # Model parameters
    model_path: str
    model_family: str = "Unknown"
    variant: str = ""
    trust_remote_code: bool = False

    # Training hyperparameters
    max_seq_length: int = 1024
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    seed: int = 3407

    # Precision & Quantization
    load_in_4bit: bool = False
    load_in_16bit: bool = True

    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))

    # Data & Paths
    data_file: str = DEFAULT_TRAIN_FILE
    output_dir: str = "models/rust_lora"
    checkpoint_dir: str | None = None
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 3
    resume_from_checkpoint: str | None = None
    dataset_num_proc: int = 2

    # Experiment tracking
    wandb_project: str | None = None
    wandb_entity: str | None = None

    def __post_init__(self) -> None:
        if self.checkpoint_dir is None:
            self.checkpoint_dir = f"{self.output_dir}/checkpoints"


def apply_hardware_profile(
    config: TrainingConfig,
    profiles: dict[str, dict[str, Any]],
    profile_name: str | None,
    logger: logging.Logger | None = None,
) -> TrainingConfig:
    """Apply hardware profile settings to the training configuration."""
    if not profile_name:
        return config

    if profile_name not in profiles:
        available = list(profiles.keys())
        if logger:
            logger.warning(
                f"Hardware profile '{profile_name}' not found. Available: {available}. Skipping."
            )
        return config

    profile = profiles[profile_name]
    if logger:
        description = profile.get("description", "No description")
        logger.info(f"Applying hardware profile '{profile_name}': {description}")

    if "batch_size" in profile:
        config.batch_size = profile["batch_size"]
    if "gradient_accumulation_steps" in profile:
        config.gradient_accumulation_steps = profile["gradient_accumulation_steps"]
    if "max_seq_length" in profile:
        config.max_seq_length = profile["max_seq_length"]
    if "load_in_4bit" in profile:
        config.load_in_4bit = bool(profile["load_in_4bit"])
        config.load_in_16bit = not config.load_in_4bit

    return config
