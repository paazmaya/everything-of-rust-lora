"""Tests for lib.config module."""

import logging
from typing import Any

from lib.config import (
    DEFAULT_REQUIRED_FIELDS,
    DEFAULT_TARGET_MODULES,
    DEFAULT_TRAIN_FILE,
    TrainingConfig,
    apply_hardware_profile,
)


def test_training_config_defaults() -> None:
    """Verify default values and post-init behavior of TrainingConfig."""
    cfg = TrainingConfig(model_path="test/model")
    assert cfg.model_path == "test/model"
    assert cfg.model_family == "Unknown"
    assert cfg.variant == ""
    assert cfg.trust_remote_code is False
    assert cfg.max_seq_length == 1024
    assert cfg.batch_size == 1
    assert cfg.gradient_accumulation_steps == 4
    assert cfg.num_epochs == 3
    assert cfg.learning_rate == 2e-4
    assert cfg.warmup_steps == 10
    assert cfg.weight_decay == 0.01
    assert cfg.lr_scheduler_type == "cosine"
    assert cfg.logging_steps == 10
    assert cfg.seed == 3407
    assert cfg.load_in_4bit is False
    assert cfg.load_in_16bit is True
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 16
    assert cfg.lora_dropout == 0.0
    assert cfg.lora_bias == "none"
    assert cfg.target_modules == list(DEFAULT_TARGET_MODULES)
    assert cfg.data_file == DEFAULT_TRAIN_FILE
    assert cfg.output_dir == "models/rust_lora"
    assert cfg.checkpoint_dir == "models/rust_lora/checkpoints"
    assert cfg.save_strategy == "steps"
    assert cfg.save_steps == 50
    assert cfg.save_total_limit == 3
    assert cfg.resume is False
    assert cfg.resume_from_checkpoint is None
    assert cfg.dataset_num_proc == 2
    assert cfg.offload_folder is None
    assert cfg.gradient_checkpointing is False
    assert cfg.device == "auto"
    assert cfg.wandb_project is None
    assert cfg.wandb_entity is None


def test_training_config_custom_checkpoint_dir() -> None:
    """Verify explicit checkpoint_dir is preserved."""
    cfg = TrainingConfig(
        model_path="test/model",
        output_dir="custom_out",
        checkpoint_dir="custom_checkpoints",
    )
    assert cfg.output_dir == "custom_out"
    assert cfg.checkpoint_dir == "custom_checkpoints"


def test_apply_hardware_profile_none() -> None:
    """Verify apply_hardware_profile returns config unchanged if profile_name is None or empty."""
    cfg = TrainingConfig(model_path="test/model")
    profiles: dict[str, dict[str, Any]] = {"profile1": {"batch_size": 2}}

    res1 = apply_hardware_profile(cfg, profiles, None)
    assert res1 is cfg
    assert res1.batch_size == 1

    res2 = apply_hardware_profile(cfg, profiles, "")
    assert res2 is cfg
    assert res2.batch_size == 1


def test_apply_hardware_profile_not_found(caplog: Any) -> None:
    """Verify behavior when profile_name is not in profiles dictionary."""
    cfg = TrainingConfig(model_path="test/model")
    profiles: dict[str, dict[str, Any]] = {"profile1": {"batch_size": 2}}
    logger = logging.getLogger("test_config")

    with caplog.at_level(logging.WARNING):
        res = apply_hardware_profile(cfg, profiles, "nonexistent", logger=logger)
    assert res is cfg
    assert "Hardware profile 'nonexistent' not found" in caplog.text

    # Without logger
    res_no_logger = apply_hardware_profile(cfg, profiles, "nonexistent", logger=None)
    assert res_no_logger is cfg


def test_apply_hardware_profile_success(caplog: Any) -> None:
    """Verify applying a valid hardware profile updates direct keys and flags."""
    cfg = TrainingConfig(model_path="test/model")
    profiles = {
        "vram_low": {
            "description": "Low VRAM profile",
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 512,
            "offload_folder": "offload_dir",
            "save_steps": 100,
            "save_total_limit": 5,
            "load_in_4bit": True,
            "gradient_checkpointing": True,
        }
    }
    logger = logging.getLogger("test_config")
    with caplog.at_level(logging.INFO):
        res = apply_hardware_profile(cfg, profiles, "vram_low", logger=logger)

    assert res.batch_size == 2
    assert res.gradient_accumulation_steps == 8
    assert res.max_seq_length == 512
    assert res.offload_folder == "offload_dir"
    assert res.save_steps == 100
    assert res.save_total_limit == 5
    assert res.load_in_4bit is True
    assert res.load_in_16bit is False
    assert res.gradient_checkpointing is True
    assert "Applying hardware profile 'vram_low': Low VRAM profile" in caplog.text


def test_apply_hardware_profile_no_description_and_4bit_false() -> None:
    """Verify profile without description and load_in_4bit False."""
    cfg = TrainingConfig(model_path="test/model", load_in_4bit=True, load_in_16bit=False)
    profiles = {
        "cpu_profile": {
            "load_in_4bit": False,
            "gradient_checkpointing": False,
        }
    }
    logger = logging.getLogger("test_config")
    res = apply_hardware_profile(cfg, profiles, "cpu_profile", logger=logger)
    assert res.load_in_4bit is False
    assert res.load_in_16bit is True
    assert res.gradient_checkpointing is False


def test_constants() -> None:
    """Verify module constants."""
    assert DEFAULT_TRAIN_FILE == "data/datasets/train.jsonl"
    assert DEFAULT_REQUIRED_FIELDS == ["instruction", "input", "output"]
    assert "q_proj" in DEFAULT_TARGET_MODULES
