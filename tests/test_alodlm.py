"""Tests for Amazon ALoDLM-8B training script."""

import argparse
import sys
from unittest.mock import patch

from lib.dataset import format_chatml_prompts
from train_alodlm import (
    ALODLM_TARGET_MODULES,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    HARDWARE_PROFILES,
    create_alodlm_config,
    main,
    train_alodlm,
)


def test_alodlm_constants_and_target_modules() -> None:
    """Verify default model, output directory, and LoRA target projections."""
    assert DEFAULT_MODEL_NAME == "amazon/ALoDLM-8B"
    assert DEFAULT_OUTPUT_DIR == "models/alodlm_8b_rust_lora"
    expected_targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert ALODLM_TARGET_MODULES == expected_targets
    assert set(expected_targets).issubset(set(ALODLM_TARGET_MODULES))


def test_alodlm_hardware_profiles_present() -> None:
    """Verify standard hardware profiles are defined with required attributes."""
    expected_profiles = [
        "mac_48gb",
        "mac_fast",
        "rtx4070",
        "rtx3090_4090",
        "rtx6000",
        "a100",
        "dev",
    ]
    for profile in expected_profiles:
        assert profile in HARDWARE_PROFILES
        spec = HARDWARE_PROFILES[profile]
        assert "batch_size" in spec
        assert "gradient_accumulation_steps" in spec
        assert "max_seq_length" in spec
        assert "load_in_4bit" in spec
        assert "description" in spec


def test_create_alodlm_config_defaults() -> None:
    """Verify create_alodlm_config initializes with appropriate model attributes."""
    args = argparse.Namespace(
        model_path=None,
        output_dir=None,
        data_file=None,
        hardware_profile=None,
        batch_size=None,
        gradient_accumulation_steps=None,
        max_seq_length=None,
        num_epochs=3,
        learning_rate=2e-4,
        load_in_4bit=False,
        checkpoint_dir=None,
        save_steps=50,
        save_total_limit=3,
        save_strategy="steps",
        resume=False,
        resume_from_checkpoint=None,
        offload_folder=None,
        trust_remote_code=False,
        wandb_project=None,
        wandb_entity=None,
    )
    cfg = create_alodlm_config(args)
    assert cfg.model_path == DEFAULT_MODEL_NAME
    assert cfg.model_family == "ALoDLM"
    assert cfg.variant == "8B"
    assert cfg.output_dir == DEFAULT_OUTPUT_DIR
    assert cfg.target_modules == ALODLM_TARGET_MODULES
    assert cfg.checkpoint_dir == f"{DEFAULT_OUTPUT_DIR}/checkpoints"


def test_create_alodlm_config_with_hardware_profile() -> None:
    """Verify hardware profile parameter application."""
    args = argparse.Namespace(
        model_path=None,
        output_dir=None,
        data_file=None,
        hardware_profile="rtx6000",
        batch_size=None,
        gradient_accumulation_steps=None,
        max_seq_length=None,
        num_epochs=3,
        learning_rate=2e-4,
        load_in_4bit=False,
        checkpoint_dir=None,
        save_steps=50,
        save_total_limit=3,
        save_strategy="steps",
        resume=False,
        resume_from_checkpoint=None,
        offload_folder=None,
        trust_remote_code=False,
        wandb_project=None,
        wandb_entity=None,
    )
    cfg = create_alodlm_config(args)
    assert cfg.batch_size == 2
    assert cfg.gradient_accumulation_steps == 4
    assert cfg.max_seq_length == 4096
    assert cfg.load_in_4bit is False


def test_create_alodlm_config_cli_overrides() -> None:
    """Verify CLI arguments take precedence over defaults and profiles."""
    args = argparse.Namespace(
        model_path="custom/local-alodlm",
        output_dir="models/custom_alodlm",
        data_file="data/custom.jsonl",
        hardware_profile="dev",
        batch_size=4,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        num_epochs=5,
        learning_rate=1e-4,
        load_in_4bit=True,
        checkpoint_dir="models/custom_alodlm/ckpts",
        save_steps=10,
        save_total_limit=2,
        save_strategy="epoch",
        resume=True,
        resume_from_checkpoint="models/custom_alodlm/ckpts/checkpoint-10",
        offload_folder="/tmp/offload",
        trust_remote_code=True,
        wandb_project="alodlm-eval",
        wandb_entity="rust-team",
    )
    cfg = create_alodlm_config(args)
    assert cfg.model_path == "custom/local-alodlm"
    assert cfg.output_dir == "models/custom_alodlm"
    assert cfg.data_file == "data/custom.jsonl"
    assert cfg.batch_size == 4
    assert cfg.gradient_accumulation_steps == 2
    assert cfg.max_seq_length == 1024
    assert cfg.num_epochs == 5
    assert cfg.learning_rate == 1e-4
    assert cfg.load_in_4bit is True
    assert cfg.checkpoint_dir == "models/custom_alodlm/ckpts"
    assert cfg.save_steps == 10
    assert cfg.save_total_limit == 2
    assert cfg.save_strategy == "epoch"
    assert cfg.resume is True
    assert cfg.resume_from_checkpoint == "models/custom_alodlm/ckpts/checkpoint-10"
    assert cfg.offload_folder == "/tmp/offload"
    assert cfg.trust_remote_code is True
    assert cfg.wandb_project == "alodlm-eval"
    assert cfg.wandb_entity == "rust-team"


def test_train_alodlm_invokes_pipeline() -> None:
    """Verify train_alodlm configures model metadata and invokes execute_training_pipeline."""
    args = argparse.Namespace(
        model_path=None,
        output_dir=None,
        data_file=None,
        hardware_profile="dev",
        batch_size=None,
        gradient_accumulation_steps=None,
        max_seq_length=None,
        num_epochs=3,
        learning_rate=2e-4,
        load_in_4bit=False,
        checkpoint_dir=None,
        save_steps=50,
        save_total_limit=3,
        save_strategy="steps",
        resume=False,
        resume_from_checkpoint=None,
        offload_folder=None,
        trust_remote_code=False,
        wandb_project=None,
        wandb_entity=None,
    )
    cfg = create_alodlm_config(args)

    with patch("train_alodlm.execute_training_pipeline") as mock_pipeline:
        train_alodlm(cfg)
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs["config"] is cfg
        assert call_kwargs["prompt_formatter_fn"] == format_chatml_prompts
        assert call_kwargs["allow_mps"] is True
        assert call_kwargs["min_vram_gb"] >= 12.0
        assert cfg.model_family == "ALoDLM"
        assert cfg.variant == "8B"


def test_main_cli_execution() -> None:
    """Verify main() parses arguments and delegates execution to run_training_cli."""
    test_args = ["train_alodlm.py", "--dry-run", "--hardware-profile", "dev"]
    with patch.object(sys, "argv", test_args):
        with patch("train_alodlm.run_training_cli") as mock_cli:
            main()
            mock_cli.assert_called_once()
            call_kwargs = mock_cli.call_args[1]
            assert call_kwargs["train_fn"] == train_alodlm
            assert call_kwargs["allow_mps"] is True
            assert call_kwargs["args"].dry_run is True
            assert call_kwargs["args"].hardware_profile == "dev"
