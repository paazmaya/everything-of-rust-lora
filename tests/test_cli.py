"""Tests for lib.cli module."""

import argparse
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lib.cli import (
    apply_args_to_config,
    build_arg_parser,
    build_resume_epilog,
    run_training_cli,
)
from lib.config import TrainingConfig


def test_build_resume_epilog() -> None:
    """Verify build_resume_epilog output with and without extra examples."""
    epilog_plain = build_resume_epilog("train_custom.py")
    assert "uv run python train_custom.py --dry-run" in epilog_plain
    assert "uv run python train_custom.py --resume" in epilog_plain

    epilog_extra = build_resume_epilog("train_custom.py", extra_examples="# Extra comment\n")
    assert "# Extra comment" in epilog_extra


def test_build_arg_parser_defaults() -> None:
    """Verify build_arg_parser constructs parser with proper default values."""
    parser = build_arg_parser(
        description="Test Trainer",
        default_model="base/model",
        default_output_dir="models/test_out",
    )
    args = parser.parse_args([])
    assert args.model_path == "base/model"
    assert args.output_dir == "models/test_out"
    assert args.dry_run is False
    assert args.hardware_profile is None
    assert args.batch_size is None
    assert args.gradient_accumulation_steps is None
    assert args.max_seq_length is None
    assert args.num_epochs == 3
    assert args.learning_rate == 2e-4
    assert args.load_in_4bit is False
    assert args.resume is False
    assert args.resume_from_checkpoint is None
    assert args.trust_remote_code is False


def test_build_arg_parser_with_profiles() -> None:
    """Verify parser generation with hardware profile choices and custom defaults."""
    profiles = {"rtx3090": {"batch_size": 2}, "rtx4090": {"batch_size": 4}}
    parser = build_arg_parser(
        description="Profile Trainer",
        default_model="base/model",
        default_output_dir="models/test_out",
        hardware_profiles=profiles,
        default_profile="rtx3090",
        default_max_seq_length=2048,
        default_batch_size=2,
        default_grad_accum=8,
        default_num_epochs=5,
        default_learning_rate=1e-4,
        default_load_in_4bit=True,
    )
    args = parser.parse_args([])
    assert args.hardware_profile == "rtx3090"
    assert args.num_epochs == 5
    assert args.learning_rate == 1e-4
    assert args.load_in_4bit is True


def test_apply_args_to_config_basic() -> None:
    """Verify apply_args_to_config propagates parsed args to TrainingConfig."""
    cfg = TrainingConfig(model_path="initial/model", output_dir="initial_out")
    args = argparse.Namespace(
        model_path="updated/model",
        batch_size=4,
        gradient_accumulation_steps=2,
        max_seq_length=512,
        num_epochs=10,
        learning_rate=3e-4,
        data_file="custom_data.jsonl",
        output_dir="new_out",
        checkpoint_dir="new_out/custom_ckpt",
        save_steps=25,
        save_total_limit=2,
        save_strategy="epoch",
        resume_from_checkpoint="new_out/custom_ckpt/checkpoint-25",
        offload_folder="offload_path",
        wandb_project="my_project",
        wandb_entity="my_team",
        hardware_profile=None,
        resume=True,
        load_in_4bit=True,
        trust_remote_code=True,
    )

    res = apply_args_to_config(cfg, args)
    assert res.model_path == "updated/model"
    assert res.batch_size == 4
    assert res.gradient_accumulation_steps == 2
    assert res.max_seq_length == 512
    assert res.num_epochs == 10
    assert res.learning_rate == 3e-4
    assert res.data_file == "custom_data.jsonl"
    assert res.output_dir == "new_out"
    assert res.checkpoint_dir == "new_out/custom_ckpt"
    assert res.save_steps == 25
    assert res.save_total_limit == 2
    assert res.save_strategy == "epoch"
    assert res.resume_from_checkpoint == "new_out/custom_ckpt/checkpoint-25"
    assert res.offload_folder == "offload_path"
    assert res.wandb_project == "my_project"
    assert res.wandb_entity == "my_team"
    assert res.resume is True
    assert res.load_in_4bit is True
    assert res.load_in_16bit is False
    assert res.trust_remote_code is True


def test_apply_args_to_config_checkpoint_dir_defaulting() -> None:
    """Verify checkpoint_dir defaults to {output_dir}/checkpoints when None."""
    cfg = TrainingConfig(model_path="initial/model", output_dir="new_out")
    args = argparse.Namespace(
        output_dir="custom_output",
        checkpoint_dir=None,
        hardware_profile=None,
        resume=False,
        load_in_4bit=False,
        trust_remote_code=False,
    )
    res = apply_args_to_config(cfg, args)
    assert res.checkpoint_dir == "custom_output/checkpoints"


def test_apply_args_to_config_with_hardware_profile() -> None:
    """Verify hardware profile application inside apply_args_to_config."""
    cfg = TrainingConfig(model_path="initial/model")
    profiles = {"fast": {"batch_size": 8, "gradient_accumulation_steps": 1}}
    args = argparse.Namespace(
        hardware_profile="fast",
        checkpoint_dir=None,
        resume=False,
        load_in_4bit=False,
        trust_remote_code=False,
    )
    res = apply_args_to_config(cfg, args, hardware_profiles=profiles)
    assert res.batch_size == 8
    assert res.gradient_accumulation_steps == 1


def test_run_training_cli_dry_run_success() -> None:
    """Verify run_training_cli exits with 0 on successful dry run."""
    cfg = TrainingConfig(model_path="test/model", model_family="Qwen 3.8")
    args = argparse.Namespace(dry_run=True)
    train_fn = MagicMock()

    with patch("lib.cli.run_preflight_checks", return_value=True) as mock_preflight:
        with pytest.raises(SystemExit) as exc_info:
            run_training_cli(cfg, args, train_fn)
        assert exc_info.value.code == 0
        mock_preflight.assert_called_once()
        train_fn.assert_not_called()


def test_run_training_cli_dry_run_failure() -> None:
    """Verify run_training_cli exits with 1 on failed dry run."""
    cfg = TrainingConfig(model_path="test/model")
    args = argparse.Namespace(dry_run=True)
    train_fn = MagicMock()

    with patch("lib.cli.run_preflight_checks", return_value=False):
        with pytest.raises(SystemExit) as exc_info:
            run_training_cli(cfg, args, train_fn)
        assert exc_info.value.code == 1
        train_fn.assert_not_called()


def test_run_training_cli_train_success() -> None:
    """Verify run_training_cli executes train_fn normally without dry-run."""
    cfg = TrainingConfig(model_path="test/model")
    args = argparse.Namespace(dry_run=False)
    train_fn = MagicMock()
    logger = logging.getLogger("test_cli")

    run_training_cli(cfg, args, train_fn, logger=logger)
    train_fn.assert_called_once_with(cfg)


def test_run_training_cli_keyboard_interrupt(caplog: Any) -> None:
    """Verify run_training_cli catches KeyboardInterrupt and exits with code 130."""
    cfg = TrainingConfig(model_path="test/model")
    args = argparse.Namespace(dry_run=False)
    train_fn = MagicMock(side_effect=KeyboardInterrupt)
    logger = logging.getLogger("test_cli")

    with caplog.at_level(logging.INFO):
        with pytest.raises(SystemExit) as exc_info:
            run_training_cli(cfg, args, train_fn, logger=logger)
        assert exc_info.value.code == 130
        assert "Training interrupted by user" in caplog.text


def test_run_training_cli_exception(caplog: Any) -> None:
    """Verify run_training_cli catches unexpected exceptions and exits with code 1."""
    cfg = TrainingConfig(model_path="test/model")
    args = argparse.Namespace(dry_run=False)
    train_fn = MagicMock(side_effect=RuntimeError("CUDA Out of memory"))
    logger = logging.getLogger("test_cli")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit) as exc_info:
            run_training_cli(cfg, args, train_fn, logger=logger)
        assert exc_info.value.code == 1
        assert "Training failed with unexpected exception" in caplog.text
