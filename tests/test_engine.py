"""Tests for lib.engine module."""

import json
import logging
import os
import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import torch

from lib.config import TrainingConfig
from lib.engine import (
    _extract_checkpoint_step,
    _find_latest_checkpoint,
    build_training_arguments,
    create_output_directories,
    execute_training_pipeline,
    get_latest_checkpoint,
    get_optimal_dtype,
    load_model_and_tokenizer,
    resolve_checkpoint_resumption,
    run_sft_training,
    save_training_metadata,
    setup_logging,
)


def test_setup_logging() -> None:
    """Verify setup_logging sets log level based on verbose flag."""
    logger_info = setup_logging(name="test_engine_info", verbose=False)
    assert logger_info.name == "test_engine_info"

    logger_debug = setup_logging(name="test_engine_debug", verbose=True)
    assert logger_debug.name == "test_engine_debug"


def test_get_optimal_dtype() -> None:
    """Verify optimal dtype resolution across CUDA bf16, MPS, CUDA fp16, and CPU."""
    # 1. CUDA bf16
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.is_bf16_supported", return_value=True):
            assert get_optimal_dtype() == torch.bfloat16

    # 2. MPS available
    with patch("torch.cuda.is_available", return_value=False):
        mock_mps = MagicMock()
        mock_mps.is_available.return_value = True
        with patch.object(torch.backends, "mps", mock_mps, create=True):
            assert get_optimal_dtype() == torch.bfloat16

    # 3. CUDA without bf16 (fp16)
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.is_bf16_supported", return_value=False):
            mock_mps = MagicMock()
            mock_mps.is_available.return_value = False
            with patch.object(torch.backends, "mps", mock_mps, create=True):
                assert get_optimal_dtype() == torch.float16

    # 4. CPU (fp32)
    with patch("torch.cuda.is_available", return_value=False):
        mock_mps = MagicMock()
        mock_mps.is_available.return_value = False
        with patch.object(torch.backends, "mps", mock_mps, create=True):
            assert get_optimal_dtype() == torch.float32


def test_extract_checkpoint_step(tmp_path: Any) -> None:
    """Verify checkpoint step extraction and validation criteria."""
    # Directory without checkpoint- prefix
    sub_other = tmp_path / "other-dir"
    sub_other.mkdir()
    entry_other = next(os.scandir(str(tmp_path)))
    assert _extract_checkpoint_step(entry_other, require_valid=True) is None

    # Directory with checkpoint- prefix but not a number
    sub_bad = tmp_path / "checkpoint-abc"
    sub_bad.mkdir()
    for entry in os.scandir(str(tmp_path)):
        if entry.name == "checkpoint-abc":
            assert _extract_checkpoint_step(entry, require_valid=False) is None

    # File starting with checkpoint-
    file_entry = tmp_path / "checkpoint-50"
    file_entry.write_text("data")
    for entry in os.scandir(str(tmp_path)):
        if entry.name == "checkpoint-50" and not entry.is_dir():
            assert _extract_checkpoint_step(entry, require_valid=False) is None

    # Directory with checkpoint-100 without trainer_state.json (require_valid=True)
    sub_100 = tmp_path / "checkpoint-100"
    sub_100.mkdir()
    for entry in os.scandir(str(tmp_path)):
        if entry.name == "checkpoint-100":
            assert _extract_checkpoint_step(entry, require_valid=True) is None
            assert _extract_checkpoint_step(entry, require_valid=False) == 100

    # Directory with checkpoint-100 with trainer_state.json
    (sub_100 / "trainer_state.json").write_text("{}")
    for entry in os.scandir(str(tmp_path)):
        if entry.name == "checkpoint-100":
            assert _extract_checkpoint_step(entry, require_valid=True) == 100


def test_get_latest_checkpoint(tmp_path: Any) -> None:
    """Verify finding latest checkpoint in directory."""
    # Non-existent directory
    assert get_latest_checkpoint(str(tmp_path / "missing")) is None

    # Empty directory
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    assert get_latest_checkpoint(str(ckpt_dir)) is None

    # Multiple valid checkpoints
    c50 = ckpt_dir / "checkpoint-50"
    c50.mkdir()
    (c50 / "trainer_state.json").write_text("{}")

    c150 = ckpt_dir / "checkpoint-150"
    c150.mkdir()
    (c150 / "trainer_state.json").write_text("{}")

    c100 = ckpt_dir / "checkpoint-100"
    c100.mkdir()
    (c100 / "trainer_state.json").write_text("{}")

    latest = get_latest_checkpoint(str(ckpt_dir))
    assert latest == str(c150)


def test_find_latest_checkpoint(tmp_path: Any) -> None:
    """Verify fallback directory searching when primary directory is empty."""
    primary = tmp_path / "primary"
    primary.mkdir()
    fallback = tmp_path / "fallback"
    fallback.mkdir()

    # None anywhere
    assert _find_latest_checkpoint(str(primary), str(fallback)) is None

    # Found in fallback
    fb_ckpt = fallback / "checkpoint-200"
    fb_ckpt.mkdir()
    (fb_ckpt / "trainer_state.json").write_text("{}")
    assert _find_latest_checkpoint(str(primary), str(fallback)) == str(fb_ckpt)


def test_create_output_directories(tmp_path: Any) -> None:
    """Verify create_output_directories creates directories idempotently."""
    out_dir = str(tmp_path / "out")
    ckpt_dir = str(tmp_path / "out" / "checkpoints")
    logger = logging.getLogger("test_engine")

    create_output_directories(out_dir, ckpt_dir, logger=logger)
    assert os.path.isdir(out_dir)
    assert os.path.isdir(ckpt_dir)

    # Calling again without logger
    create_output_directories(out_dir, ckpt_dir, logger=None)
    assert os.path.isdir(out_dir)


def test_save_training_metadata(tmp_path: Any) -> None:
    """Verify save_training_metadata writes proper JSON manifest."""
    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir, exist_ok=True)
    cfg = TrainingConfig(model_path="test/model", model_family="Nemotron", variant="nano")
    logger = logging.getLogger("test_engine")

    manifest_path = save_training_metadata(
        out_dir, cfg, extra_metadata={"custom_metric": 42}, logger=logger
    )
    assert os.path.isfile(manifest_path)

    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["model_path"] == "test/model"
    assert data["model_family"] == "Nemotron"
    assert data["variant"] == "nano"
    assert data["custom_metric"] == 42


def test_resolve_checkpoint_resumption(tmp_path: Any) -> None:
    """Verify resolve_checkpoint_resumption handles explicit path, latest, and missing."""
    logger = logging.getLogger("test_engine")
    ckpt_dir = str(tmp_path / "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Neither resume nor resume_from_checkpoint
    cfg_none = TrainingConfig(model_path="test/model", resume=False, resume_from_checkpoint=None)
    assert resolve_checkpoint_resumption(cfg_none, ckpt_dir, logger) is None

    # Explicit nonexistent directory
    cfg_missing = TrainingConfig(
        model_path="test/model", resume_from_checkpoint=str(tmp_path / "nonexistent")
    )
    assert resolve_checkpoint_resumption(cfg_missing, ckpt_dir, logger) is None

    # Explicit directory without trainer_state.json
    spec_dir = tmp_path / "specific_ckpt"
    spec_dir.mkdir()
    cfg_invalid_spec = TrainingConfig(model_path="test/model", resume_from_checkpoint=str(spec_dir))
    assert resolve_checkpoint_resumption(cfg_invalid_spec, ckpt_dir, logger) == str(spec_dir)

    # Auto-resume ("latest" or resume=True) with checkpoint found
    c25 = tmp_path / "ckpts" / "checkpoint-25"
    c25.mkdir()
    (c25 / "trainer_state.json").write_text("{}")
    cfg_auto = TrainingConfig(model_path="test/model", resume=True)
    assert resolve_checkpoint_resumption(cfg_auto, ckpt_dir, logger) == str(c25)

    # Auto-resume ("latest") when no checkpoint found
    empty_dir = str(tmp_path / "empty_ckpts")
    os.makedirs(empty_dir, exist_ok=True)
    cfg_latest_empty = TrainingConfig(model_path="test/model", resume_from_checkpoint="latest")
    assert resolve_checkpoint_resumption(cfg_latest_empty, empty_dir, logger) is None


def test_build_training_arguments() -> None:
    """Verify build_training_arguments dynamically selects optimizers and precision."""
    cfg = TrainingConfig(
        model_path="test/model",
        save_strategy="epoch",
        gradient_checkpointing=True,
    )

    # CUDA available with bf16
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.is_bf16_supported", return_value=True):
            args = build_training_arguments(cfg, checkpoint_dir="test_ckpt", report_to=["wandb"])
            assert args.bf16 is True
            assert args.fp16 is False
            assert args.optim == "adamw_8bit"
            assert args.save_strategy == "epoch"
            assert args.save_steps == 500
            assert args.gradient_checkpointing is True
            assert args.report_to == ["wandb"]

    # CPU environment
    with patch("torch.cuda.is_available", return_value=False):
        mock_mps = MagicMock()
        mock_mps.is_available.return_value = False
        with patch.object(torch.backends, "mps", mock_mps, create=True):
            cfg_steps = TrainingConfig(
                model_path="test/model", save_strategy="steps", save_steps=75
            )
            args_cpu = build_training_arguments(cfg_steps, checkpoint_dir="test_ckpt", report_to=[])
            assert args_cpu.bf16 is False
            assert args_cpu.fp16 is False
            assert args_cpu.optim == "adamw_torch"
            assert args_cpu.save_steps == 75


def test_load_model_and_tokenizer_unsloth_success() -> None:
    """Verify load_model_and_tokenizer success with Unsloth."""
    cfg = TrainingConfig(model_path="test/model", lora_r=32, lora_alpha=64)
    logger = logging.getLogger("test_engine")

    mock_model = MagicMock()
    mock_peft_model = MagicMock()
    mock_tokenizer = MagicMock()

    mock_flm = MagicMock()
    mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
    mock_flm.get_peft_model.return_value = mock_peft_model

    mock_unsloth = ModuleType("unsloth")
    mock_unsloth.FastLanguageModel = mock_flm  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"unsloth": mock_unsloth}):
        model, tokenizer = load_model_and_tokenizer(cfg, logger=logger)
        assert model is mock_peft_model
        assert tokenizer is mock_tokenizer
        mock_flm.from_pretrained.assert_called_once()
        mock_flm.get_peft_model.assert_called_once()


def test_load_model_and_tokenizer_fallback_transformers_peft(tmp_path: Any) -> None:
    """Verify fallback to AutoModelForCausalLM + PEFT when Unsloth fails or is absent."""
    offload_dir = str(tmp_path / "offload")
    cfg = TrainingConfig(
        model_path="test/model",
        load_in_4bit=True,
        offload_folder=offload_dir,
        gradient_checkpointing=True,
    )
    logger = logging.getLogger("test_engine")

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos>"

    mock_base_model = MagicMock()
    mock_peft_model = MagicMock()

    mock_unsloth = ModuleType("unsloth")
    mock_flm = MagicMock()
    mock_flm.from_pretrained.side_effect = RuntimeError("Unsloth unsupported arch")
    mock_unsloth.FastLanguageModel = mock_flm  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"unsloth": mock_unsloth}):
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
            with patch(
                "transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_base_model
            ) as mock_causal:
                with patch("peft.get_peft_model", return_value=mock_peft_model):
                    model, tokenizer = load_model_and_tokenizer(cfg, logger=logger)
                    assert model is mock_peft_model
                    assert tokenizer is mock_tokenizer
                    assert tokenizer.pad_token == "<eos>"
                    assert os.path.isdir(offload_dir)
                    mock_base_model.gradient_checkpointing_enable.assert_called_once()
                    mock_causal.assert_called_once()


def test_run_sft_training(tmp_path: Any) -> None:
    """Verify run_sft_training manages output dirs, wandb env, trainer loop, and model saving."""
    out_dir = str(tmp_path / "out")
    cfg = TrainingConfig(
        model_path="test/model",
        output_dir=out_dir,
        wandb_project="test_proj",
        wandb_entity="test_ent",
        resume=True,
    )
    logger = logging.getLogger("test_engine")

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_dataset = [{"text": "Sample text"}]

    mock_trainer = MagicMock()
    stats = MagicMock()
    stats.training_loss = 0.42
    mock_trainer.train.return_value = stats

    with patch("lib.engine.resolve_checkpoint_resumption", return_value="test/ckpt-10"):
        with patch("lib.engine.SFTTrainer", return_value=mock_trainer):
            run_sft_training(mock_model, mock_tokenizer, mock_dataset, cfg, logger=logger)

            assert os.environ.get("WANDB_PROJECT") == "test_proj"
            assert os.environ.get("WANDB_ENTITY") == "test_ent"
            mock_trainer.train.assert_called_once_with(resume_from_checkpoint="test/ckpt-10")
            mock_model.save_pretrained.assert_called_once_with(out_dir)
            mock_tokenizer.save_pretrained.assert_called_once_with(out_dir)
            assert os.path.isfile(os.path.join(out_dir, "manifest.json"))


def test_run_sft_training_no_resume(tmp_path: Any) -> None:
    """Verify run_sft_training without resumption."""
    out_dir = str(tmp_path / "out_clean")
    cfg = TrainingConfig(model_path="test/model", output_dir=out_dir)
    logger = logging.getLogger("test_engine")

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_dataset = [{"text": "Sample text"}]

    mock_trainer = MagicMock()
    mock_trainer.train.return_value = MagicMock(spec=[])

    with patch("lib.engine.resolve_checkpoint_resumption", return_value=None):
        with patch("lib.engine.SFTTrainer", return_value=mock_trainer):
            run_sft_training(mock_model, mock_tokenizer, mock_dataset, cfg, logger=logger)
            mock_trainer.train.assert_called_once_with()


def test_execute_training_pipeline() -> None:
    """Verify execute_training_pipeline runs all 4 steps in sequence."""
    cfg = TrainingConfig(model_path="test/model", model_family="Granite")
    logger = logging.getLogger("test_engine")

    with patch("lib.engine.run_preflight_checks", return_value=False) as mock_preflight:
        with patch(
            "lib.engine.load_model_and_tokenizer", return_value=("mock_model", "mock_tok")
        ) as mock_load:
            with patch(
                "lib.engine.load_and_prepare_dataset", return_value="mock_dataset"
            ) as mock_prep:
                with patch("lib.engine.run_sft_training") as mock_train:
                    execute_training_pipeline(
                        config=cfg,
                        prompt_formatter_fn=lambda x: x,
                        logger=logger,
                    )
                    mock_preflight.assert_called_once()
                    mock_load.assert_called_once()
                    mock_prep.assert_called_once()
                    mock_train.assert_called_once_with(
                        model="mock_model",
                        tokenizer="mock_tok",
                        dataset="mock_dataset",
                        config=cfg,
                        logger=logger,
                    )
