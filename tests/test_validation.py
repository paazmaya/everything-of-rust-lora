"""Tests for lib.validation module."""

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import torch

from lib.config import DEFAULT_REQUIRED_FIELDS, DEFAULT_TRAIN_FILE, TrainingConfig
from lib.validation import (
    check_cuda_available,
    check_dependencies,
    check_gpu_memory,
    check_hardware_acceleration,
    check_system_memory,
    run_preflight_checks,
    validate_data_file,
)


def test_check_hardware_acceleration_cuda(caplog: Any) -> None:
    """Verify check_hardware_acceleration when CUDA is available."""
    logger = logging.getLogger("test_validation")
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
            with caplog.at_level(logging.INFO):
                device, accelerated = check_hardware_acceleration(logger=logger)
            assert device == "cuda"
            assert accelerated is True
            assert "CUDA available. Active GPU: NVIDIA RTX 4090" in caplog.text


def test_check_hardware_acceleration_mps(caplog: Any) -> None:
    """Verify check_hardware_acceleration when MPS is available."""
    logger = logging.getLogger("test_validation")
    with patch("torch.cuda.is_available", return_value=False):
        mock_mps = MagicMock()
        mock_mps.is_available.return_value = True
        with patch.object(torch.backends, "mps", mock_mps, create=True):
            with caplog.at_level(logging.INFO):
                device, accelerated = check_hardware_acceleration(logger=logger)
            assert device == "mps"
            assert accelerated is True
            assert "Metal Performance Shaders (MPS)" in caplog.text


def test_check_hardware_acceleration_cpu(caplog: Any) -> None:
    """Verify check_hardware_acceleration fallback to CPU."""
    logger = logging.getLogger("test_validation")
    with patch("torch.cuda.is_available", return_value=False):
        mock_mps = MagicMock()
        mock_mps.is_available.return_value = False
        with patch.object(torch.backends, "mps", mock_mps, create=True):
            with caplog.at_level(logging.WARNING):
                device, accelerated = check_hardware_acceleration(logger=logger)
            assert device == "cpu"
            assert accelerated is False
            assert "Training will run on CPU" in caplog.text


def test_check_cuda_available() -> None:
    """Verify check_cuda_available across CUDA, MPS, and missing scenarios."""
    logger = logging.getLogger("test_validation")

    # Case 1: CUDA available
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.get_device_name", return_value="Tesla T4"):
            assert check_cuda_available(logger=logger) is True

    # Case 2: CUDA not available, MPS allowed and available
    with patch("torch.cuda.is_available", return_value=False):
        mock_mps = MagicMock()
        mock_mps.is_available.return_value = True
        with patch.object(torch.backends, "mps", mock_mps, create=True):
            assert check_cuda_available(logger=logger, allow_mps=True) is True

    # Case 3: CUDA not available, MPS available but allow_mps=False
    with patch("torch.cuda.is_available", return_value=False):
        mock_mps = MagicMock()
        mock_mps.is_available.return_value = True
        with patch.object(torch.backends, "mps", mock_mps, create=True):
            assert check_cuda_available(logger=logger, allow_mps=False) is False

    # Case 4: Neither available
    with patch("torch.cuda.is_available", return_value=False):
        mock_mps = MagicMock()
        mock_mps.is_available.return_value = False
        with patch.object(torch.backends, "mps", mock_mps, create=True):
            assert check_cuda_available(logger=logger, allow_mps=True) is False


def test_check_gpu_memory() -> None:
    """Verify check_gpu_memory with CUDA disabled, sufficient VRAM, and insufficient VRAM."""
    logger = logging.getLogger("test_validation")

    # CUDA disabled
    with patch("torch.cuda.is_available", return_value=False):
        assert check_gpu_memory(min_vram_gb=12.0, logger=logger) is False

    # Sufficient VRAM (24GB)
    with patch("torch.cuda.is_available", return_value=True):
        mock_props = MagicMock()
        mock_props.total_memory = 24e9
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            assert check_gpu_memory(min_vram_gb=12.0, logger=logger) is True

    # Insufficient VRAM (8GB vs 12GB required)
    with patch("torch.cuda.is_available", return_value=True):
        mock_props = MagicMock()
        mock_props.total_memory = 8e9
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            assert check_gpu_memory(min_vram_gb=12.0, logger=logger) is False


def test_check_system_memory() -> None:
    """Verify check_system_memory under CUDA, POSIX sysconf, low memory, and exception."""
    logger = logging.getLogger("test_validation")

    # CUDA path
    with patch("torch.cuda.is_available", return_value=True):
        with patch("lib.validation.check_gpu_memory", return_value=True) as mock_gpu:
            assert check_system_memory(min_ram_gb=16.0, logger=logger) is True
            mock_gpu.assert_called_once_with(min_vram_gb=16.0, logger=logger)

    # POSIX sysconf sufficient memory (32GB)
    with patch("torch.cuda.is_available", return_value=False):
        with patch(
            "os.sysconf", side_effect=lambda name: 8388608 if name == "SC_PHYS_PAGES" else 4096
        ):
            assert check_system_memory(min_ram_gb=16.0, logger=logger) is True

    # POSIX sysconf insufficient memory (8GB vs 16GB required)
    with patch("torch.cuda.is_available", return_value=False):
        with patch(
            "os.sysconf", side_effect=lambda name: 2097152 if name == "SC_PHYS_PAGES" else 4096
        ):
            assert check_system_memory(min_ram_gb=16.0, logger=logger) is False

    # Non-POSIX or sysconf exception
    with patch("torch.cuda.is_available", return_value=False):
        with patch("os.sysconf", side_effect=AttributeError("sysconf not available")):
            assert check_system_memory(min_ram_gb=16.0, logger=logger) is True


def test_check_dependencies() -> None:
    """Verify check_dependencies handling of present packages, missing packages, and mamba-ssm."""
    logger = logging.getLogger("test_validation")

    # All packages present
    assert check_dependencies(required_packages=["os", "sys"], logger=logger) is True

    # Missing package
    assert check_dependencies(required_packages=["nonexistent_package_xyz"], logger=logger) is False

    # Default package list check
    with patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: MagicMock()):
        assert check_dependencies(logger=logger) is True

    # trust_remote_code with mamba_ssm missing (logs warning but does not fail)
    def import_mock(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mamba_ssm":
            raise ImportError("No module named mamba_ssm")
        return MagicMock()

    with patch("builtins.__import__", side_effect=import_mock):
        assert (
            check_dependencies(required_packages=["os"], trust_remote_code=True, logger=logger)
            is True
        )


def test_validate_data_file(tmp_path: Any) -> None:
    """Verify validate_data_file checks file existence, emptyness, JSON validity, and schema fields."""
    logger = logging.getLogger("test_validation")

    # Missing file
    non_existent = str(tmp_path / "does_not_exist.jsonl")
    assert validate_data_file(non_existent, logger=logger) is False

    # Empty file
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    assert validate_data_file(str(empty_file), logger=logger) is False

    # Invalid JSON line
    invalid_file = tmp_path / "invalid.jsonl"
    invalid_file.write_text("not json at all\n", encoding="utf-8")
    assert validate_data_file(str(invalid_file), logger=logger) is False

    # Missing required field
    missing_fields_file = tmp_path / "missing_fields.jsonl"
    missing_fields_file.write_text('{"instruction": "hi", "output": "bye"}\n', encoding="utf-8")
    assert (
        validate_data_file(
            str(missing_fields_file),
            required_fields=DEFAULT_REQUIRED_FIELDS,
            logger=logger,
        )
        is False
    )

    # Valid file with default fields
    valid_file = tmp_path / "valid.jsonl"
    valid_file.write_text('{"instruction": "hi", "input": "", "output": "bye"}\n', encoding="utf-8")
    assert validate_data_file(str(valid_file), logger=logger) is True

    # Valid file using default data_path parameter
    assert DEFAULT_TRAIN_FILE == "data/datasets/train.jsonl"
    with patch("lib.validation.DEFAULT_TRAIN_FILE", str(valid_file)):
        assert validate_data_file(logger=logger) is True


def test_run_preflight_checks_success(tmp_path: Any) -> None:
    """Verify run_preflight_checks returns True when all checks pass."""
    valid_file = tmp_path / "valid.jsonl"
    valid_file.write_text('{"instruction": "hi", "input": "", "output": "bye"}\n', encoding="utf-8")

    cfg = TrainingConfig(model_path="test/model", data_file=str(valid_file))
    logger = logging.getLogger("test_validation")

    with patch("lib.validation.check_hardware_acceleration", return_value=("cuda", True)):
        with patch("lib.validation.check_system_memory", return_value=True):
            with patch("lib.validation.check_dependencies", return_value=True):
                passed = run_preflight_checks(cfg, min_vram_gb=12.0, logger=logger)
                assert passed is True


def test_run_preflight_checks_failure_and_cpu_branch(tmp_path: Any) -> None:
    """Verify run_preflight_checks when dependencies fail and CPU is used with allow_mps=False."""
    cfg = TrainingConfig(model_path="test/model", data_file="does_not_exist.jsonl")
    logger = logging.getLogger("test_validation")

    with patch("lib.validation.check_hardware_acceleration", return_value=("cpu", False)):
        with patch("lib.validation.check_dependencies", return_value=False):
            passed = run_preflight_checks(cfg, allow_mps=False, logger=logger)
            assert passed is False
