"""
Validation and environment preflight checks for LoRA training.
"""

import json
import logging
import os
from typing import Any

import torch

from .config import DEFAULT_REQUIRED_FIELDS, TrainingConfig

_logger = logging.getLogger("rust_lora.validation")


def check_cuda_available(logger: logging.Logger | None = None) -> bool:
    """Verify CUDA availability and report active GPU."""
    log = logger or _logger
    if not torch.cuda.is_available():
        log.error("CUDA is not available. GPU acceleration is required for training.")
        return False

    device_name = torch.cuda.get_device_name(0)
    log.info(f"CUDA available. Active GPU: {device_name}")
    return True


def check_gpu_memory(min_vram_gb: float = 12.0, logger: logging.Logger | None = None) -> bool:
    """Check if GPU has sufficient VRAM."""
    log = logger or _logger
    if not torch.cuda.is_available():
        return False

    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"GPU VRAM: {total_memory:.1f} GB")

    if total_memory < min_vram_gb:
        log.warning(
            f"GPU has {total_memory:.1f} GB, but at least {min_vram_gb:.1f} GB is recommended. "
            "Consider using 4-bit quantization (--load-in-4bit or a lower-resource hardware profile)."
        )
        return False

    return True


def check_dependencies(
    required_packages: list[str] | None = None,
    trust_remote_code: bool = False,
    logger: logging.Logger | None = None,
) -> bool:
    """Verify required packages are installed."""
    log = logger or _logger
    packages = required_packages or ["transformers", "trl", "peft"]
    missing: list[str] = []

    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if trust_remote_code:
        try:
            __import__("mamba_ssm")
        except ImportError:
            log.warning(
                "Model may require `mamba-ssm` package when trust_remote_code=True.\n"
                "Install with: uv pip install mamba-ssm"
            )

    if missing:
        log.error(f"Missing required dependencies: {', '.join(missing)}")
        log.error("Install with: uv sync or uv pip install <package>")
        return False

    log.info(f"All core dependencies available: {', '.join(packages)}")
    return True


def validate_data_file(
    data_path: str,
    required_fields: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> bool:
    """Validate that training data exists and conforms to JSONL schema."""
    log = logger or _logger
    if not os.path.exists(data_path):
        log.error(f"Training data not found at: {data_path}")
        log.error("Create training data in JSONL format with fields: instruction, input, output")
        return False

    log.info(f"Found training data at: {data_path}")
    fields = required_fields or DEFAULT_REQUIRED_FIELDS

    try:
        with open(data_path, encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                log.error("Training data file is empty")
                return False

            record: dict[str, Any] = json.loads(first_line)
            missing_fields = [fld for fld in fields if fld not in record]

            if missing_fields:
                log.error(f"Missing fields in data: {missing_fields}. Required fields: {fields}")
                return False

            log.info(f"Data schema valid (sample fields: {list(record.keys())})")
            return True

    except json.JSONDecodeError:
        log.exception("Failed to parse training data as JSONL")
        return False


def run_preflight_checks(
    config: TrainingConfig,
    min_vram_gb: float = 12.0,
    required_packages: list[str] | None = None,
    logger: logging.Logger | None = None,
) -> bool:
    """Run all preflight environment and dataset checks."""
    log = logger or _logger
    log.info("Running preflight validation checks...")

    cuda_ok = check_cuda_available(logger=log)
    if cuda_ok:
        check_gpu_memory(min_vram_gb=min_vram_gb, logger=log)

    deps_ok = check_dependencies(
        required_packages=required_packages,
        trust_remote_code=config.trust_remote_code,
        logger=log,
    )

    data_ok = validate_data_file(
        data_path=config.data_file,
        logger=log,
    )

    all_passed = bool(cuda_ok and deps_ok and data_ok)
    if all_passed:
        log.info("All preflight validation checks passed successfully!")
    else:
        log.warning("Preflight checks encountered issues (see warnings/errors above).")
    return all_passed
