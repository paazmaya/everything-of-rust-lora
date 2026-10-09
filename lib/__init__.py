"""
Shared library for model fine-tuning across IBM Granite, IFM K2-Horizon, NVIDIA Nemotron, and Qwen.
"""

from .cli import apply_args_to_config, build_arg_parser
from .config import (
    DEFAULT_REQUIRED_FIELDS,
    DEFAULT_TARGET_MODULES,
    DEFAULT_TRAIN_FILE,
    TrainingConfig,
    apply_hardware_profile,
)
from .dataset import load_and_prepare_dataset
from .engine import (
    create_output_directories,
    get_latest_checkpoint,
    get_optimal_dtype,
    load_model_and_tokenizer,
    run_sft_training,
    save_training_metadata,
    setup_logging,
)
from .validation import (
    check_cuda_available,
    check_dependencies,
    check_gpu_memory,
    check_hardware_acceleration,
    check_system_memory,
    run_preflight_checks,
    validate_data_file,
)

__all__ = [
    "DEFAULT_REQUIRED_FIELDS",
    "DEFAULT_TARGET_MODULES",
    "DEFAULT_TRAIN_FILE",
    "TrainingConfig",
    "apply_args_to_config",
    "apply_hardware_profile",
    "build_arg_parser",
    "check_cuda_available",
    "check_dependencies",
    "check_gpu_memory",
    "check_hardware_acceleration",
    "check_system_memory",
    "create_output_directories",
    "get_latest_checkpoint",
    "get_optimal_dtype",
    "load_and_prepare_dataset",
    "load_model_and_tokenizer",
    "run_preflight_checks",
    "run_sft_training",
    "save_training_metadata",
    "setup_logging",
    "validate_data_file",
]
