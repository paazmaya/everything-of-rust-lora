"""
Shared library for model fine-tuning across IBM Granite, IFM K2-Horizon, NVIDIA Nemotron, and Qwen.
"""

from .cli import apply_args_to_config, build_arg_parser, build_resume_epilog, run_training_cli
from .config import (
    DEFAULT_REQUIRED_FIELDS,
    DEFAULT_TARGET_MODULES,
    DEFAULT_TRAIN_FILE,
    TrainingConfig,
    apply_hardware_profile,
)
from .dataset import (
    DEFAULT_RUST_SYSTEM_PROMPT,
    format_alpaca_prompts,
    format_chatml_prompts,
    load_and_prepare_dataset,
)
from .engine import (
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
    "DEFAULT_RUST_SYSTEM_PROMPT",
    "DEFAULT_TARGET_MODULES",
    "DEFAULT_TRAIN_FILE",
    "TrainingConfig",
    "apply_args_to_config",
    "apply_hardware_profile",
    "build_arg_parser",
    "build_resume_epilog",
    "check_cuda_available",
    "check_dependencies",
    "check_gpu_memory",
    "check_hardware_acceleration",
    "check_system_memory",
    "create_output_directories",
    "execute_training_pipeline",
    "format_alpaca_prompts",
    "format_chatml_prompts",
    "get_latest_checkpoint",
    "get_optimal_dtype",
    "load_and_prepare_dataset",
    "load_model_and_tokenizer",
    "resolve_checkpoint_resumption",
    "run_preflight_checks",
    "run_sft_training",
    "run_training_cli",
    "save_training_metadata",
    "setup_logging",
    "validate_data_file",
]
