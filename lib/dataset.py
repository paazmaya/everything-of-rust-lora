"""
Dataset loading and prompt formatting utilities.
"""

import logging
from collections.abc import Callable, Sized
from typing import Any

from datasets import load_dataset

_logger = logging.getLogger("rust_lora.dataset")


def load_and_prepare_dataset(
    data_file: str,
    prompt_formatter_fn: Callable[[dict[str, list[str]]], dict[str, list[str]]],
    tokenizer: Any | None = None,
    max_seq_length: int | None = None,
    logger: logging.Logger | None = None,
) -> Any:
    """Load JSONL dataset and format records using the model-specific prompt function."""
    log = logger or _logger
    log.info(f"Loading training dataset from: {data_file}")
    dataset = load_dataset("json", data_files=data_file, split="train")

    if isinstance(dataset, Sized):
        log.info(f"Loaded {len(dataset)} training examples")

    log.info("Formatting dataset with prompt template...")
    formatted_dataset = dataset.map(prompt_formatter_fn, batched=True)

    if tokenizer is not None and len(formatted_dataset) > 0:
        sample_text = formatted_dataset[0]["text"]
        sample_tokens = len(tokenizer.encode(sample_text))
        limit_msg = f" (max limit: {max_seq_length})" if max_seq_length else ""
        log.info(f"Sample token length: {sample_tokens} tokens{limit_msg}")

    return formatted_dataset
