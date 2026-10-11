"""
Dataset loading and prompt formatting utilities.
"""

import logging
from collections.abc import Callable, Sized
from typing import Any

from datasets import load_dataset

_logger = logging.getLogger("rust_lora.dataset")

DEFAULT_RUST_SYSTEM_PROMPT = (
    "You are an expert Rust programmer, specializing in systems programming, "
    "async Rust, ESP32 embedded development, and popular Rust libraries."
)

CHATML_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "{system_prompt}<|im_end|>\n"
    "<|im_start|>user\n"
    "{user_message}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "{assistant_message}<|im_end|>"
)

ALPACA_PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def format_chatml_prompts(
    examples: dict[str, list[str]],
    system_prompt: str = DEFAULT_RUST_SYSTEM_PROMPT,
) -> dict[str, list[str]]:
    """Format instruction examples into standard ChatML format."""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts: list[str] = []
    for instruction, input_text, output in zip(instructions, inputs, outputs, strict=False):
        user_msg = f"{instruction}\n{input_text}" if input_text else instruction
        text = CHATML_PROMPT_TEMPLATE.format(
            system_prompt=system_prompt,
            user_message=user_msg,
            assistant_message=output,
        )
        texts.append(text)
    return {"text": texts}


def format_alpaca_prompts(examples: dict[str, list[str]]) -> dict[str, list[str]]:
    """Format instruction examples into Alpaca prompt format."""
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts: list[str] = []
    for instruction, input_text, output in zip(instructions, inputs, outputs, strict=False):
        text = ALPACA_PROMPT_TEMPLATE.format(instruction, input_text, output)
        texts.append(text)
    return {"text": texts}


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
