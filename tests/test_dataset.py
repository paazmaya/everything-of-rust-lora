"""Tests for lib.dataset module."""

import logging
from unittest.mock import MagicMock, patch

from lib.dataset import (
    ALPACA_PROMPT_TEMPLATE,
    CHATML_PROMPT_TEMPLATE,
    DEFAULT_RUST_SYSTEM_PROMPT,
    format_alpaca_prompts,
    format_chatml_prompts,
    load_and_prepare_dataset,
)


def test_format_chatml_prompts_with_and_without_input() -> None:
    """Verify format_chatml_prompts handles records with input and without input."""
    examples = {
        "instruction": ["Explain Rust ownership", "Fix this code"],
        "input": ["", "fn main() { let x = 1; }"],
        "output": ["Ownership is a core Rust feature.", "Here is the fix."],
    }
    res = format_chatml_prompts(examples)
    assert "text" in res
    assert len(res["text"]) == 2

    # First record has no input, user_message should just be the instruction
    expected_msg_0 = CHATML_PROMPT_TEMPLATE.format(
        system_prompt=DEFAULT_RUST_SYSTEM_PROMPT,
        user_message="Explain Rust ownership",
        assistant_message="Ownership is a core Rust feature.",
    )
    assert res["text"][0] == expected_msg_0

    # Second record has input, user_message should be instruction + \n + input
    expected_msg_1 = CHATML_PROMPT_TEMPLATE.format(
        system_prompt=DEFAULT_RUST_SYSTEM_PROMPT,
        user_message="Fix this code\nfn main() { let x = 1; }",
        assistant_message="Here is the fix.",
    )
    assert res["text"][1] == expected_msg_1


def test_format_chatml_prompts_custom_system_prompt() -> None:
    """Verify custom system prompt in format_chatml_prompts."""
    examples = {
        "instruction": ["Do something"],
        "input": [""],
        "output": ["Done"],
    }
    custom_sys = "Custom system prompt."
    res = format_chatml_prompts(examples, system_prompt=custom_sys)
    assert custom_sys in res["text"][0]


def test_format_alpaca_prompts() -> None:
    """Verify format_alpaca_prompts generates Alpaca prompt format."""
    examples = {
        "instruction": ["Write a macro"],
        "input": ["macro_rules! my_macro"],
        "output": ["macro_rules! my_macro { () => {} }"],
    }
    res = format_alpaca_prompts(examples)
    assert "text" in res
    assert len(res["text"]) == 1

    expected = ALPACA_PROMPT_TEMPLATE.format(
        "Write a macro",
        "macro_rules! my_macro",
        "macro_rules! my_macro { () => {} }",
    )
    assert res["text"][0] == expected


def test_load_and_prepare_dataset_with_tokenizer() -> None:
    """Verify load_and_prepare_dataset loads, formats, and logs sample token length."""
    mock_raw_dataset = [{"instruction": "test", "input": "", "output": "result"}]

    class MockDataset(list):
        def map(self, fn: object, batched: bool = True) -> "MockDataset":
            return MockDataset([{"text": "Sample prompt text"}])

    mock_loaded = MockDataset(mock_raw_dataset)

    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    logger = logging.getLogger("test_dataset")

    with patch("lib.dataset.load_dataset", return_value=mock_loaded) as mock_load:
        formatted = load_and_prepare_dataset(
            data_file="dummy.jsonl",
            prompt_formatter_fn=lambda x: x,
            tokenizer=mock_tokenizer,
            max_seq_length=1024,
            logger=logger,
        )
        mock_load.assert_called_once_with("json", data_files="dummy.jsonl", split="train")
        assert len(formatted) == 1
        assert formatted[0]["text"] == "Sample prompt text"
        mock_tokenizer.encode.assert_called_once_with("Sample prompt text")


def test_load_and_prepare_dataset_without_tokenizer_and_unsized() -> None:
    """Verify load_and_prepare_dataset when dataset is unsized and tokenizer is None."""

    class UnsizedDataset:
        def map(self, fn: object, batched: bool = True) -> list[dict[str, str]]:
            return [{"text": "Mapped text"}]

    mock_loaded = UnsizedDataset()

    with patch("lib.dataset.load_dataset", return_value=mock_loaded):
        res = load_and_prepare_dataset(
            data_file="dummy.jsonl",
            prompt_formatter_fn=lambda x: x,
            tokenizer=None,
            logger=None,
        )
        assert len(res) == 1
        assert res[0]["text"] == "Mapped text"


def test_load_and_prepare_dataset_empty_with_tokenizer() -> None:
    """Verify load_and_prepare_dataset when formatted dataset has length 0."""

    class MockEmptyDataset(list):
        def map(self, fn: object, batched: bool = True) -> "MockEmptyDataset":
            return MockEmptyDataset()

    mock_loaded = MockEmptyDataset()
    mock_tokenizer = MagicMock()

    with patch("lib.dataset.load_dataset", return_value=mock_loaded):
        res = load_and_prepare_dataset(
            data_file="empty.jsonl",
            prompt_formatter_fn=lambda x: x,
            tokenizer=mock_tokenizer,
            max_seq_length=None,
        )
        assert len(res) == 0
        mock_tokenizer.encode.assert_not_called()
