#!/usr/bin/env python3
"""
Unsloth Studio Training Script for IBM Granite 4.2 (8B and 3B variants)

Training uses the Granite 4.2 model with automatic variant detection from the model ID or path.
Supported variants: 8B (default) and 3B.

After training, the export script (scripts/10_export_ollama.py) merges the LoRA
and creates a GGUF file with Q4_K_M quantization.

Usage:
    # Use default Granite 4.2 8B model (downloads if not cached):
    uv run python train_granite.py

    # Use Granite 4.2 3B variant:
    uv run python train_granite.py --model-path ibm-granite/granite-4.2-3b

    # Use local model directory (auto-detects from path name):
    uv run python train_granite.py --model-path /path/to/granite-4.2-8b
    uv run python train_granite.py --model-path /path/to/granite-4.2-3b

Variant Detection:
    The script auto-detects the model size (8B or 3B) from the model ID or directory path.
    This determines the output directory name for clarity.

Documentation References:
- Granite 4.2 Collection: https://huggingface.co/collections/ibm-granite/granite-42-language-models
- Granite 4.2 GitHub: https://github.com/ibm-granite/granite-4.2-language-models
- HF Blog: https://huggingface.co/blog/ibm-granite/granite-4-2
- IBM Docs: https://www.ibm.com/granite/docs/
"""

# ruff: noqa: I001
import re
from unsloth import FastLanguageModel

import argparse

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl.trainer.sft_trainer import SFTTrainer

max_seq_length = 4096
dtype = None
load_in_4bit = True


def detect_model_variant(model_path: str) -> str:
    """
    Detect the model variant (8B or 3B) from the model ID or local path.

    Args:
        model_path: HuggingFace model ID (e.g., "ibm-granite/granite-4.2-8b")
                   or local directory path (e.g., "/path/to/granite-4.2-3b")

    Returns:
        "8B" or "3B" based on pattern matching in the model path/name.
        Defaults to "8B" if no clear pattern is found.
    """
    # Case-insensitive search for variant in the path
    if re.search(r"3b", model_path, re.IGNORECASE):
        return "3B"
    elif re.search(r"8b", model_path, re.IGNORECASE):
        return "8B"
    # Default to 8B if unclear
    return "8B"


def train_granite(
    model_path: str = "ibm-granite/granite-4.2-8b",
    max_seq_length: int = 4096,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
):
    """
    Train a LoRA adapter on IBM Granite 4.2 model (8B or 3B variant).

    Args:
        model_path: HuggingFace model ID or local directory path.
                   Supports both 8B and 3B variants.
                   Examples:
                   - "ibm-granite/granite-4.2-8b" (default)
                   - "ibm-granite/granite-4.2-3b"
                   - "/path/to/granite-4.2-8b"
                   - "/path/to/granite-4.2-3b"
        max_seq_length: Maximum sequence length for training.
        batch_size: Per-device batch size for training.
        gradient_accumulation_steps: Gradient accumulation steps to simulate a larger batch size.
    """
    # Detect variant from model path
    variant = detect_model_variant(model_path)
    output_dir = f"models/granite_4.2_{variant.lower()}_rust_lora"

    print("Granite 4.2 LoRA Training")
    print(f"Detected model variant: {variant}")
    print(f"Loading model from: {model_path}")
    print(f"Output directory: {output_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs, strict=False):
            text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
            texts.append(text)
        return {"text": texts}

    dataset = load_dataset("json", data_files="data/datasets/train.jsonl", split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",  # type: ignore[arg-type]
        max_seq_length=max_seq_length,  # type: ignore[arg-type]
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            warmup_steps=100,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            output_dir=output_dir,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
        ),
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete! LoRA saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LoRA on IBM Granite 4.2 (8B or 3B variant)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default Granite 4.2 8B:
  uv run python train_granite.py

  # Train with Granite 4.2 3B variant:
  uv run python train_granite.py --model-path ibm-granite/granite-4.2-3b

  # Train with local model (auto-detects variant from path):
  uv run python train_granite.py --model-path /path/to/granite-4.2-3b
        """,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="ibm-granite/granite-4.2-8b",
        help="Model path (HuggingFace ID or local directory). Supports 8B and 3B variants. "
        "Auto-detects variant from path name. "
        "Examples: 'ibm-granite/granite-4.2-8b', 'ibm-granite/granite-4.2-3b', or '/path/to/granite-4.2-3b'",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=max_seq_length,
        help="Maximum sequence length for training. Lower values reduce VRAM use.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device batch size for training.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps to simulate a larger batch size.",
    )
    args = parser.parse_args()

    train_granite(
        model_path=args.model_path,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
