#!/usr/bin/env python3
"""
Unsloth Studio Training Script for IBM Granite 4.1 8B

Training uses the base HuggingFace model (ibm-granite/granite-4.1-8b) or a local model path.
After training, the export script (scripts/09_export_ollama.py) merges the LoRA
and creates a GGUF file with Q4_K_M quantization.

Usage:
    # Use HuggingFace model (downloads if not cached):
    uv run python train_granite.py

    # Use local model directory:
    uv run python train_granite.py --model-path /path/to/granite-4.1-8b

Documentation References:
- Unsloth: https://unsloth.ai/docs/models/ibm-granite-4.1
- HF Model Card: https://huggingface.co/unsloth/granite-4.1-8b-GGUF?show_file_info=granite-4.1-8b-Q4_K_M.gguf
- HF Blog: https://huggingface.co/blog/ibm-granite/granite-4-1
- IBM Docs: https://www.ibm.com/granite/docs/models/granite4-1
"""

# ruff: noqa: I001
from unsloth import FastLanguageModel

import argparse

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl.trainer.sft_trainer import SFTTrainer

max_seq_length = 4096
dtype = None
load_in_4bit = True


def train_granite(model_path: str = "ibm-granite/granite-4.1-8b"):
    """
    Train a LoRA adapter on IBM Granite 4.1 8B model.

    Args:
        model_path: HuggingFace model ID or local directory path
                   (e.g., "ibm-granite/granite-4.1-8b" or "/path/to/granite-4.1-8b")
    """
    print(f"Loading model from: {model_path}")
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
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            warmup_steps=100,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            output_dir="models/granite_rust_lora",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
        ),
    )

    trainer.train()
    model.save_pretrained("models/granite_rust_lora")
    tokenizer.save_pretrained("models/granite_rust_lora")
    print("Training complete! LoRA saved to models/granite_rust_lora")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA on IBM Granite 4.1 8B")
    parser.add_argument(
        "--model-path",
        type=str,
        default="ibm-granite/granite-4.1-8b",
        help="Model path (HuggingFace ID or local directory). "
        "Examples: 'ibm-granite/granite-4.1-8b' or '/path/to/granite-4.1-8b'",
    )
    args = parser.parse_args()

    train_granite(model_path=args.model_path)
