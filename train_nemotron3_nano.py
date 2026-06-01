#!/usr/bin/env python3
"""
Unsloth Studio Training Script for NVIDIA Nemotron 3 Nano 4B

Training uses the base HuggingFace model (NVIDIA/Nemotron-3-Nano-4B) or a local
model path. This script uses 16-bit/bf16 LoRA training, which is the recommended
fine-tuning mode for Nemotron 3 Nano.

Usage:
    # Use HuggingFace model (downloads if not cached):
    uv run python train_nemotron3_nano.py

    # Use local model directory:
    uv run python train_nemotron3_nano.py --model-path /path/to/Nemotron-3-Nano-4B

Documentation References:
- Nemotron 3 Nano docs: https://unsloth.ai/docs/models/nemotron-3
- Unsloth GGUF: https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF
- Unsloth Documentation: https://unsloth.ai/docs/models/nemotron-3
"""

# ruff: noqa: I001
from unsloth import FastLanguageModel

import argparse

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl.trainer.sft_trainer import SFTTrainer

max_seq_length = 1024
# Nemotron 3 Nano is best fine-tuned with bf16/16-bit LoRA. 4-bit QLoRA is not
# recommended for this family.
# Defaults are tuned for a 12GB GPU such as an RTX 4070.
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
load_in_4bit = False
load_in_16bit = True


def train_nemotron(
    model_path: str = "NVIDIA/Nemotron-3-Nano-4B",
    max_seq_length: int = max_seq_length,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
):
    """Train a LoRA adapter on NVIDIA Nemotron 3 Nano 4B."""
    print(f"Loading model from: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        load_in_16bit=load_in_16bit,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    nemotron_prompt = """<|im_start|>system
You are an expert Rust programmer, specializing in systems programming, async Rust, ESP32 embedded development, and popular Rust libraries.<|im_end|>
<|im_start|>user
{}{}<|im_end|>
<|im_start|>assistant
{}"""

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs, strict=False):
            user_msg = instruction + "\n" + input if input else instruction
            text = nemotron_prompt.format(user_msg, "", output) + "<|im_end|>"
            texts.append(text)
        return {"text": texts}

    dataset = load_dataset("json", data_files="data/datasets/train.jsonl", split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=10,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            output_dir="models/nemotron3_nano_rust_lora",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
        ),
    )

    trainer.train()
    model.save_pretrained("models/nemotron3_nano_rust_lora")
    tokenizer.save_pretrained("models/nemotron3_nano_rust_lora")
    print("Training complete! LoRA saved to models/nemotron3_nano_rust_lora")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA on NVIDIA Nemotron 3 Nano 4B")
    parser.add_argument(
        "--model-path",
        type=str,
        default="NVIDIA/Nemotron-3-Nano-4B",
        help="Model path (HuggingFace ID or local directory). "
        "Examples: 'NVIDIA/Nemotron-3-Nano-4B' or '/path/to/Nemotron-3-Nano-4B'",
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
        default=4,
        help="Gradient accumulation steps to simulate a larger batch size.",
    )
    args = parser.parse_args()

    train_nemotron(
        model_path=args.model_path,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
