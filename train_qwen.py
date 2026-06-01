#!/usr/bin/env python3
"""
Unsloth Studio Training Script for Qwen 3.5 4B Instruct

Training uses the base HuggingFace model (Qwen/Qwen3.5-4B) or a local model path.
After training, the export script can merge the LoRA adapter and package the model
for deployment.

Usage:
    # Use HuggingFace model (downloads if not cached):
    uv run python train_qwen.py

    # Use local model directory:
    uv run python train_qwen.py --model-path /path/to/Qwen3.5-4B

Documentation References:
- HF Model Card: https://huggingface.co/Qwen/Qwen3.5-4B
- Unsloth GGUF: https://huggingface.co/unsloth/Qwen3.5-4B-GGUF
- Unsloth Documentation: https://unsloth.ai/docs/models/qwen3.5
"""

# ruff: noqa: I001
import argparse

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl.trainer.sft_trainer import SFTTrainer
from unsloth import FastLanguageModel

max_seq_length = 2048
# Qwen3.5 4B is best trained with bf16/16-bit LoRA; 4-bit QLoRA is not recommended for this model family.
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
load_in_4bit = False
load_in_16bit = True


def train_qwen(model_path: str = "Qwen/Qwen3.5-4B"):
    """Train a LoRA adapter on Qwen 3.5 4B."""
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

    # Qwen uses ChatML format
    qwen_prompt = """<|im_start|>system
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
            text = qwen_prompt.format(user_msg, "", output) + "<|im_end|>"
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
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            output_dir="models/qwen3_5_4b_rust_lora",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
        ),
    )

    trainer.train()
    model.save_pretrained("models/qwen3_5_4b_rust_lora")
    tokenizer.save_pretrained("models/qwen3_5_4b_rust_lora")
    print("Training complete! LoRA saved to models/qwen3_5_4b_rust_lora")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoRA on Qwen 3.5 4B")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3.5-4B",
        help="Model path (HuggingFace ID or local directory). "
        "Examples: 'Qwen/Qwen3.5-4B' or '/path/to/Qwen3.5-4B'",
    )
    args = parser.parse_args()

    train_qwen(model_path=args.model_path)
