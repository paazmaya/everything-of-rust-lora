#!/usr/bin/env python3
"""
Unsloth Studio Training Script for Qwen 2.5 7B Instruct
Run this in Unsloth Studio (Google Colab) or locally with a 24GB+ GPU.
"""

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

max_seq_length = 4096
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=128,
    lora_dropout=0.05,
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
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        output_dir="models/qwen_rust_lora",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
    ),
)

trainer.train()
model.save_pretrained("models/qwen_rust_lora")
tokenizer.save_pretrained("models/qwen_rust_lora")
print("Training complete! LoRA saved to models/qwen_rust_lora")
