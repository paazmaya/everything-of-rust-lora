#!/usr/bin/env python3
"""
Export a trained LoRA adapter checkpoint into a packaged adapter directory and
generate a metadata-rich Modelfile describing the base model and training
economics.

This script does not quantize the LoRA adapter to GGUF. LoRA adapters cannot be
quantized directly; instead this script exports the adapter weights and creates
metadata for a later merge with the base model.

The generated Modelfile contains:
- Base model reference and GGUF link
- Training economics summary
- Adapter package notes and usage guidance
"""

import argparse
import datetime
import os
from pathlib import Path
from typing import Dict, Optional

from unsloth import FastLanguageModel
from unsloth.save import save_lora_to_custom_dir

BASE_MODEL_METADATA: Dict[str, Dict[str, str]] = {
    "granite": {
        "name": "IBM Granite 4.1 8B",
        "reference": "ibm-granite/granite-4.1-8b",
        "gguf": "unsloth/granite-4.1-8b-GGUF",
        "source_url": "https://huggingface.co/unsloth/granite-4.1-8b-GGUF",
        "training_economics": (
            "LoRA rank=64, alpha=128, optimizer=AdamW 8-bit, scheduler=cosine, "
            "batch_size=1, gradient_accumulation_steps=8 (effective=8), epochs=3, "
            "VRAM≈16GB, training time≈2-3 hours on a 24GB GPU."
        ),
        "notes": (
            "This adapter was trained on a Rust-specific corpus targeting systems, "
            "async, and embedded Rust. Merge with the Granite base model and quantize "
            "the merged model separately."
        ),
    },
    "qwen": {
        "name": "Qwen 3.5 4B Instruct",
        "reference": "Qwen/Qwen3.5-4B",
        "gguf": "unsloth/Qwen3.5-4B-GGUF",
        "source_url": "https://huggingface.co/Qwen/Qwen3.5-4B",
        "training_economics": (
            "LoRA rank=16, alpha=16, optimizer=AdamW 8-bit, scheduler=cosine, "
            "batch_size=1, gradient_accumulation_steps=4 (effective=4), epochs=3, "
            "VRAM≈12GB, best used with bf16/16-bit LoRA training."
        ),
        "notes": (
            "This adapter is designed for the Qwen 3.5 4B instruction model and uses "
            "the Qwen ChatML prompt style. Merge with the base model before quantization."
        ),
    },
    "nemotron": {
        "name": "NVIDIA Nemotron 3 Nano 4B",
        "reference": "NVIDIA/Nemotron-3-Nano-4B",
        "gguf": "unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF",
        "source_url": "https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF",
        "training_economics": (
            "LoRA rank=16, alpha=16, optimizer=AdamW 8-bit, scheduler=cosine, "
            "batch_size=1, gradient_accumulation_steps=4 (effective=4), epochs=3, "
            "VRAM≈12GB, recommended bf16/16-bit LoRA."
        ),
        "notes": (
            "This adapter targets the Nemotron 3 Nano family and is best merged with "
            "the corresponding base model before deployment."
        ),
    },
}


def get_base_metadata(base_key: str) -> Dict[str, str]:
    if base_key not in BASE_MODEL_METADATA:
        raise ValueError(
            f"Base model key must be one of: {', '.join(BASE_MODEL_METADATA)}. Got '{base_key}'."
        )
    return BASE_MODEL_METADATA[base_key]


def build_modelfile_content(
    model_name: str,
    base_metadata: Dict[str, str],
    base_override: Optional[str] = None,
) -> str:
    base_reference = base_override or base_metadata["reference"]
    created_at = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    lines = [
        "# Modelfile for a packaged LoRA adapter",
        "# This file documents the base model and training economics.",
        "# LoRA adapters cannot be quantized directly to GGUF; merge with the base model first.",
        "",
        f"name: {model_name}",
        f"created_at: {created_at}",
        "",
        "base_model:",
        f"  name: {base_metadata['name']}",
        f"  reference: {base_reference}",
        f"  gguf: {base_metadata['gguf']}",
        f"  source_url: {base_metadata['source_url']}",
        "",
        "training_economics:",
        f"  summary: '{base_metadata['training_economics']}'",
        "  dataset: data/datasets/train.jsonl",
        "  instruction: Rust programming, systems, async Rust, embedded Rust, and library usage.",
        "",
        "lora_adapter:",
        f"  directory: models/{model_name}_lora",
        "  export_type: adapter-only",
        "  notes: LoRA adapter package contains only adapter weights and tokenizer metadata.",
        "",
        "notes:",
        f"  - {base_metadata['notes']}",
        "  - 'To create a runnable GGUF model, merge this adapter with the base model and quantize the merged model.'",
    ]

    return "\n".join(lines) + "\n"


def export_lora_adapter(
    model_path: str,
    model_name: str,
    base_key: str,
    base_override: Optional[str] = None,
) -> None:
    if not os.path.isdir(model_path) or not os.path.exists(
        os.path.join(model_path, "adapter_config.json")
    ):
        raise ValueError(
            "Model path must be a directory containing a LoRA adapter checkpoint (with adapter_config.json)."
        )

    print(f"Loading LoRA checkpoint from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )

    output_dir = Path("models") / f"{model_name}_lora"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving LoRA adapter-only package to {output_dir}...")
    save_lora_to_custom_dir(model, tokenizer, str(output_dir))

    base_metadata = get_base_metadata(base_key)
    modelfile_path = output_dir / "Modelfile"
    print(f"Writing Modelfile to {modelfile_path}...")
    modelfile_content = build_modelfile_content(model_name, base_metadata, base_override)
    modelfile_path.write_text(modelfile_content, encoding="utf-8")

    print("\nLoRA adapter export complete!")
    print(f"Adapter package created at: {output_dir}")
    print(f"Modelfile created at: {modelfile_path}")
    print("\nNext step: merge this adapter with the specified base model and quantize the merged model separately.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Export a LoRA adapter checkpoint and generate a Modelfile containing "
            "the base model and training economics metadata."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained LoRA checkpoint directory (e.g., models/granite_rust_lora)",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for the exported adapter package (e.g., rust-granite)",
    )
    parser.add_argument(
        "--base",
        type=str,
        choices=list(BASE_MODEL_METADATA.keys()),
        default="granite",
        help=(
            "Base model family for this adapter. Choices: granite, qwen, nemotron. "
            "Used to populate base model metadata and training economics."
        ),
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help=(
            "Optional override for the base model reference. Can be a local path or model ID. "
            "If provided, this value will be used in the Modelfile instead of the default base reference."
        ),
    )
    args = parser.parse_args()

    export_lora_adapter(args.model, args.name, args.base, args.base_model)
