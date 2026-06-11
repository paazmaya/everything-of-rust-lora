#!/usr/bin/env python3
"""
Export trained LoRA adapter weights without merging the base model.
Run this AFTER training is complete.

This saves only the adapter checkpoint, not a full GGUF merge.

Documentation References:
- Unsloth: https://unsloth.ai/docs/models/ibm-granite-4.1
- HF Model Card: https://huggingface.co/unsloth/granite-4.1-8b-GGUF?show_file_info=granite-4.1-8b-Q4_K_M.gguf
- HF Blog: https://huggingface.co/blog/ibm-granite/granite-4-1
- IBM Docs: https://www.ibm.com/granite/docs/models/granite4-1
"""
import os

from unsloth import FastLanguageModel
from unsloth.save import save_lora_to_custom_dir


def export_lora_adapter(model_path: str, model_name: str):
    """
    Export trained LoRA adapter only, without merging the base model.

    Args:
        model_path: Path to trained LoRA weights (e.g., models/granite_rust_lora)
        model_name: Name for the exported adapter directory (e.g., rust-granite)
    """
    if os.path.isdir(model_path) and os.path.exists(
        os.path.join(model_path, "adapter_config.json")
    ):
        print(f"Loading LoRA checkpoint and its base model from {model_path}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
        )
    else:
        raise ValueError(
            "Model path must be a directory containing a LoRA adapter checkpoint (with adapter_config.json)."
        )

    # Save only the LoRA adapter, without merging the base model.
    lora_dir = f"models/{model_name}_lora"
    print(f"Saving LoRA-only adapter to {lora_dir}...")
    save_lora_to_custom_dir(model, tokenizer, lora_dir)

    print("LoRA-only export complete! The saved directory contains only the adapter weights.")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained LoRA (e.g., models/granite_rust_lora)",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name for the exported LoRA adapter directory (e.g., rust-expert)"
    )
    args = parser.parse_args()
    export_lora_adapter(args.model, args.name)
