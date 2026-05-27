#!/usr/bin/env python3
"""
Export trained LoRA to Ollama GGUF format with Q4_K_M quantization.
Run this AFTER training is complete.

Quantization: Q4_K_M (4-bit K_M) - recommended for Granite 4.1 8B
See: https://huggingface.co/unsloth/granite-4.1-8b-GGUF?show_file_info=granite-4.1-8b-Q4_K_M.gguf

Documentation References:
- Unsloth: https://unsloth.ai/docs/models/ibm-granite-4.1
- HF Model Card: https://huggingface.co/unsloth/granite-4.1-8b-GGUF?show_file_info=granite-4.1-8b-Q4_K_M.gguf
- HF Blog: https://huggingface.co/blog/ibm-granite/granite-4-1
- IBM Docs: https://www.ibm.com/granite/docs/models/granite4-1
"""

from unsloth import FastLanguageModel


def export_to_ollama(model_path: str, model_name: str, base_model: str, quant: str = "q4_k_m"):
    """
    Export trained LoRA to GGUF format and create Ollama Modelfile.

    Args:
        model_path: Path to trained LoRA weights (e.g., models/granite_rust_lora)
        model_name: Name for the exported model (e.g., rust-granite)
        base_model: Base model identifier or local path. Can be:
            - HuggingFace model ID: "unsloth/granite-4.1-8b-GGUF" (downloads if not cached)
            - Local file path: "/path/to/granite-4.1-8b-Q4_K_M.gguf" (no download needed)
        quant: Quantization method (default: q4_k_m = Q4_K_M, 4-bit K_M)
    """
    print(f"Loading model from {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )

    # Merge LoRA weights into base model
    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    # Save as GGUF
    gguf_dir = f"models/{model_name}_gguf"
    print(f"Saving to GGUF ({quant}) at {gguf_dir}...")
    model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=quant)

    # Create Ollama Modelfile
    modelfile_content = f"""FROM {gguf_dir}
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.2
SYSTEM You are an expert Rust programmer specializing in systems programming, async Rust, ESP32 embedded development, and the Rust ecosystem.
"""

    modelfile_path = f"{gguf_dir}/Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    print("\n" + "=" * 50)
    print("GGUF Export Complete!")
    print("=" * 50)
    print("\nTo import into Ollama, run:")
    print(f"  ollama create {model_name} -f {modelfile_path}")
    print("\nThen run it with:")
    print(f"  ollama run {model_name}")


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
        "--name", type=str, required=True, help="Name for Ollama model (e.g., rust-expert)"
    )
    parser.add_argument(
        "--base",
        type=str,
        default="unsloth/granite-4.1-8b-GGUF",
        help="Base model (HF model ID or local GGUF file path). Examples: 'unsloth/granite-4.1-8b-GGUF' or '/path/to/granite-4.1-8b-Q4_K_M.gguf'",
    )
    args = parser.parse_args()

    export_to_ollama(args.model, args.name, args.base)
