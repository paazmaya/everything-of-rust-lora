#!/usr/bin/env python3
"""
Export trained LoRA to Ollama GGUF format.
Run this AFTER training is complete.
"""

from unsloth import FastLanguageModel


def export_to_ollama(model_path: str, model_name: str, base_model: str, quant: str = "q4_k_m"):
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
        "--base", type=str, default="ibm-granite/granite-8b-code-instruct", help="Base model used"
    )
    args = parser.parse_args()

    export_to_ollama(args.model, args.name, args.base)
