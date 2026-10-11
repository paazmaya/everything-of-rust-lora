#!/usr/bin/env python3
"""
Export trained LoRA to Ollama GGUF format with Q4_K_M quantization.
Run this AFTER training is complete.

Quantization: Q4_K_M (4-bit K_M) - recommended for Granite 4.2 8B
See: https://huggingface.co/ibm-granite/granite-4.2-8b-GGUF?show_file_info=granite-4.2-8b-Q4_K_M.gguf

Documentation References:
- Unsloth: https://unsloth.ai/docs/models/ibm-granite-4.1
- HF Model Card: https://huggingface.co/ibm-granite/granite-4.2-8b-GGUF
- HF Blog: https://huggingface.co/blog/ibm-granite/granite-4-1
- IBM Docs: https://www.ibm.com/granite/docs/models/granite4-1
"""

import os
from pathlib import Path
from typing import Any

import yaml
from peft import PeftModel
from unsloth import FastLanguageModel


def get_cutoff_date() -> str:
    """Read knowledge cutoff date from config/cybersec.yaml."""
    cfg_file = Path(__file__).parent.parent / "config" / "cybersec.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Missing required configuration file: {cfg_file}")
    with open(cfg_file) as f:
        cfg = yaml.safe_load(f) or {}
    cutoff = cfg.get("metadata", {}).get("cutoff_date")
    if not cutoff:
        raise ValueError(f"cutoff_date not found in {cfg_file} under metadata.cutoff_date")
    return str(cutoff)


def detect_family(model_hint: str) -> str:
    """Detect architectural family from model path or identifier."""
    hint = model_hint.lower()
    if "granite" in hint:
        return "granite"
    if "k2" in hint or "horizon" in hint:
        return "k2_horizon"
    if "nemotron" in hint or "nemo" in hint:
        return "nemotron"
    if "3.8" in hint or "qwen3.8" in hint or "27b" in hint:
        return "qwen3_8"
    if "qwen" in hint:
        return "qwen"
    return "qwen"


def build_ollama_modelfile_content(
    from_path: str,
    family: str,
    system_prompt: str,
) -> str:
    """Generate family-tailored Ollama Modelfile content with precise templates and stop tokens."""
    if family == "k2_horizon":
        return f"""FROM {from_path}
PARAMETER temperature 0.2
PARAMETER num_ctx 32768
PARAMETER stop "<|ifm|im_end|>"
PARAMETER stop "<|ifm|endoftext|>"
PARAMETER stop "<|im_end|>"
TEMPLATE \"\"\"{{{{ if .System }}}}<|ifm|im_start|>system
{{{{ .System }}}}<|ifm|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|ifm|im_start|>user
{{{{ .Prompt }}}}<|ifm|im_end|>
{{{{ end }}}}<|ifm|im_start|>assistant
<ifm|think>
</ifm|think>{{{{ .Response }}}}<|ifm|im_end|>\"\"\"
SYSTEM {system_prompt}
"""
    elif family == "granite":
        return f"""FROM {from_path}
PARAMETER temperature 0.2
PARAMETER num_ctx 32768
PARAMETER stop "<|end_of_text|>"
PARAMETER stop "<|start_of_role|>"
TEMPLATE \"\"\"{{{{- range .Messages }}}}
{{{{- if eq .Role "system" }}}}<|start_of_role|>system<|end_of_role|>{{{{ .Content }}}}<|end_of_text|>
{{{{- else if eq .Role "user" }}}}<|start_of_role|>user<|end_of_role|>{{{{ .Content }}}}<|end_of_text|>
{{{{- else if eq .Role "assistant" }}}}<|start_of_role|>assistant<|end_of_role|>{{{{ .Content }}}}<|end_of_text|>
{{{{- end }}}}
{{{{- end }}}}
{{{{- if .Prompt }}}}<|start_of_role|>user<|end_of_role|>{{{{ .Prompt }}}}<|end_of_text|><|start_of_role|>assistant<|end_of_role|>{{{{- end }}}}\"\"\"
SYSTEM {system_prompt}
"""
    elif family == "nemotron":
        return f"""FROM {from_path}
PARAMETER temperature 0.2
PARAMETER num_ctx 32768
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<extra_id_1>"
PARAMETER stop "<extra_id_2>"
TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>\"\"\"
SYSTEM {system_prompt}
"""
    elif family == "qwen3_8":
        return f"""FROM {from_path}
PARAMETER temperature 0.2
PARAMETER num_ctx 65536
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>\"\"\"
SYSTEM {system_prompt}
"""
    else:  # qwen 3.5
        return f"""FROM {from_path}
PARAMETER temperature 0.2
PARAMETER num_ctx 32768
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>\"\"\"
SYSTEM {system_prompt}
"""


def export_to_ollama(
    model_path: str, model_name: str, base_model: str, quant: str = "q4_k_m"
) -> None:
    """
    Export trained LoRA to GGUF format and create Ollama Modelfile.

    Args:
        model_path: Path to trained LoRA weights (e.g., models/granite_rust_lora)
        model_name: Name for the exported model (e.g., rust-granite)
        base_model: Base model identifier or local path. Can be:
            - HuggingFace model ID: "ibm-granite/granite-4.2-8b" or "Qwen/Qwen3.5-4B"
            - Local file path: "/path/to/granite-4.2-8b-Q4_K_M.gguf"
        quant: Quantization method (default: q4_k_m = Q4_K_M, 4-bit K_M)
    """
    model: Any
    tokenizer: Any

    effective_base = base_model
    adapter_cfg_path = os.path.join(model_path, "adapter_config.json")
    if not effective_base and os.path.exists(adapter_cfg_path):
        import json

        with open(adapter_cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
            effective_base = cfg.get("base_model_name_or_path", "")

    family = detect_family(effective_base or model_path or model_name)
    print(f"Detected model architecture family: {family}")

    if effective_base and effective_base != model_path:
        print(f"Loading base model from {effective_base}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=effective_base,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        print(f"Applying LoRA adapter from {model_path}...")
        model = PeftModel.from_pretrained(model, model_path, is_trainable=False)
    elif os.path.isdir(model_path) and os.path.exists(adapter_cfg_path):
        print(f"Loading LoRA checkpoint and configured base model from {model_path}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
        )
    else:
        raise ValueError(
            f"Cannot resolve base model for {model_path}. Provide --base <model_id_or_path>."
        )

    # Merge LoRA weights into base model
    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    # Save as GGUF
    gguf_dir = f"models/{model_name}_gguf"
    print(f"Saving to GGUF ({quant}) at {gguf_dir}...")
    result = model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=quant)
    gguf_files = result.get("gguf_files") if isinstance(result, dict) else None
    gguf_filename = None
    if gguf_files:
        gguf_filename = os.path.basename(gguf_files[0])
    else:
        # Fallback: find the first GGUF file in the output directory
        if os.path.isdir(gguf_dir):
            for entry in os.listdir(gguf_dir):
                if entry.lower().endswith(".gguf"):
                    gguf_filename = entry
                    break

    # Create Ollama Modelfile
    if gguf_filename is None:
        raise RuntimeError(f"Could not determine GGUF filename in {gguf_dir}.")
    from_path = gguf_filename

    cutoff_date = get_cutoff_date()

    system_prompt = (
        f"You are an expert Rust programmer and cybersecurity engineer specializing in systems programming, "
        f"async Rust, ESP32 embedded development, RustSec vulnerability analysis, Kali Linux tooling, and "
        f"living-off-the-land (LOTL) security. Your cybersecurity, vulnerability, and tooling knowledge cutoff "
        f"date is {cutoff_date}. For vulnerabilities discovered after this date, advise running cargo-audit."
    )

    modelfile_content = build_ollama_modelfile_content(
        from_path=from_path,
        family=family,
        system_prompt=system_prompt,
    )

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
        default="ibm-granite/granite-4.2-8b-GGUF",
        help="Base model (HF model ID or local GGUF file path). Examples: 'ibm-granite/granite-4.2-8b-GGUF' or '/path/to/granite-4.2-8b-Q4_K_M.gguf'",
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="q4_k_m",
        help="Quantization method (default: q4_k_m = Q4_K_M, 4-bit K_M)",
    )
    args = parser.parse_args()
    export_to_ollama(args.model, args.name, args.base, args.quant)
