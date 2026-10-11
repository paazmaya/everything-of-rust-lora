#!/usr/bin/env python3
"""
Export trained Rust LoRA adapter for Unsloth Studio compatibility.

Supports three export targets:
1. `merged-hf`: Merges the LoRA adapter into the base model weights, outputting
   a complete Hugging Face model directory (safetensors, configs, tokenizer,
   `chat_template.jinja`, `generation_config.json`, and an Unsloth Studio model card).
2. `adapter`: Packages the standalone LoRA checkpoint with complete tokenizer assets,
   `chat_template.jinja`, and an Unsloth Studio adapter manifest for direct loading
   into Unsloth Studio side-by-side with the base model.
3. `gguf`: Merges and exports directly to GGUF format via Unsloth's native
   `FastLanguageModel.save_pretrained_gguf` engine, ensuring the chat template and
   special tokens are embedded into the GGUF header.

Usage Examples:
    # 1. Export merged Hugging Face model for Unsloth Studio
    uv run python scripts/11_export_unsloth.py \
        --model models/granite_rust_lora \
        --base ibm-granite/granite-4.2-8b \
        --mode merged-hf

    # 2. Package standalone adapter for Unsloth Studio
    uv run python scripts/11_export_unsloth.py \
        --model models/k2_horizon_rust_lora \
        --base IFM/K2-Horizon-3.7B \
        --mode adapter

    # 3. Export quantized GGUF (Q4_K_M) for Unsloth Studio
    uv run python scripts/11_export_unsloth.py \
        --model models/qwen3_8_27b_rust_lora \
        --base Qwen/Qwen3.8-27B \
        --mode gguf \
        --quant q4_k_m
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rust_lora.unsloth_export")


# Standard Jinja chat templates for supported architectures
CHAT_TEMPLATES: dict[str, str] = {
    "granite": (
        "{%- for message in messages %}"
        "{{- '<|start_of_role|>' + message['role'] + '<|end_of_role|>' + message['content'] + '<|end_of_text|>' }}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "{{- '<|start_of_role|>assistant<|end_of_role|>' }}"
        "{%- endif %}"
    ),
    "k2_horizon": (
        "{%- for message in messages %}"
        "{{- '<|ifm|im_start|>' + message['role'] + '\\n' + message['content'] + '<|ifm|im_end|>\\n' }}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "{{- '<|ifm|im_start|>assistant\\n<ifm|think>\\n</ifm|think>' }}"
        "{%- endif %}"
    ),
    "nemotron": (
        "{%- for message in messages %}"
        "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "{{- '<|im_start|>assistant\\n' }}"
        "{%- endif %}"
    ),
    "qwen": (
        "{%- if messages[0]['role'] == 'system' %}"
        "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
        "{%- set messages = messages[1:] %}"
        "{%- else %}"
        "{{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}"
        "{%- endif %}"
        "{%- for message in messages %}"
        "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "{{- '<|im_start|>assistant\\n' }}"
        "{%- endif %}"
    ),
    "qwen3_8": (
        "{%- if not messages %}{{- raise_exception('No messages provided.') }}{%- endif %}"
        "{%- if messages[0]['role'] == 'system' %}"
        "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
        "{%- set messages = messages[1:] %}"
        "{%- endif %}"
        "{%- for message in messages %}"
        "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "{{- '<|im_start|>assistant\\n<think>\\n</think>' }}"
        "{%- endif %}"
    ),
}

DEFAULT_STOP_TOKENS: dict[str, list[str]] = {
    "granite": ["<|end_of_text|>", "<|start_of_role|>"],
    "k2_horizon": ["<|ifm|im_end|>", "<|ifm|endoftext|>", "<|im_end|>"],
    "nemotron": ["<|im_end|>", "<extra_id_1>", "<extra_id_2>"],
    "qwen": ["<|im_end|>", "<|endoftext|>"],
    "qwen3_8": ["<|im_end|>", "<|endoftext|>", "<|im_start|>"],
}


def get_cutoff_date() -> str:
    """Read knowledge cutoff date from config/cybersec.yaml."""
    cfg_file = Path(__file__).parent.parent / "config" / "cybersec.yaml"
    if not cfg_file.exists():
        return "2026-10-11"
    try:
        with open(cfg_file, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return str(cfg.get("metadata", {}).get("cutoff_date", "2026-10-11"))
    except Exception:
        return "2026-10-11"


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


def create_unsloth_model_card(
    output_dir: Path,
    model_name: str,
    base_model: str,
    family: str,
    is_adapter: bool = False,
    quant_method: str | None = None,
) -> None:
    """Write an Unsloth Studio compatible README.md model card."""
    cutoff_date = get_cutoff_date()
    quant_str = f" ({quant_method})" if quant_method else ""
    artifact_kind = "LoRA Adapter" if is_adapter else f"Merged Model{quant_str}"

    tags = ["unsloth", "rust", "rust-lang", "systems-programming", "cybersecurity", "lotl"]
    if is_adapter:
        tags.extend(["peft", "lora"])
    if quant_method:
        tags.append("gguf")

    readme_content = f"""---
base_model: {base_model}
library_name: {"peft" if is_adapter else "transformers"}
tags:
{yaml.dump(tags, default_flow_style=False).strip()}
license: apache-2.0
pipeline_tag: text-generation
---

# {model_name} - {artifact_kind} for Unsloth Studio

This model was produced by the [everything-of-rust-lora](https://github.com/everything-of-rust-lora) training pipeline.
It specializes in comprehensive Rust systems programming, async runtimes (Tokio, async-std), embedded Rust (ESP32/ESP-RS),
RustSec advisory triage, and living-off-the-land (LOTL) cybersecurity concepts.

## Key Metadata
- **Base Model:** `{base_model}`
- **Model Family:** `{family}`
- **Knowledge Cutoff:** `{cutoff_date}`
- **Format:** `{"Standalone LoRA Adapter" if is_adapter else "Merged Model"}`
- **Unsloth Studio Compatibility:** Full (includes `chat_template.jinja` & generation configs)

## Loading in Unsloth Studio

### Using Unsloth CLI / Web Interface
1. Launch Unsloth Studio:
   ```bash
   unsloth studio
   ```
2. In the Model Browser, point to this directory:
   `{output_dir.resolve().as_posix()}`
3. The embedded `chat_template.jinja` and special tokens will be loaded automatically.

## System Prompt
```
You are an expert Rust programmer and cybersecurity engineer specializing in systems programming,
async Rust, ESP32 embedded development, RustSec vulnerability analysis, Kali Linux tooling, and
living-off-the-land (LOTL) security. Your cybersecurity cutoff date is {cutoff_date}.
```
"""
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    logger.info(f"Generated Unsloth Studio model card at {readme_path}")


def ensure_chat_template(output_dir: Path, family: str, base_model_path: str | None = None) -> None:
    """Ensure chat_template.jinja exists in output_dir, copying or generating if needed."""
    target_jinja = output_dir / "chat_template.jinja"
    if target_jinja.exists():
        return

    # Check if base model has chat_template.jinja
    if base_model_path and os.path.isdir(base_model_path):
        base_jinja = Path(base_model_path) / "chat_template.jinja"
        if base_jinja.exists():
            shutil.copy2(base_jinja, target_jinja)
            logger.info(f"Copied chat_template.jinja from base model: {base_jinja}")
            return

    # Fallback to family-specific default template
    template_str = CHAT_TEMPLATES.get(family, CHAT_TEMPLATES["qwen"])
    target_jinja.write_text(template_str, encoding="utf-8")
    logger.info(f"Created default chat_template.jinja for family '{family}' at {target_jinja}")


def ensure_generation_config(output_dir: Path, family: str) -> None:
    """Ensure generation_config.json exists with recommended sampling settings."""
    config_path = output_dir / "generation_config.json"
    if config_path.exists():
        return

    stop_tokens = DEFAULT_STOP_TOKENS.get(family, ["<|im_end|>", "<|endoftext|>"])
    config_data = {
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.05,
        "max_new_tokens": 4096,
        "stop_strings": stop_tokens,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    logger.info(f"Created generation_config.json at {config_path}")


def export_adapter_package(
    model_path: str,
    base_model: str,
    output_dir: Path,
    family: str,
) -> None:
    """Package standalone LoRA with tokenizer and Unsloth Studio metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(model_path)

    logger.info(f"Copying LoRA adapter weights and configs from {src_dir} to {output_dir}...")
    for item in src_dir.iterdir():
        if item.is_file() and not item.name.startswith("."):
            shutil.copy2(item, output_dir / item.name)

    # Ensure tokenizer files exist; if not present in adapter, fetch from base model
    tok_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "special_tokens_map.json"]
    has_tokenizer = any((output_dir / f).exists() for f in tok_files)
    if not has_tokenizer:
        logger.info(f"Fetching tokenizer files from base model '{base_model}'...")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)

    ensure_chat_template(output_dir, family, base_model if os.path.isdir(base_model) else None)
    ensure_generation_config(output_dir, family)
    create_unsloth_model_card(
        output_dir=output_dir,
        model_name=output_dir.name,
        base_model=base_model,
        family=family,
        is_adapter=True,
    )
    logger.info(f"Standalone adapter package ready for Unsloth Studio at: {output_dir}")


def export_merged_hf(
    model_path: str,
    base_model: str,
    output_dir: Path,
    family: str,
    save_dtype: str = "bfloat16",
    max_seq_length: int = 4096,
    trust_remote_code: bool = True,
) -> None:
    """Merge LoRA into base model and save in full Hugging Face format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    import torch

    dtype = torch.bfloat16 if save_dtype == "bfloat16" else torch.float16

    # Attempt merge with Unsloth FastLanguageModel
    merged_successfully = False
    try:
        from unsloth import FastLanguageModel

        logger.info(f"Loading LoRA and base model via Unsloth from {model_path} / {base_model}...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path if os.path.isdir(model_path) else base_model,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=False,
            trust_remote_code=trust_remote_code,
        )
        logger.info("Merging LoRA weights with Unsloth merge_and_unload()...")
        model = model.merge_and_unload()
        logger.info(f"Saving merged HF model to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        merged_successfully = True
    except Exception as exc:
        logger.warning(
            f"Unsloth merge failed ({exc}). Falling back to Transformers + PEFT pipeline..."
        )

    if not merged_successfully:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading base model via Transformers: {base_model}...")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        logger.info(f"Applying LoRA adapter from {model_path}...")
        model = PeftModel.from_pretrained(model, model_path)
        logger.info("Merging LoRA weights into base model...")
        model = model.merge_and_unload()
        logger.info(f"Saving merged model weights to {output_dir}...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    ensure_chat_template(output_dir, family, base_model if os.path.isdir(base_model) else None)
    ensure_generation_config(output_dir, family)
    create_unsloth_model_card(
        output_dir=output_dir,
        model_name=output_dir.name,
        base_model=base_model,
        family=family,
        is_adapter=False,
    )
    logger.info(f"Merged Hugging Face model ready for Unsloth Studio at: {output_dir}")


def export_gguf_unsloth(
    model_path: str,
    base_model: str,
    output_dir: Path,
    family: str,
    quant: str = "q4_k_m",
    max_seq_length: int = 4096,
    trust_remote_code: bool = True,
) -> None:
    """Merge LoRA and export to GGUF using Unsloth native save_pretrained_gguf."""
    output_dir.mkdir(parents=True, exist_ok=True)
    from unsloth import FastLanguageModel

    logger.info(f"Loading LoRA/base for GGUF export ({quant}) via Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path if os.path.isdir(model_path) else base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=trust_remote_code,
    )

    logger.info("Merging LoRA adapter...")
    model = model.merge_and_unload()

    logger.info(f"Saving GGUF ({quant}) to {output_dir}...")
    model.save_pretrained_gguf(output_dir.as_posix(), tokenizer, quantization_method=quant)

    create_unsloth_model_card(
        output_dir=output_dir,
        model_name=output_dir.name,
        base_model=base_model,
        family=family,
        is_adapter=False,
        quant_method=quant,
    )
    logger.info(f"GGUF export complete at: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export trained LoRA adapter for Unsloth Studio compatibility."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained LoRA adapter directory (e.g. models/qwen3_8_27b_rust_lora)",
    )
    parser.add_argument(
        "--base",
        type=str,
        default="",
        help="Base model HF repo ID or local directory (e.g. Qwen/Qwen3.8-27B, ibm-granite/granite-4.2-8b)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["merged-hf", "adapter", "gguf"],
        default="merged-hf",
        help="Export mode: 'merged-hf' (full HF weights), 'adapter' (standalone LoRA package), or 'gguf' (GGUF file)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Destination directory. Defaults to models/<model_name>_unsloth",
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="q4_k_m",
        help="Quantization method for GGUF mode (e.g. q4_k_m, q5_k_m, q8_0, f16)",
    )
    parser.add_argument(
        "--save-dtype",
        type=str,
        choices=["bfloat16", "float16"],
        default="bfloat16",
        help="Data type for merged HF export (default: bfloat16)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Enable trust_remote_code (default: True)",
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"LoRA model path not found: {model_path}")
        sys.exit(1)

    base_model = args.base
    if not base_model:
        # Try reading base model from manifest.json or adapter_config.json
        manifest_file = model_path / "manifest.json"
        adapter_cfg = model_path / "adapter_config.json"
        if manifest_file.exists():
            with open(manifest_file, encoding="utf-8") as f:
                data = json.load(f)
                base_model = data.get("model_path", "")
        if not base_model and adapter_cfg.exists():
            with open(adapter_cfg, encoding="utf-8") as f:
                data = json.load(f)
                base_model = data.get("base_model_name_or_path", "")

    if not base_model:
        logger.error("Base model identifier is required. Specify via --base <repo_or_dir>.")
        sys.exit(1)

    family = detect_family(base_model or args.model)
    logger.info(f"Detected architecture family: {family}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = (
            "adapter" if args.mode == "adapter" else ("gguf" if args.mode == "gguf" else "merged")
        )
        output_dir = Path(f"models/{model_path.name}_{suffix}_unsloth")

    logger.info(f"Target export mode: {args.mode}")
    logger.info(f"Output directory: {output_dir}")

    if args.mode == "adapter":
        export_adapter_package(
            model_path=args.model,
            base_model=base_model,
            output_dir=output_dir,
            family=family,
        )
    elif args.mode == "merged-hf":
        export_merged_hf(
            model_path=args.model,
            base_model=base_model,
            output_dir=output_dir,
            family=family,
            save_dtype=args.save_dtype,
            max_seq_length=args.max_seq_length,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.mode == "gguf":
        export_gguf_unsloth(
            model_path=args.model,
            base_model=base_model,
            output_dir=output_dir,
            family=family,
            quant=args.quant,
            max_seq_length=args.max_seq_length,
            trust_remote_code=args.trust_remote_code,
        )


if __name__ == "__main__":
    main()
