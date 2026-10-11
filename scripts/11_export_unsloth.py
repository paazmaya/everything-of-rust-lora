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
from typing import Any, cast

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rust_lora.unsloth_export")


TOKEN_IM_END = "<|im_end|>"
TOKEN_ENDOFTEXT = "<|endoftext|>"

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
        "{{- '<|ifm|im_start|>assistant\\n<ifm|think>\\n' }}"
        "{%- endif %}"
    ),
    "nemotron": (
        "{%- for message in messages %}"
        "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '"
        + TOKEN_IM_END
        + "\\n' }}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "{{- '<|im_start|>assistant\\n<think>\\n' }}"
        "{%- endif %}"
    ),
    "qwen": (
        "{%- if messages[0]['role'] == 'system' %}"
        "{{- '<|im_start|>system\\n' + messages[0]['content'] + '" + TOKEN_IM_END + "\\n' }}"
        "{%- set messages = messages[1:] %}"
        "{%- else %}"
        "{{- '<|im_start|>system\\nYou are an expert Rust programming assistant."
        + TOKEN_IM_END
        + "\\n' }}"
        "{%- endif %}"
        "{%- for message in messages %}"
        "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '"
        + TOKEN_IM_END
        + "\\n' }}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "{{- '<|im_start|>assistant\\n' }}"
        "{%- if enable_thinking is not defined or enable_thinking %}"
        "{{- '<think>\\n' }}"
        "{%- endif %}"
        "{%- endif %}"
    ),
    "qwen3_8": (
        "{%- if not messages %}{{- raise_exception('No messages provided.') }}{%- endif %}"
        "{%- if messages[0]['role'] == 'system' %}"
        "{{- '<|im_start|>system\\n' + messages[0]['content'] + '" + TOKEN_IM_END + "\\n' }}"
        "{%- set messages = messages[1:] %}"
        "{%- endif %}"
        "{%- for message in messages %}"
        "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '"
        + TOKEN_IM_END
        + "\\n' }}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "{{- '<|im_start|>assistant\\n<think>\\n' }}"
        "{%- endif %}"
    ),
}

DEFAULT_STOP_TOKENS: dict[str, list[str]] = {
    "granite": ["<|end_of_text|>", "<|start_of_role|>"],
    "k2_horizon": ["<|ifm|im_end|>", "<|ifm|endoftext|>", TOKEN_IM_END],
    "nemotron": [TOKEN_IM_END, "<extra_id_1>", "<extra_id_2>"],
    "qwen": [TOKEN_IM_END, TOKEN_ENDOFTEXT],
    "qwen3_8": [TOKEN_IM_END, TOKEN_ENDOFTEXT, "<|im_start|>"],
}

# Unsloth-optimized default generation configurations per architectural family
FAMILY_GENERATION_CONFIGS: dict[str, dict[str, Any]] = {
    "granite": {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 4096,
        "stop_strings": ["<|end_of_text|>", "<|start_of_role|>"],
        "notes": "Deterministic instruction following recommended by Unsloth Granite documentation.",
    },
    "k2_horizon": {
        "temperature": 0.6,
        "top_p": 0.95,
        "min_p": 0.01,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "max_new_tokens": 8192,
        "stop_strings": ["<|ifm|im_end|>", "<|ifm|endoftext|>", TOKEN_IM_END],
        "notes": "Native INT4 base architecture; min_p=0.01 suppresses low-probability outlier tokens.",
    },
    "nemotron": {
        "temperature": 0.6,
        "top_p": 0.95,
        "repetition_penalty": 1.05,
        "max_new_tokens": 8192,
        "stop_strings": [TOKEN_IM_END, "<extra_id_1>", "<extra_id_2>"],
        "notes": "Hybrid MoE with NoPE architecture; uses <think> token ID 12 and </think> token ID 13.",
    },
    "qwen": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "enable_thinking": True,
        "max_new_tokens": 8192,
        "stop_strings": [TOKEN_IM_END, TOKEN_ENDOFTEXT],
        "notes": "Precise coding preset; reasoning is enabled explicitly via template kwargs.",
    },
    "qwen3_8": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "reasoning_effort": "medium",
        "preserve_thinking": True,
        "max_new_tokens": 16384,
        "stop_strings": [TOKEN_IM_END, TOKEN_ENDOFTEXT, "<|im_start|>"],
        "notes": "Hybrid thinking architecture with MTP speculative decoding and reasoning_effort control.",
    },
}

# Unsloth Dynamic 3.0 / 2.0 quantization profiles
DYNAMIC_QUANT_PRESETS = [
    "ud_q4_k_xl",
    "ud_q3_k_xl",
    "ud_q2_k_xl",
    "ud_iq2_s",
    "ud_iq1_s",
    "q4_k_m",
    "q5_k_m",
    "q8_0",
    "f16",
    "bf16",
]


def get_cutoff_date() -> str:
    """Read knowledge cutoff date from config/cybersec.yaml."""
    cfg_file = Path(__file__).parent.parent / "config" / "cybersec.yaml"
    if not cfg_file.exists():
        return "2026-10-11"
    try:
        with open(cfg_file, encoding="utf-8") as f:
            cfg = cast(dict[str, Any], yaml.safe_load(f) or {})
        raw_meta = cfg.get("metadata")
        meta = cast(dict[str, Any], raw_meta if isinstance(raw_meta, dict) else {})
        cutoff_val = meta.get("cutoff_date", "2026-10-11")
        return str(cutoff_val)
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
    """Write an Unsloth Studio compatible README.md model card with architecture-specific runtime guidance."""
    cutoff_date = get_cutoff_date()
    quant_str = f" ({quant_method})" if quant_method else ""
    artifact_kind = "LoRA Adapter" if is_adapter else f"Merged Model{quant_str}"

    tags = ["unsloth", "rust", "rust-lang", "systems-programming", "cybersecurity", "lotl"]
    if is_adapter:
        tags.extend(["peft", "lora"])
    if quant_method:
        tags.append("gguf")

    family_cfg = FAMILY_GENERATION_CONFIGS.get(family, FAMILY_GENERATION_CONFIGS["qwen"])

    runtime_guidance = ""
    if family == "qwen3_8":
        runtime_guidance = """### Qwen 3.8 27B Specific Optimizations
- **Multi-Token Prediction (MTP):** Run with speculative decoding for 1.4x-2.2x speedup:
  ```bash
  llama-server --model <model.gguf> --spec-type draft-mtp --spec-draft-n-max 2
  ```
- **Reasoning Effort Control:** Set reasoning depth in Unsloth Studio or llama-server:
  ```bash
  --chat-template-kwargs '{"reasoning_effort":"medium"}'  # options: xhigh, medium, low, none
  ```
- **Dynamic 3.0 Quantization:** For agentic Rust coding, `UD-Q4_K_XL` (20.7GB) or `UD-Q2_K_XL` (9.8GB) is recommended. Avoid 1-bit quants as Divergence-300 @32 drops below 10%."""
    elif family == "qwen":
        runtime_guidance = """### Qwen 3.5 4B Specific Optimizations
- **Thinking Mode Activation:** On compact Qwen 3.5 models (0.8B - 9B), reasoning is disabled by default. Enable via:
  ```bash
  --chat-template-kwargs '{"enable_thinking":true}'
  ```
- **Coding Sampling:** Recommended settings for precise Rust code generation:
  `temperature=0.6, top_p=0.95, top_k=20, min_p=0.0`
- **Protected Tensors:** Keep `ssm_out` and attention gates out of low-bit quantization."""
    elif family == "nemotron":
        runtime_guidance = """### Nemotron 3 Nano 4B Specific Optimizations
- **NoPE Positional Embeddings:** Context window expands up to 1M tokens without RoPE / YaRN scaling.
- **Reasoning Tokens:** Nemotron 3 uses `<think>` (ID 12) and `</think>` (ID 13). Use `--special` to view tokens in llama.cpp:
  ```bash
  llama-cli --model <model.gguf> --special --ctx-size 32768 --temp 0.6 --top-p 0.95
  ```
- **Stop Tokens:** `<|im_end|>`, `<extra_id_1>`, `<extra_id_2>`"""
    elif family == "k2_horizon":
        runtime_guidance = """### IFM K2-Horizon 3.7B Specific Optimizations
- **MoE CPU Expert Offloading:** Offload sparse expert MLPs to system RAM while keeping attention in VRAM:
  ```bash
  llama-cli --model <model.gguf> -ot ".ffn_.*_exps.=CPU" --temp 0.6 --min-p 0.01
  ```
- **Native INT4 Base Precision:** Preserved accurately in `UD-Q4_K_XL` GGUFs."""
    elif family == "granite":
        runtime_guidance = """### IBM Granite 4.2 8B Specific Optimizations
- **Deterministic Sampling:** Granite instruction models excel with zero-temperature greedy generation:
  `temperature=0.0, top_k=0, top_p=1.0`
- **Native Context Window:** 131,072 tokens."""

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
- **Default Sampling:** `temperature={family_cfg.get("temperature", 0.6)}, top_p={family_cfg.get("top_p", 0.95)}`

{runtime_guidance}

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
    """Ensure generation_config.json exists with family-specific recommended sampling settings."""
    config_path = output_dir / "generation_config.json"
    if config_path.exists():
        return

    family_cfg = FAMILY_GENERATION_CONFIGS.get(family, FAMILY_GENERATION_CONFIGS["qwen"])
    config_data = {k: v for k, v in family_cfg.items() if k != "notes"}
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    logger.info(f"Created family-optimized generation_config.json ({family}) at {config_path}")


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

        auto_tok: Any = AutoTokenizer
        tokenizer: Any = auto_tok.from_pretrained(base_model, trust_remote_code=True)
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
        from unsloth import FastLanguageModel  # type: ignore[import-untyped]

        logger.info(f"Loading LoRA and base model via Unsloth from {model_path} / {base_model}...")
        fast_lm: Any = FastLanguageModel
        model, tokenizer = fast_lm.from_pretrained(
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
        auto_tok_fb: Any = AutoTokenizer
        tokenizer_fallback: Any = auto_tok_fb.from_pretrained(
            base_model, trust_remote_code=trust_remote_code
        )
        auto_model: Any = AutoModelForCausalLM
        base_causal_model: Any = auto_model.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        logger.info(f"Applying LoRA adapter from {model_path}...")
        peft_cls: Any = PeftModel
        peft_model: Any = peft_cls.from_pretrained(base_causal_model, model_path)
        logger.info("Merging LoRA weights into base model...")
        merged_model: Any = peft_model.merge_and_unload()
        logger.info(f"Saving merged model weights to {output_dir}...")
        merged_model.save_pretrained(output_dir)
        tokenizer_fallback.save_pretrained(output_dir)

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
    """Merge LoRA and export to GGUF using Unsloth native save_pretrained_gguf with Dynamic 3.0 checks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    quant_lower = quant.lower()

    # Dynamic 3.0 Divergence-300 @32 check
    if any(k in quant_lower for k in ["iq1", "tq1", "q1_0", "1bit", "1_bit"]):
        logger.warning(
            "⚠️  QUANTIZATION WARNING (Unsloth Dynamic 3.0): You selected a 1-bit quantization variant (%s). "
            "Unsloth Divergence-300 @32 benchmarks demonstrate that 1-bit quants suffer from sharp accuracy drops "
            "(<8-10%% trajectory retention), tool-calling failure, and endless generation loops. "
            "For Rust coding and agentic workflows, UD-Q2_K_XL or UD-Q4_K_XL is strongly recommended.",
            quant,
        )

    # Check for MTP auxiliary layers
    base_path = Path(base_model) if os.path.isdir(base_model) else None
    if base_path and (base_path / "model-mtp-restored.safetensors").exists():
        logger.info(
            "⚡ Multi-Token Prediction (MTP) weights detected! "
            "Ensure MTP projection layers remain at Q8_0 or FP16 to retain >55-65%% speculative acceptance rates."
        )

    from unsloth import FastLanguageModel  # type: ignore[import-untyped]

    logger.info(f"Loading LoRA/base for GGUF export ({quant}) via Unsloth...")
    fast_lm_gguf: Any = FastLanguageModel
    model, tokenizer = fast_lm_gguf.from_pretrained(
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


def resolve_base_model(model_path: Path, base_arg: str) -> str:
    """Resolve base model identifier from CLI argument or adapter metadata."""
    if base_arg:
        return base_arg
    manifest_file = model_path / "manifest.json"
    if manifest_file.exists():
        with open(manifest_file, encoding="utf-8") as f:
            data = json.load(f)
            base_model = data.get("model_path", "")
            if base_model:
                return str(base_model)
    adapter_cfg = model_path / "adapter_config.json"
    if adapter_cfg.exists():
        with open(adapter_cfg, encoding="utf-8") as f:
            data = json.load(f)
            return str(data.get("base_model_name_or_path", ""))
    return ""


def determine_output_dir(model_name: str, mode: str, explicit_output_dir: str) -> Path:
    """Determine output directory based on export mode or explicit flag."""
    if explicit_output_dir:
        return Path(explicit_output_dir)
    suffix_map = {
        "adapter": "adapter",
        "gguf": "gguf",
        "merged-hf": "merged",
    }
    suffix = suffix_map.get(mode, "merged")
    return Path(f"models/{model_name}_{suffix}_unsloth")


def execute_export(
    args: argparse.Namespace, base_model: str, output_dir: Path, family: str
) -> None:
    """Dispatch export operation based on selected mode."""
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
        help="Quantization method for GGUF mode (e.g. ud_q4_k_xl, ud_q2_k_xl, q4_k_m, q5_k_m, q8_0, f16)",
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

    base_model = resolve_base_model(model_path, args.base)
    if not base_model:
        logger.error("Base model identifier is required. Specify via --base <repo_or_dir>.")
        sys.exit(1)

    family = detect_family(base_model or args.model)
    logger.info(f"Detected architecture family: {family}")

    output_dir = determine_output_dir(model_path.name, args.mode, args.output_dir)
    logger.info(f"Target export mode: {args.mode}")
    logger.info(f"Output directory: {output_dir}")

    execute_export(args, base_model, output_dir, family)


if __name__ == "__main__":
    main()
