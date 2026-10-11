#!/usr/bin/env python3
"""
Create a high-quality Modelfile for a trained LoRA adapter checkpoint.

This script does not modify or export adapter weights. It validates the adapter
checkpoint directory, reads the adapter config, and writes a structured Modelfile
for later merging and GGUF export with script 10.

The generated Modelfile includes:
- Base model metadata
- Training economics and prompt guidance
- LoRA adapter metadata and file inventory
- Recommended merge command for script 10
"""

import argparse
import datetime
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ADAPTER_WEIGHTS = "adapter_model.safetensors"

BASE_MODEL_METADATA: dict[str, dict[str, str]] = {
    "granite": {
        "name": "IBM Granite 4.2 8B",
        "reference": "ibm-granite/granite-4.2-8b",
        "gguf": "ibm-granite/granite-4.2-8b-GGUF",
        "source_url": "https://huggingface.co/ibm-granite/granite-4.2-8b-GGUF",
        "training_economics": (
            "LoRA rank=64, alpha=128, optimizer=AdamW 8-bit, scheduler=cosine, "
            "batch_size=1, gradient_accumulation_steps=8 (effective=8), epochs=3, "
            "VRAM≈16GB, training time≈2-3 hours on a 24GB GPU. "
            "Supports post-training SVD rank truncation (r=64 -> r=16) for 75% smaller adapters."
        ),
        "notes": (
            "This adapter was trained on a Rust-specific corpus targeting systems, "
            "async, and embedded Rust. Recommended Unsloth inference parameters: "
            "temperature=0.0, top_k=0, top_p=1.0 (deterministic instruction following). "
            "Native context window: 131,072 tokens."
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
            "This adapter is designed for Qwen 3.5 4B using the ChatML style. "
            "Recommended coding parameters: temperature=0.6, top_p=0.95, top_k=20, min_p=0.0. "
            "Note: Qwen 3.5 compact models (0.8B-9B) disable reasoning by default; "
            "enable via --chat-template-kwargs '{\"enable_thinking\":true}' in llama-server. "
            "Native context window: 262,144 tokens."
        ),
    },
    "qwen3_8": {
        "name": "Qwen 3.8 27B Instruct",
        "reference": "Qwen/Qwen3.8-27B",
        "gguf": "unsloth/Qwen3.8-27B-GGUF",
        "source_url": "https://huggingface.co/Qwen/Qwen3.8-27B",
        "training_economics": (
            "LoRA rank=16, alpha=16, optimizer=AdamW (torch/8-bit), scheduler=cosine, "
            "batch_size=1, gradient_accumulation_steps=16, epochs=3, "
            "supports Apple Silicon MPS with offload and 24-80GB CUDA workstations."
        ),
        "notes": (
            "Targets the 64-layer hybrid architecture of Qwen 3.8 27B. "
            "Supports Multi-Token Prediction (MTP) speculative decoding (--spec-type draft-mtp --spec-draft-n-max 2), "
            "reasoning_effort controls ('xhigh', 'medium', 'low', 'none'), and Unsloth Dynamic 3.0 quants "
            "(UD-Q4_K_XL / UD-Q2_K_XL). Native context window: 262,144 tokens."
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
            "Hybrid MoE model with NoPE (No Positional Embeddings), allowing context expansion up to 1M tokens. "
            "Special reasoning tokens: <think> (ID 12) and </think> (ID 13) requiring --special flag. "
            "Recommended coding parameters: temperature=0.6, top_p=0.95."
        ),
    },
    "k2_horizon": {
        "name": "IFM K2-Horizon 3.7B",
        "reference": "IFM/K2-Horizon-3.7B",
        "gguf": "IFM/K2-Horizon-3.7B-GGUF",
        "source_url": "https://huggingface.co/IFM/K2-Horizon-3.7B",
        "training_economics": (
            "LoRA rank=16, alpha=16, optimizer=AdamW 8-bit, scheduler=cosine, "
            "batch_size=1, gradient_accumulation_steps=4 (effective=4), epochs=3, "
            "VRAM≈12-24GB, dense 3.7B model with native 512K context."
        ),
        "notes": (
            "Targets IFM/K2-Horizon-3.7B based on the Kimi K2 architecture. "
            'Supports MoE CPU expert offloading (-ot ".ffn_.*_exps.=CPU"). '
            "Native INT4 base precision is preserved in UD-Q4_K_XL quants. "
            "Recommended parameters: temperature=0.6, min_p=0.01, top_p=0.95."
        ),
    },
}


def get_base_metadata(base_key: str) -> dict[str, str]:
    normalized_key = (
        "k2_horizon" if base_key.lower() in ("k2", "k2-horizon", "k2_horizon") else base_key.lower()
    )
    if normalized_key not in BASE_MODEL_METADATA:
        raise ValueError(
            f"Base model key must be one of: {', '.join(BASE_MODEL_METADATA)}. Got '{base_key}'."
        )
    return BASE_MODEL_METADATA[normalized_key]


def safe_yaml_scalar(raw: Any) -> str:
    text = str(raw)
    if "\n" in text:
        escaped = text.replace("'", "''")
        return "|\n  " + escaped.replace("\n", "\n  ")
    if text == "" or text.strip() != text or any(c in text for c in ":#\"'"):
        escaped = text.replace("'", "''")
        return f"'{escaped}'"
    return text


def yaml_list(items: list[str], indent: int = 2) -> list[str]:
    return [" " * indent + f"- {safe_yaml_scalar(item)}" for item in items]


def load_adapter_config(adapter_dir: Path) -> dict[str, object]:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing adapter_config.json in LoRA adapter directory: {adapter_dir}"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def inventory_files(adapter_dir: Path) -> list[str]:
    return sorted(
        [
            str(path.name)
            for path in adapter_dir.iterdir()
            if path.is_file() and not path.name.startswith(".")
        ]
    )


def format_adapter_config(adapter_config: dict[str, object]) -> list[str]:
    parts: list[str] = []
    for key in [
        "r",
        "lora_alpha",
        "lora_dropout",
        "bias",
        "fan_in_fan_out",
        "target_modules",
    ]:
        if key in adapter_config:
            value = adapter_config[key]
            if isinstance(value, list):
                parts.append(f"  {key}:")
                parts.extend(yaml_list([str(v) for v in value], indent=4))
            else:
                parts.append(f"  {key}: {safe_yaml_scalar(value)}")
    return parts


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


def build_modelfile_content(
    model_name: str,
    base_metadata: dict[str, str],
    adapter_dir: Path,
    adapter_config: dict[str, object],
    base_override: str | None = None,
) -> str:
    base_reference = base_override or base_metadata["reference"]
    created_at = datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()
    adapter_id = str(adapter_config.get("adapter_id", "unknown"))
    adapter_files = inventory_files(adapter_dir)
    merge_command = (
        f"uv run python scripts/10_export_ollama.py --model {adapter_dir.as_posix()} "
        f"--name {model_name} --base {base_reference}"
    )

    cutoff_date = get_cutoff_date()

    lines: list[str] = [
        "format_version: '1.0'",
        "artifact_type: loRA_adapter_metadata",
        f"name: {safe_yaml_scalar(model_name)}",
        f"created_at: {safe_yaml_scalar(created_at)}",
        f"knowledge_cutoff_date: {safe_yaml_scalar(cutoff_date)}",
        "",
        "base_model:",
        f"  name: {safe_yaml_scalar(base_metadata['name'])}",
        f"  reference: {safe_yaml_scalar(base_reference)}",
        f"  gguf: {safe_yaml_scalar(base_metadata['gguf'])}",
        f"  source_url: {safe_yaml_scalar(base_metadata['source_url'])}",
        "",
        "training_economics:",
        f"  summary: {safe_yaml_scalar(base_metadata['training_economics'])}",
        "  dataset: data/datasets/train.jsonl",
        "  instruction: Rust programming, systems, async Rust, embedded Rust, cybersecurity, RustSec advisories, and Kali/LOTL tooling.",
        f"  cutoff_date: {safe_yaml_scalar(cutoff_date)}",
        "",
        "adapter:",
        f"  path: {safe_yaml_scalar(adapter_dir.as_posix())}",
        f"  id: {safe_yaml_scalar(adapter_id)}",
        "  type: adapter-only",
        "  notes: This directory contains only the LoRA adapter checkpoint. Do not quantize it directly.",
    ]

    config_lines = format_adapter_config(adapter_config)
    if config_lines:
        lines.append("  config:")
        lines.extend(config_lines)

    if adapter_files:
        lines.append("  files:")
        lines.extend(yaml_list(adapter_files, indent=4))

    lines.extend(
        [
            "",
            "merge:",
            "  script: scripts/10_export_ollama.py",
            f"  recommended_command: {safe_yaml_scalar(merge_command)}",
            "  quantization: q4_k_m",
            "  output_dir: models/{model_name}_gguf",
            "",
            "notes:",
            f"  - {safe_yaml_scalar(base_metadata['notes'])}",
            "  - 'To create a runnable GGUF model, merge this adapter with the base model using script 10.'",
            "  - 'The Modelfile is a metadata manifest, not a runnable GGUF model itself.'",
        ]
    )

    return "\n".join(lines) + "\n"


def build_ollama_modelfile_content(
    base_reference: str,
    base_key: str,
    adapter_filename: str = DEFAULT_ADAPTER_WEIGHTS,
) -> str:
    """Build a syntactically valid Ollama Modelfile with ADAPTER instruction."""
    cutoff_date = get_cutoff_date()
    system_prompt = (
        f"You are an expert Rust programmer and cybersecurity engineer specializing in systems programming, "
        f"async Rust, ESP32 embedded development, RustSec vulnerability analysis, Kali Linux tooling, and "
        f"living-off-the-land (LOTL) security. Your cybersecurity, vulnerability, and tooling knowledge cutoff "
        f"date is {cutoff_date}. For vulnerabilities discovered after this date, advise running cargo-audit."
    )

    norm_key = base_key.lower()
    if norm_key in ("k2", "k2-horizon", "k2_horizon"):
        return f'''FROM {base_reference}
ADAPTER {adapter_filename}
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER min_p 0.01
PARAMETER num_ctx 65536
PARAMETER stop "<|ifm|im_end|>"
PARAMETER stop "<|ifm|endoftext|>"
PARAMETER stop "<|im_end|>"
TEMPLATE """{{{{ if .System }}}}<|ifm|im_start|>system
{{{{ .System }}}}<|ifm|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|ifm|im_start|>user
{{{{ .Prompt }}}}<|ifm|im_end|>
{{{{ end }}}}<|ifm|im_start|>assistant
<ifm|think>
</ifm|think>{{{{ .Response }}}}<|ifm|im_end|>"""
SYSTEM """{system_prompt}"""
'''
    elif norm_key == "granite":
        return f'''FROM {base_reference}
ADAPTER {adapter_filename}
PARAMETER temperature 0.0
PARAMETER top_p 1.0
PARAMETER top_k 0
PARAMETER num_ctx 32768
PARAMETER stop "<|end_of_text|>"
PARAMETER stop "<|start_of_role|>"
TEMPLATE """{{{{- range .Messages }}}}
{{{{- if eq .Role "system" }}}}<|start_of_role|>system<|end_of_role|>{{{{ .Content }}}}<|end_of_text|>
{{{{- else if eq .Role "user" }}}}<|start_of_role|>user<|end_of_role|>{{{{ .Content }}}}<|end_of_text|>
{{{{- else if eq .Role "assistant" }}}}<|start_of_role|>assistant<|end_of_role|>{{{{ .Content }}}}<|end_of_text|>
{{{{- end }}}}
{{{{- end }}}}
{{{{- if .Prompt }}}}<|start_of_role|>user<|end_of_role|>{{{{ .Prompt }}}}<|end_of_text|><|start_of_role|>assistant<|end_of_role|>{{{{- end }}}}"""
SYSTEM """{system_prompt}"""
'''
    elif norm_key == "nemotron":
        return f'''FROM {base_reference}
ADAPTER {adapter_filename}
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER num_ctx 65536
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<extra_id_1>"
PARAMETER stop "<extra_id_2>"
TEMPLATE """{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>"""
SYSTEM """{system_prompt}"""
'''
    elif norm_key == "qwen3_8":
        return f'''FROM {base_reference}
ADAPTER {adapter_filename}
PARAMETER temperature 1.0
PARAMETER top_p 0.95
PARAMETER top_k 20
PARAMETER min_p 0.0
PARAMETER num_ctx 65536
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
TEMPLATE """{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>"""
SYSTEM """{system_prompt}"""
'''
    else:  # qwen 3.5 4b
        return f'''FROM {base_reference}
ADAPTER {adapter_filename}
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER top_k 20
PARAMETER min_p 0.0
PARAMETER num_ctx 65536
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
TEMPLATE """{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>"""
SYSTEM """{system_prompt}"""
'''


def create_modelfile(
    adapter_dir: Path,
    model_name: str,
    base_key: str,
    base_override: str | None = None,
    output_path: Path | None = None,
) -> Path:
    if not adapter_dir.is_dir():
        raise ValueError(f"Adapter path must be a directory: {adapter_dir}")

    adapter_config = load_adapter_config(adapter_dir)
    base_metadata = get_base_metadata(base_key)
    output_path = output_path or adapter_dir
    output_path.mkdir(parents=True, exist_ok=True)

    base_reference = base_override or base_metadata["reference"]

    # 1. Write YAML manifest
    manifest_path = output_path / "lora_manifest.yaml"
    print(f"Writing LoRA YAML metadata manifest to {manifest_path}...")
    manifest_content = build_modelfile_content(
        model_name,
        base_metadata,
        adapter_dir,
        adapter_config,
        base_override,
    )
    manifest_path.write_text(manifest_content, encoding="utf-8")

    # 2. Write executable Ollama Modelfile with ADAPTER
    modelfile_path = output_path / "Modelfile"
    print(f"Writing executable Ollama Modelfile to {modelfile_path}...")
    adapter_file = DEFAULT_ADAPTER_WEIGHTS
    if not (adapter_dir / adapter_file).exists():
        # Fallback to .bin or search for any adapter weight file
        for candidate in ["adapter_model.bin", DEFAULT_ADAPTER_WEIGHTS]:
            if (adapter_dir / candidate).exists():
                adapter_file = candidate
                break

    modelfile_content = build_ollama_modelfile_content(
        base_reference=base_reference,
        base_key=base_key,
        adapter_filename=adapter_file,
    )
    modelfile_path.write_text(modelfile_content, encoding="utf-8")

    return modelfile_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create a high-quality Modelfile for a trained LoRA adapter checkpoint. "
            "This script does not export or merge weights."
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
        help="Name for the Modelfile package (e.g., rust-granite)",
    )
    parser.add_argument(
        "--base",
        type=str,
        choices=list(BASE_MODEL_METADATA.keys()),
        default="granite",
        help=(
            f"Base model family for this adapter. Choices: {', '.join(BASE_MODEL_METADATA.keys())}. "
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
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=("Optional output directory for the Modelfile. Defaults to the adapter directory."),
    )
    args = parser.parse_args()

    adapter_dir = Path(args.model)
    output_dir = Path(args.output) if args.output else None
    modelfile_path = create_modelfile(
        adapter_dir,
        args.name,
        args.base,
        args.base_model,
        output_dir,
    )

    print("\nModelfile creation complete!")
    print(f"Modelfile saved to: {modelfile_path}")
    print("Use script 10 to merge the adapter with the base model and export GGUF.")
