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

BASE_MODEL_METADATA: dict[str, dict[str, str]] = {
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


def get_base_metadata(base_key: str) -> dict[str, str]:
    if base_key not in BASE_MODEL_METADATA:
        raise ValueError(
            f"Base model key must be one of: {', '.join(BASE_MODEL_METADATA)}. Got '{base_key}'."
        )
    return BASE_MODEL_METADATA[base_key]


def safe_yaml_scalar(raw: str) -> str:
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


def build_modelfile_content(
    model_name: str,
    base_metadata: dict[str, str],
    adapter_dir: Path,
    adapter_config: dict[str, object],
    base_override: str | None = None,
) -> str:
    base_reference = base_override or base_metadata["reference"]
    created_at = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    adapter_id = adapter_config.get("adapter_id", "unknown")
    adapter_files = inventory_files(adapter_dir)
    merge_command = (
        f"python scripts/10_export_ollama.py --model {adapter_dir.as_posix()} "
        f"--name {model_name} --base {base_reference}"
    )

    lines: list[str] = [
        "format_version: '1.0'",
        "artifact_type: loRA_adapter_metadata",
        f"name: {safe_yaml_scalar(model_name)}",
        f"created_at: {safe_yaml_scalar(created_at)}",
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
        "  instruction: Rust programming, systems, async Rust, embedded Rust, and library usage.",
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

    modelfile_path = output_path / "Modelfile"
    print(f"Writing Modelfile to {modelfile_path}...")
    modelfile_content = build_modelfile_content(
        model_name,
        base_metadata,
        adapter_dir,
        adapter_config,
        base_override,
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
