#!/usr/bin/env python3
import json
import random
from pathlib import Path
from typing import Any

from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATASETS_DIR = BASE_DIR / "data" / "datasets"


def _format_changelog(library: str, version: str, title: str) -> tuple[str, str]:
    instruction = f"Explain the changes and migration details for the Rust crate '{library}' in version {version}."
    input_text = f"Crate: {library}\nVersion: {version}\nTopic: {title}"
    return instruction, input_text


def _format_docs_rs(library: str, version: str, title: str) -> tuple[str, str]:
    if version != "current":
        instruction = f"Explain the Rust crate '{library}' (version {version}) and its usage based on the documentation."
        input_text = f"Crate: {library}\nVersion: {version}\nTopic: {title}"
    else:
        instruction = (
            f"Explain the Rust crate '{library}' and its usage based on the documentation."
        )
        input_text = f"Crate: {library}\nTopic: {title}"
    return instruction, input_text


def _format_github(library: str, version: str, title: str, meta: dict[str, Any]) -> tuple[str, str]:
    repo = str(meta.get("repo", library))
    if version != "current":
        instruction = f"Explain the implementation, architecture, or usage of '{library}' (version {version}) from repository documentation."
        input_text = f"Repository: {repo}\nVersion: {version}\nTopic: {title}"
    else:
        instruction = f"Explain the implementation, architecture, or usage of '{library}' from repository documentation."
        input_text = f"Repository: {repo}\nTopic: {title}"
    return instruction, input_text


def _format_official_docs(version: str, title: str) -> tuple[str, str]:
    if version in ["2018", "2021", "2024"]:
        instruction = f"Explain this Rust concept or language feature according to the Rust {version} Edition."
        input_text = f"Edition: {version}\nTopic: {title}"
    else:
        instruction = "Explain this Rust concept or standard library feature based on official Rust documentation."
        input_text = f"Topic: {title}"
    return instruction, input_text


def _format_generic(library: str, version: str, title: str, source: str) -> tuple[str, str]:
    if source == "esp_rs":
        return (
            "Provide details about ESP32 and embedded Rust development.",
            f"Platform: ESP-RS\nTopic: {title}",
        )
    if version not in ["current", "recent"]:
        instruction = f"Explain this Rust concept or best practice from {library} ({version})."
        input_text = f"Source: {library}\nRelease/Date: {version}\nTopic: {title}"
    else:
        instruction = f"Explain this Rust concept or best practice from {library}."
        input_text = f"Source: {library}\nTopic: {title}"
    return instruction, input_text


def format_alpaca(chunk: dict[str, Any]) -> dict[str, str] | None:
    content = str(chunk.get("content", ""))
    if len(content) < 50:
        return None

    source = str(chunk.get("source_type", ""))
    meta = chunk.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    library = str(chunk.get("library") or meta.get("crate_name") or meta.get("library") or "Rust")
    version = str(chunk.get("version") or meta.get("version") or "current")
    title = str(chunk.get("title") or meta.get("title") or "Documentation")
    is_changelog = bool(meta.get("is_changelog_section", False) or "CHANGELOG" in title)

    if is_changelog:
        instruction, input_text = _format_changelog(library, version, title)
    elif source == "docs_rs":
        instruction, input_text = _format_docs_rs(library, version, title)
    elif source == "github":
        instruction, input_text = _format_github(library, version, title, meta)
    elif source in ["rust_book", "official_docs"]:
        instruction, input_text = _format_official_docs(version, title)
    else:
        instruction, input_text = _format_generic(library, version, title, source)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": content,
    }


def main() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    chunks_file = PROCESSED_DIR / "all_chunks.jsonl"

    print("Creating Alpaca dataset with version-aware instruction prompts...")
    dataset: list[dict[str, str]] = []
    total_chunks = 0
    with open(chunks_file) as f:
        for line in tqdm(f):
            total_chunks += 1
            chunk: dict[str, Any] = json.loads(line)
            formatted = format_alpaca(chunk)
            if formatted:
                dataset.append(formatted)

    random.shuffle(dataset)
    split = int(0.98 * len(dataset))

    with open(DATASETS_DIR / "train.jsonl", "w") as f:
        for item in dataset[:split]:
            f.write(json.dumps(item) + "\n")
    with open(DATASETS_DIR / "val.jsonl", "w") as f:
        for item in dataset[split:]:
            f.write(json.dumps(item) + "\n")

    print(
        f"Converted {total_chunks} chunks into {len(dataset)} dataset samples. Train: {split}, Val: {len(dataset) - split}"
    )


if __name__ == "__main__":
    main()
