#!/usr/bin/env python3
"""
Create instruction-tuning datasets (train.jsonl / val.jsonl) formatted for Alpaca / ChatML.
Features multi-angle prompt synthesis for:
1. Consecutive minor version diffs (e.g. 1.80.0 -> 1.81.0)
2. Rust edition migrations (2015, 2018, 2021, 2024)
3. Technical rustc release notes (Language, Stabilized APIs, Compiler, Cargo, Compatibility)
4. Reverse-lookup micro Q&A pairs for stabilized APIs
5. Stratified sampling boosting version knowledge to 10-15% of the total dataset mix.
"""

import json
import random
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATASETS_DIR = BASE_DIR / "data" / "datasets"

TARGET_VERSION_RATIO = 0.12  # 12% representation for version & edition knowledge


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


def _format_edition_docs(version: str, title: str) -> tuple[str, str]:
    if version == "2015":
        templates = [
            f"Explain the Rust 2015 Edition, its foundational stability guarantees, and how '{title}' applies to Rust 1.0.",
            f"What does the Rust 2015 Edition establish regarding '{title}' and backwards compatibility?",
        ]
        return random.choice(templates), f"Edition: 2015\nTopic: {title}"
    elif version == "2018":
        templates = [
            f"Explain the language changes, module system improvements, and idioms introduced in the Rust 2018 Edition regarding '{title}'.",
            f"How did Rust 2018 evolve from Rust 2015 in terms of '{title}' and developer ergonomics?",
        ]
        return random.choice(templates), f"Edition: 2018\nTopic: {title}"
    elif version == "2021":
        templates = [
            f"Explain the Rust 2021 Edition updates, prelude changes, closure disjoint captures, and migration guidelines for '{title}'.",
            f"What are the key behavioral changes and upgrade instructions in Rust 2021 regarding '{title}'?",
        ]
        return random.choice(templates), f"Edition: 2021\nTopic: {title}"
    elif version == "2024":
        templates = [
            f"Explain the Rust 2024 Edition changes, language features, breaking modifications, and transition strategies for '{title}'.",
            f"What changes in compiler semantics, syntax, and standard prelude were introduced in Rust 2024 for '{title}'?",
        ]
        return random.choice(templates), f"Edition: 2024\nTopic: {title}"
    else:
        return (
            "Explain this Rust concept or standard library feature based on official Rust documentation.",
            f"Topic: {title}",
        )


def _format_version_diff(from_ver: str, to_ver: str, title: str) -> tuple[str, str]:
    templates = [
        (
            f"What changes, language features, compiler updates, and stabilized APIs were introduced between Rust {from_ver} and Rust {to_ver}?",
            f"From Version: {from_ver}\nTo Version: {to_ver}\nTopic: {title}",
        ),
        (
            f"Explain the technical delta, newly stabilized library features, and migration considerations when upgrading from Rust {from_ver} to {to_ver}.",
            f"Version Transition: {from_ver} -> {to_ver}\nTopic: {title}",
        ),
        (
            f"Compare Rust {from_ver} with Rust {to_ver} and detail all language, standard library, and tooling updates.",
            f"Compiler: rustc\nUpgrade: {from_ver} -> {to_ver}",
        ),
    ]
    return random.choice(templates)


def _format_rustc_release(
    version: str, title: str, section: str | None, post_type: str | None
) -> tuple[str, str]:
    if post_type == "release_announcement":
        instruction = f"Explain the key highlights, code examples, and feature advancements announced in the official Rust {version} release."
        input_text = f"Release: Rust {version}\nAnnouncement: {title}"
    elif section == "Language":
        instruction = f"What language features, syntax additions, and core compiler semantics were stabilized or changed in Rust {version}?"
        input_text = f"Compiler: rustc\nVersion: {version}\nCategory: Language Features"
    elif section in ["Stabilized APIs", "Libraries"]:
        instruction = (
            f"Which standard library APIs, methods, and traits were stabilized in Rust {version}?"
        )
        input_text = (
            f"Compiler: rustc\nVersion: {version}\nCategory: Stabilized Standard Library APIs"
        )
    elif section == "Compiler":
        instruction = f"What compiler flags, optimization changes, target promotions, and diagnostic updates were introduced in rustc {version}?"
        input_text = f"Compiler: rustc\nVersion: {version}\nCategory: Compiler & Tooling"
    elif section == "Cargo":
        instruction = f"What new Cargo configuration options, CLI flags, and package management updates were added in Rust {version}?"
        input_text = f"Cargo Version: {version}\nCategory: Cargo Tooling"
    elif section in ["Compatibility Notes", "Breaking Changes"]:
        instruction = f"What compatibility notes, behavior adjustments, or potential breaking changes should be considered in Rust {version}?"
        input_text = f"Compiler: rustc\nVersion: {version}\nCategory: Compatibility Notes"
    else:
        instruction = (
            f"Explain the official technical release notes and updates for Rust {version}."
        )
        input_text = f"Release: Rust {version}\nTopic: {title}"
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


def extract_micro_qa_pairs(chunk: dict[str, Any]) -> list[dict[str, str]]:
    """Extract targeted micro Q&A pairs from stabilized API lists and bullet items."""
    content = str(chunk.get("content", ""))
    meta = chunk.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    version = str(chunk.get("version") or meta.get("version") or "")
    if not version or version == "current":
        return []

    section = str(meta.get("section", ""))
    if section not in ["Stabilized APIs", "Libraries", "Language"]:
        return []

    micro_pairs: list[dict[str, str]] = []
    # Find bullet lines that reference a specific item/type/method e.g. - [`Type::method`] or - [Feature name](url)
    lines = content.split("\n")
    for line in lines:
        line_clean = line.strip()
        if not line_clean.startswith(("- [", "* [")):
            continue
        m = re.search(r"^[-*]\s*\[(?:The\s+)?`?([^`\]]+)`?", line_clean)
        if m:
            item_name = m.group(1).strip()
            if len(item_name) < 3 or len(item_name) > 80:
                continue

            instruction = f"In which Rust version was `{item_name}` stabilized or introduced, and what functionality does it provide?"
            input_text = f"API / Feature: {item_name}\nTarget Rust Version: {version}"
            output = f"In Rust {version}, `{item_name}` was stabilized in the standard library / compiler.\n\nDetails:\n{line_clean}"
            micro_pairs.append(
                {
                    "instruction": instruction,
                    "input": input_text,
                    "output": output,
                    "_is_version_sample": "true",
                }
            )
    return micro_pairs[:5]  # Limit to top 5 per section to prevent skew


def format_alpaca(chunk: dict[str, Any]) -> tuple[dict[str, str] | None, bool]:
    """Returns (sample, is_version_sample)."""
    content = str(chunk.get("content", ""))
    if len(content) < 50:
        return None, False

    source = str(chunk.get("source_type", ""))
    meta = chunk.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    library = str(chunk.get("library") or meta.get("crate_name") or meta.get("library") or "Rust")
    version = str(chunk.get("version") or meta.get("version") or "current")
    title = str(chunk.get("title") or meta.get("title") or "Documentation")

    from_ver = meta.get("from_version")
    to_ver = meta.get("to_version") or version
    section = meta.get("section")
    post_type = meta.get("post_type")

    is_version_sample = False

    # 1. Consecutive version diffs (e.g. 1.80.0 -> 1.81.0)
    if meta.get("is_migration_guide") and from_ver:
        instruction, input_text = _format_version_diff(str(from_ver), str(to_ver), title)
        is_version_sample = True
    # 2. Official rustc release records & blog announcements
    elif source == "official_releases" or library == "rustc":
        instruction, input_text = _format_rustc_release(version, title, section, post_type)
        is_version_sample = True
    # 3. Rust Edition Guides (2015, 2018, 2021, 2024)
    elif library in ["rust_edition_guide", "edition_guide"] or version in [
        "2015",
        "2018",
        "2021",
        "2024",
    ]:
        instruction, input_text = _format_edition_docs(version, title)
        is_version_sample = True
    # 4. Crate changelogs
    elif bool(meta.get("is_changelog_section", False) or "CHANGELOG" in title):
        instruction, input_text = _format_changelog(library, version, title)
        is_version_sample = True
    # 5. Crate docs
    elif source in ["docs_rs", "api_docs"]:
        instruction, input_text = _format_docs_rs(library, version, title)
    # 6. GitHub repos
    elif source in ["github", "github_repo"]:
        instruction, input_text = _format_github(library, version, title, meta)
    # 7. Official general docs
    elif source in ["rust_book", "official_docs"]:
        instruction, input_text = _format_edition_docs(version, title)
    else:
        instruction, input_text = _format_generic(library, version, title, source)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": content,
    }, is_version_sample


def main() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    chunks_file = PROCESSED_DIR / "all_chunks.jsonl"

    if not chunks_file.exists():
        print(f"Error: Processed chunks file not found at {chunks_file}")
        return

    print("Creating Alpaca dataset with version-aware instruction prompts...")
    general_samples: list[dict[str, str]] = []
    version_samples: list[dict[str, str]] = []
    total_chunks = 0

    with open(chunks_file) as f:
        for line in tqdm(f, desc="Processing chunks"):
            total_chunks += 1
            chunk: dict[str, Any] = json.loads(line)
            formatted, is_ver = format_alpaca(chunk)
            if formatted:
                if is_ver:
                    version_samples.append(formatted)
                else:
                    general_samples.append(formatted)

            # Also extract micro Q&A pairs for stabilized APIs
            micro_pairs = extract_micro_qa_pairs(chunk)
            for mp in micro_pairs:
                version_samples.append(
                    {
                        "instruction": mp["instruction"],
                        "input": mp["input"],
                        "output": mp["output"],
                    }
                )

    print(
        f"\nInitial extraction: {len(general_samples)} general samples, {len(version_samples)} version/edition samples."
    )

    # Stratified Sampling / Boosted Mix:
    # Ensure version & edition knowledge constitutes 10-15% of the total dataset
    current_ratio = len(version_samples) / max(len(general_samples) + len(version_samples), 1)
    print(f"Initial version ratio: {current_ratio * 100:.2f}%")

    if current_ratio < TARGET_VERSION_RATIO and version_samples:
        target_version_count = int(
            len(general_samples) * (TARGET_VERSION_RATIO / (1.0 - TARGET_VERSION_RATIO))
        )
        print(
            f"Applying stratified boost: scaling version samples from {len(version_samples)} to {target_version_count} (~{TARGET_VERSION_RATIO * 100:.1f}% mix)..."
        )
        # Resample with replacement to achieve the target mix
        boosted_version_samples = random.choices(version_samples, k=target_version_count)
    else:
        boosted_version_samples = version_samples

    dataset = general_samples + boosted_version_samples
    random.shuffle(dataset)

    split = int(0.98 * len(dataset))
    train_set = dataset[:split]
    val_set = dataset[split:]

    with open(DATASETS_DIR / "train.jsonl", "w") as f:
        for item in train_set:
            f.write(json.dumps(item) + "\n")
    with open(DATASETS_DIR / "val.jsonl", "w") as f:
        for item in val_set:
            f.write(json.dumps(item) + "\n")

    final_version_ratio = len(boosted_version_samples) / max(len(dataset), 1)
    print("\nFinal Dataset Summary:")
    print(f"  - Total chunks processed: {total_chunks}")
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Version & Edition representation: {final_version_ratio * 100:.1f}%")
    print(f"  - Train split (98%): {len(train_set)}")
    print(f"  - Val split (2%): {len(val_set)}")
    print(f"Saved to {DATASETS_DIR / 'train.jsonl'} and {DATASETS_DIR / 'val.jsonl'}")


if __name__ == "__main__":
    main()
