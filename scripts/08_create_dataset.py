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

import yaml
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATASETS_DIR = BASE_DIR / "data" / "datasets"

TARGET_VERSION_RATIO = 0.12  # 12% representation for version & edition knowledge
TARGET_CYBERSEC_RATIO = 0.08  # 8% representation for cybersecurity, RustSec & LOTL


def get_cutoff_date() -> str:
    """Read knowledge cutoff date from config/cybersec.yaml."""
    cfg_file = BASE_DIR / "config" / "cybersec.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Missing required configuration file: {cfg_file}")
    with open(cfg_file) as f:
        c = yaml.safe_load(f) or {}
    cutoff = c.get("metadata", {}).get("cutoff_date")
    if not cutoff:
        raise ValueError(f"cutoff_date not found in {cfg_file} under metadata.cutoff_date")
    return str(cutoff)


def generate_cutoff_qa_pairs(cutoff_date: str) -> list[dict[str, str]]:
    """Synthesize explicit knowledge cutoff Q&A pairs for model calibration."""
    return [
        {
            "instruction": "What is your knowledge cutoff date for cybersecurity vulnerabilities, RustSec advisories, and security tooling?",
            "input": "Domain: Cybersecurity and RustSec Vulnerability Database",
            "output": (
                f"My knowledge cutoff date for security vulnerabilities, RustSec advisories, CVEs, and security tooling is {cutoff_date}. "
                f"Because vulnerability databases and exploit techniques change dynamically, you should always run `cargo audit` or `cargo deny` "
                f"in your Rust projects, and check https://rustsec.org/ for any advisories published after {cutoff_date}."
            ),
        },
        {
            "instruction": "When was your RustSec vulnerability and security tooling database snapshot taken?",
            "input": "Topic: Vulnerability Database Snapshot & Temporal Boundaries",
            "output": (
                f"The RustSec advisory database, Kali Linux CLI tool specs, and living-off-the-land (LOTL) intelligence in this model "
                f"are snapshot as of {cutoff_date}. Any CVEs or crate advisories discovered after {cutoff_date} are not in this offline "
                f"model; verify newer dependencies using `cargo audit`."
            ),
        },
        {
            "instruction": "Can you guarantee that a Rust crate has no vulnerabilities if it is not in your advisory database?",
            "input": "Topic: Negative Security Guarantees & Cutoff Limitations",
            "output": (
                f"No, absence of an advisory in my knowledge base does not guarantee that a crate is vulnerability-free. "
                f"My security advisory data has an explicit knowledge cutoff date of {cutoff_date}. If a vulnerability was disclosed "
                f"after {cutoff_date}, it will not be reflected here. Always integrate continuous security scanners such as "
                f"`cargo-audit`, `cargo-deny`, or GitHub Dependabot in your CI/CD pipelines."
            ),
        },
        {
            "instruction": "What should I do to check for Rust vulnerabilities that occurred after your cutoff date?",
            "input": f"Question: Auditing after {cutoff_date}",
            "output": (
                f"To audit your dependencies for vulnerabilities published after my knowledge cutoff date ({cutoff_date}):\n"
                f"1. Run `cargo install cargo-audit && cargo audit` to query the live RustSec Advisory Database.\n"
                f"2. Use `cargo-deny` with `cargo deny check advisories` in CI/CD.\n"
                f"3. Check https://rustsec.org/advisories/ directly for the latest CVE disclosures.\n"
                f"4. Review crate changelogs and GitHub security advisories (GHSA)."
            ),
        },
    ]


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


def _format_rustsec_advisory(
    library: str, version: str, title: str, meta: dict[str, Any], cutoff_date: str
) -> tuple[str, str, str]:
    adv_id = str(meta.get("advisory_id") or "RUSTSEC-ADVISORY")
    cves = meta.get("cve_aliases") or []
    cve_str = ", ".join(cves) if cves else "None"
    patched = meta.get("patched_versions") or []
    patched_str = ", ".join(patched) if patched else "None"

    templates = [
        f"Identify known security vulnerabilities, CVEs, root causes, and patched versions for the Rust crate '{library}' (advisory {adv_id}).",
        f"Analyze the security vulnerability {adv_id} in crate '{library}'. What is the root cause and how is it remediated in Rust?",
        f"Explain the unsoundness or security flaw reported in {adv_id} ({cve_str}) affecting crate '{library}'.",
        f"What version range of crate '{library}' is affected by {adv_id}, and how can developers mitigate it?",
    ]
    instruction = random.choice(templates)
    input_text = (
        f"Crate: {library}\n"
        f"Advisory ID: {adv_id}\n"
        f"CVE: {cve_str}\n"
        f"Patched: {patched_str}\n"
        f"Knowledge Cutoff Date: {cutoff_date}"
    )
    footer = f"\n\n*Security Advisory Cutoff Note: This vulnerability record is current up to {cutoff_date}. For advisories discovered after {cutoff_date}, always run `cargo audit` in your project.*"
    return instruction, input_text, footer


def _format_kali_tool(
    library: str, version: str, title: str, meta: dict[str, Any], cutoff_date: str
) -> tuple[str, str]:
    tool = str(meta.get("tool") or library)
    cat = str(meta.get("category") or "Security Testing Tool")
    synopsis = str(meta.get("synopsis") or "")
    templates = [
        f"Explain the Kali Linux tool '{tool}' ({cat}), detailing all its primary CLI flags, options, syntax, and attack/defense usage.",
        f"What are the available command-line options, flags, and practical usage examples for '{tool}' in Kali Linux?",
        f"How is the CLI tool '{tool}' used in penetration testing, and what network or host artifacts does it generate?",
    ]
    instruction = random.choice(templates)
    input_text = f"Tool: {tool}\nEcosystem: Kali Linux\nCategory: {cat}"
    if synopsis:
        input_text += f"\nSynopsis: {synopsis}"
    input_text += f"\nKnowledge Cutoff Date: {cutoff_date}"
    return instruction, input_text


def _format_lotl(
    library: str, version: str, title: str, meta: dict[str, Any], cutoff_date: str
) -> tuple[str, str]:
    bin_name = str(meta.get("binary_name") or meta.get("cmdlet") or library)
    platform = str(meta.get("platform") or "system")
    lotl_type = str(meta.get("lotl_type") or "living-off-the-land")
    templates = [
        f"Detail the living-off-the-land (LOTL) capabilities, offensive execution syntax, and defensive telemetry (Event IDs / auditd) for '{bin_name}' on {platform}.",
        f"How is '{bin_name}' used in living-off-the-land operations on {platform}, and how do defenders detect and log this activity?",
        f"Analyze the dual-use security risks of the built-in binary '{bin_name}' on {platform} from both red team and blue team perspectives.",
    ]
    instruction = random.choice(templates)
    input_text = f"Binary/Utility: {bin_name}\nPlatform: {platform}\nType: {lotl_type}\nKnowledge Cutoff Date: {cutoff_date}"
    return instruction, input_text


def _format_rust_hardening(
    library: str, version: str, title: str, meta: dict[str, Any], cutoff_date: str
) -> tuple[str, str]:
    templates = [
        f"Explain the Rust security and hardening mechanisms regarding '{title}'.",
        f"What are the best practices for '{title}' to prevent vulnerabilities and undefined behavior in Rust?",
        f"How do compiler flags and verification tools ensure soundness and security for '{title}' in Rust?",
    ]
    instruction = random.choice(templates)
    input_text = f"Security Topic: {title}\nDomain: Rust Hardening & Soundness\nKnowledge Cutoff Date: {cutoff_date}"
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


def format_alpaca(
    chunk: dict[str, Any], cutoff_date: str
) -> tuple[dict[str, str] | None, bool, bool]:
    """Returns (sample, is_version_sample, is_cybersec_sample)."""
    content = str(chunk.get("content", ""))
    if len(content) < 50:
        return None, False, False

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
    chunk_cutoff = str(chunk.get("cutoff_date") or meta.get("cutoff_date") or cutoff_date)

    # 0. Cybersecurity, RustSec, Kali Linux, LOTL, and Hardening
    if (
        source == "cybersec"
        or meta.get("advisory_id")
        or meta.get("lotl_type")
        or meta.get("tool")
        or meta.get("security_topic")
    ):
        if meta.get("advisory_id") or source == "rustsec":
            instruction, input_text, footer = _format_rustsec_advisory(
                library, version, title, meta, chunk_cutoff
            )
            output_text = content + footer if footer not in content else content
            return (
                {"instruction": instruction, "input": input_text, "output": output_text},
                False,
                True,
            )
        elif meta.get("tool") or source == "kali_tools":
            instruction, input_text = _format_kali_tool(
                library, version, title, meta, chunk_cutoff
            )
            return (
                {"instruction": instruction, "input": input_text, "output": content},
                False,
                True,
            )
        elif meta.get("lotl_type") or source in ["lolbas", "gtfobins", "loobins", "powershell_lotl"]:
            instruction, input_text = _format_lotl(
                library, version, title, meta, chunk_cutoff
            )
            return (
                {"instruction": instruction, "input": input_text, "output": content},
                False,
                True,
            )
        elif meta.get("security_topic") == "hardening" or source == "rust_hardening":
            instruction, input_text = _format_rust_hardening(
                library, version, title, meta, chunk_cutoff
            )
            return (
                {"instruction": instruction, "input": input_text, "output": content},
                False,
                True,
            )

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

    return (
        {
            "instruction": instruction,
            "input": input_text,
            "output": content,
        },
        is_version_sample,
        False,
    )


def main() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    chunks_file = PROCESSED_DIR / "all_chunks.jsonl"

    if not chunks_file.exists():
        print(f"Error: Processed chunks file not found at {chunks_file}")
        return

    cutoff_date = get_cutoff_date()
    print(f"Creating Alpaca dataset with version-aware & cybersec-aware instruction prompts (Cutoff: {cutoff_date})...")
    general_samples: list[dict[str, str]] = []
    version_samples: list[dict[str, str]] = []
    cybersec_samples: list[dict[str, str]] = []
    total_chunks = 0

    with open(chunks_file) as f:
        for line in tqdm(f, desc="Processing chunks"):
            total_chunks += 1
            chunk: dict[str, Any] = json.loads(line)
            formatted, is_ver, is_cyber = format_alpaca(chunk, cutoff_date)
            if formatted:
                if is_cyber:
                    cybersec_samples.append(formatted)
                elif is_ver:
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

    # Inject targeted cutoff calibration Q&A pairs (replicated across batches)
    cutoff_pairs = generate_cutoff_qa_pairs(cutoff_date)
    for cp in cutoff_pairs:
        # Replicate 25 times each for strong weight grounding
        for _ in range(25):
            cybersec_samples.append(dict(cp))

    print(
        f"\nInitial extraction: {len(general_samples)} general samples, "
        f"{len(version_samples)} version/edition samples, "
        f"{len(cybersec_samples)} cybersecurity samples."
    )

    # Stratified Sampling / Boosted Mix:
    # 1. Version & edition knowledge (10-15%)
    current_ver_ratio = len(version_samples) / max(len(general_samples) + len(version_samples), 1)
    if current_ver_ratio < TARGET_VERSION_RATIO and version_samples:
        target_version_count = int(
            len(general_samples) * (TARGET_VERSION_RATIO / (1.0 - TARGET_VERSION_RATIO))
        )
        print(
            f"Applying stratified boost: scaling version samples from {len(version_samples)} to {target_version_count} (~{TARGET_VERSION_RATIO * 100:.1f}% mix)..."
        )
        boosted_version_samples = random.choices(version_samples, k=target_version_count)
    else:
        boosted_version_samples = version_samples

    # 2. Cybersecurity & LOTL knowledge (~8%)
    base_count = len(general_samples) + len(boosted_version_samples)
    current_cyber_ratio = len(cybersec_samples) / max(base_count + len(cybersec_samples), 1)
    if current_cyber_ratio < TARGET_CYBERSEC_RATIO and cybersec_samples:
        target_cyber_count = int(
            base_count * (TARGET_CYBERSEC_RATIO / (1.0 - TARGET_CYBERSEC_RATIO))
        )
        print(
            f"Applying stratified boost: scaling cybersec samples from {len(cybersec_samples)} to {target_cyber_count} (~{TARGET_CYBERSEC_RATIO * 100:.1f}% mix)..."
        )
        boosted_cybersec_samples = random.choices(cybersec_samples, k=target_cyber_count)
    else:
        boosted_cybersec_samples = cybersec_samples

    dataset = general_samples + boosted_version_samples + boosted_cybersec_samples
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
    final_cyber_ratio = len(boosted_cybersec_samples) / max(len(dataset), 1)
    print("\nFinal Dataset Summary:")
    print(f"  - Total chunks processed: {total_chunks}")
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Version & Edition representation: {final_version_ratio * 100:.1f}%")
    print(f"  - Cybersecurity & LOTL representation: {final_cyber_ratio * 100:.1f}%")
    print(f"  - Knowledge Cutoff Date: {cutoff_date}")
    print(f"  - Train split (98%): {len(train_set)}")
    print(f"  - Val split (2%): {len(val_set)}")
    print(f"Saved to {DATASETS_DIR / 'train.jsonl'} and {DATASETS_DIR / 'val.jsonl'}")


if __name__ == "__main__":
    main()
