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

import argparse
import hashlib
import json
import random
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag

import yaml
from tqdm import tqdm

from lib.config import (
    DEFAULT_CHUNKS_FILE,
    DEFAULT_DATASETS_DIR,
    DEFAULT_PROCESSED_DIR,
)

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / DEFAULT_PROCESSED_DIR
DATASETS_DIR = BASE_DIR / DEFAULT_DATASETS_DIR

TARGET_VERSION_RATIO = 0.12  # 12% representation for version & edition knowledge
TARGET_CYBERSEC_RATIO = 0.08  # 8% representation for cybersecurity, RustSec & LOTL
RNG = random.Random(3407)


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
        return RNG.choice(templates), f"Edition: 2015\nTopic: {title}"
    elif version == "2018":
        templates = [
            f"Explain the language changes, module system improvements, and idioms introduced in the Rust 2018 Edition regarding '{title}'.",
            f"How did Rust 2018 evolve from Rust 2015 in terms of '{title}' and developer ergonomics?",
        ]
        return RNG.choice(templates), f"Edition: 2018\nTopic: {title}"
    elif version == "2021":
        templates = [
            f"Explain the Rust 2021 Edition updates, prelude changes, closure disjoint captures, and migration guidelines for '{title}'.",
            f"What are the key behavioral changes and upgrade instructions in Rust 2021 regarding '{title}'?",
        ]
        return RNG.choice(templates), f"Edition: 2021\nTopic: {title}"
    elif version == "2024":
        templates = [
            f"Explain the Rust 2024 Edition changes, language features, breaking modifications, and transition strategies for '{title}'.",
            f"What changes in compiler semantics, syntax, and standard prelude were introduced in Rust 2024 for '{title}'?",
        ]
        return RNG.choice(templates), f"Edition: 2024\nTopic: {title}"
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
    return RNG.choice(templates)


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
    library: str, meta: dict[str, Any], cutoff_date: str
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
    instruction = RNG.choice(templates)
    input_text = (
        f"Crate: {library}\n"
        f"Advisory ID: {adv_id}\n"
        f"CVE: {cve_str}\n"
        f"Patched: {patched_str}\n"
        f"Knowledge Cutoff Date: {cutoff_date}"
    )
    footer = f"\n\n*Security Advisory Cutoff Note: This vulnerability record is current up to {cutoff_date}. For advisories discovered after {cutoff_date}, always run `cargo audit` in your project.*"
    return instruction, input_text, footer


def _format_kali_tool(library: str, meta: dict[str, Any], cutoff_date: str) -> tuple[str, str]:
    tool = str(meta.get("tool") or library)
    cat = str(meta.get("category") or "Security Testing Tool")
    synopsis = str(meta.get("synopsis") or "")
    templates = [
        f"Explain the Kali Linux tool '{tool}' ({cat}), detailing all its primary CLI flags, options, syntax, and attack/defense usage.",
        f"What are the available command-line options, flags, and practical usage examples for '{tool}' in Kali Linux?",
        f"How is the CLI tool '{tool}' used in penetration testing, and what network or host artifacts does it generate?",
    ]
    instruction = RNG.choice(templates)
    input_text = f"Tool: {tool}\nEcosystem: Kali Linux\nCategory: {cat}"
    if synopsis:
        input_text += f"\nSynopsis: {synopsis}"
    input_text += f"\nKnowledge Cutoff Date: {cutoff_date}"
    return instruction, input_text


def _format_lotl(library: str, meta: dict[str, Any], cutoff_date: str) -> tuple[str, str]:
    bin_name = str(meta.get("binary_name") or meta.get("cmdlet") or library)
    platform = str(meta.get("platform") or "system")
    lotl_type = str(meta.get("lotl_type") or "living-off-the-land")
    templates = [
        f"Detail the living-off-the-land (LOTL) capabilities, offensive execution syntax, and defensive telemetry (Event IDs / auditd) for '{bin_name}' on {platform}.",
        f"How is '{bin_name}' used in living-off-the-land operations on {platform}, and how do defenders detect and log this activity?",
        f"Analyze the dual-use security risks of the built-in binary '{bin_name}' on {platform} from both red team and blue team perspectives.",
    ]
    instruction = RNG.choice(templates)
    input_text = f"Binary/Utility: {bin_name}\nPlatform: {platform}\nType: {lotl_type}\nKnowledge Cutoff Date: {cutoff_date}"
    return instruction, input_text


def _format_rust_hardening(title: str, cutoff_date: str) -> tuple[str, str]:
    templates = [
        f"Explain the Rust security and hardening mechanisms regarding '{title}'.",
        f"What are the best practices for '{title}' to prevent vulnerabilities and undefined behavior in Rust?",
        f"How do compiler flags and verification tools ensure soundness and security for '{title}' in Rust?",
    ]
    instruction = RNG.choice(templates)
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


def _format_cybersec_alpaca(
    content: str,
    source: str,
    meta: dict[str, Any],
    library: str,
    title: str,
    chunk_cutoff: str,
) -> tuple[dict[str, str], bool, bool] | None:
    if not (
        source == "cybersec"
        or meta.get("advisory_id")
        or meta.get("lotl_type")
        or meta.get("tool")
        or meta.get("security_topic")
    ):
        return None

    if meta.get("advisory_id") or source == "rustsec":
        instruction, input_text, footer = _format_rustsec_advisory(library, meta, chunk_cutoff)
        output_text = content + footer if footer not in content else content
        return (
            {"instruction": instruction, "input": input_text, "output": output_text},
            False,
            True,
        )
    if meta.get("tool") or source == "kali_tools":
        instruction, input_text = _format_kali_tool(library, meta, chunk_cutoff)
        return (
            {"instruction": instruction, "input": input_text, "output": content},
            False,
            True,
        )
    if meta.get("lotl_type") or source in [
        "lolbas",
        "gtfobins",
        "loobins",
        "powershell_lotl",
    ]:
        instruction, input_text = _format_lotl(library, meta, chunk_cutoff)
        return (
            {"instruction": instruction, "input": input_text, "output": content},
            False,
            True,
        )
    if meta.get("security_topic") == "hardening" or source == "rust_hardening":
        instruction, input_text = _format_rust_hardening(title, chunk_cutoff)
        return (
            {"instruction": instruction, "input": input_text, "output": content},
            False,
            True,
        )
    return None


def _format_general_or_version_prompt(
    source: str,
    library: str,
    version: str,
    title: str,
    meta: dict[str, Any],
) -> tuple[str, str, bool]:
    """Returns (instruction, input_text, is_version_sample)."""
    from_ver = meta.get("from_version")
    to_ver = meta.get("to_version") or version
    section = meta.get("section")
    post_type = meta.get("post_type")

    # 1. Consecutive version diffs (e.g. 1.80.0 -> 1.81.0)
    if meta.get("is_migration_guide") and from_ver:
        inst, inp = _format_version_diff(str(from_ver), str(to_ver), title)
        return inst, inp, True
    # 2. Official rustc release records & blog announcements
    if source == "official_releases" or library == "rustc":
        inst, inp = _format_rustc_release(version, title, section, post_type)
        return inst, inp, True
    # 3. Rust Edition Guides (2015, 2018, 2021, 2024)
    if library in ["rust_edition_guide", "edition_guide"] or version in [
        "2015",
        "2018",
        "2021",
        "2024",
    ]:
        inst, inp = _format_edition_docs(version, title)
        return inst, inp, True
    # 4. Crate changelogs
    if bool(meta.get("is_changelog_section", False) or "CHANGELOG" in title):
        inst, inp = _format_changelog(library, version, title)
        return inst, inp, True
    # 5. Crate docs
    if source in ["docs_rs", "api_docs"]:
        inst, inp = _format_docs_rs(library, version, title)
        return inst, inp, False
    # 6. GitHub repos
    if source in ["github", "github_repo"]:
        inst, inp = _format_github(library, version, title, meta)
        return inst, inp, False
    # 7. Official general docs
    if source in ["rust_book", "official_docs"]:
        inst, inp = _format_edition_docs(version, title)
        return inst, inp, False

    inst, inp = _format_generic(library, version, title, source)
    return inst, inp, False


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
    chunk_cutoff = str(chunk.get("cutoff_date") or meta.get("cutoff_date") or cutoff_date)

    # 0. Cybersecurity, RustSec, Kali Linux, LOTL, and Hardening
    cybersec_result = _format_cybersec_alpaca(content, source, meta, library, title, chunk_cutoff)
    if cybersec_result is not None:
        return cybersec_result

    instruction, input_text, is_version_sample = _format_general_or_version_prompt(
        source, library, version, title, meta
    )
    return (
        {
            "instruction": instruction,
            "input": input_text,
            "output": content,
        },
        is_version_sample,
        False,
    )


def _extract_all_samples(
    chunks_file: Path, cutoff_date: str
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], int]:
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
                formatted["_group"] = source_group(chunk)
                formatted["_category"] = (
                    "cybersec" if is_cyber else "version" if is_ver else "general"
                )
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
                        "_group": source_group(chunk),
                        "_category": "version",
                    }
                )

    # Calibration replicas belong only to training, never to validation.
    cutoff_pairs = generate_cutoff_qa_pairs(cutoff_date)
    for cp in cutoff_pairs:
        cybersec_samples.append(
            {
                **cp,
                "_group": "cutoff-calibration",
                "_category": "cybersec",
                "_train_only": "true",
            }
        )

    return general_samples, version_samples, cybersec_samples, total_chunks


def _stratify_and_mix(
    general_samples: list[dict[str, str]],
    version_samples: list[dict[str, str]],
    cybersec_samples: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    # 1. Version & edition knowledge (10-15%)
    current_ver_ratio = len(version_samples) / max(len(general_samples) + len(version_samples), 1)
    if current_ver_ratio < TARGET_VERSION_RATIO and version_samples:
        target_version_count = int(
            len(general_samples) * (TARGET_VERSION_RATIO / (1.0 - TARGET_VERSION_RATIO))
        )
        print(
            f"Applying stratified boost: scaling version samples from {len(version_samples)} to {target_version_count} (~{TARGET_VERSION_RATIO * 100:.1f}% mix)..."
        )
        boosted_version_samples = version_samples + RNG.choices(
            version_samples, k=target_version_count - len(version_samples)
        )
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
        boosted_cybersec_samples = cybersec_samples + RNG.choices(
            cybersec_samples, k=target_cyber_count - len(cybersec_samples)
        )
    else:
        boosted_cybersec_samples = cybersec_samples

    return boosted_version_samples, boosted_cybersec_samples


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_group(chunk: dict[str, Any]) -> str:
    meta = chunk.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    source = str(chunk.get("source_type") or "unknown")
    library = str(chunk.get("library") or meta.get("library") or meta.get("crate_name") or "")
    version = str(chunk.get("version") or meta.get("version") or "current")
    url = str(chunk.get("url") or meta.get("url") or "")
    # Fragments identify sections, not independent source documents.
    document = (
        urldefrag(url)[0] if url else str(meta.get("document_id") or meta.get("source_file") or "")
    )
    if not document:
        # Without document identity, keep the entire source/library/version
        # together rather than pretend independent chunks are safe holdouts.
        document = "unknown-document"
    return digest(json.dumps([source, library, version, document]))


def output_key(sample: dict[str, str]) -> str:
    return digest(re.sub(r"\s+", "", sample["output"]).casefold())


def record_key(sample: dict[str, str]) -> str:
    return digest(json.dumps([sample[k] for k in ("instruction", "input", "output")]))


def connected_groups(samples: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    parents: dict[str, str] = {}

    def find(group: str) -> str:
        parents.setdefault(group, group)
        root = group
        while parents[root] != root:
            root = parents[root]
        while parents[group] != group:
            next_group = parents[group]
            parents[group] = root
            group = next_group
        return root

    outputs: dict[str, str] = {}
    for sample in samples:
        sample.setdefault("_source_group", sample["_group"])
        group = find(sample["_group"])
        key = output_key(sample)
        if key in outputs:
            parents[find(outputs[key])] = find(group)
        else:
            outputs[key] = group
    grouped: dict[str, list[dict[str, str]]] = {}
    for sample in samples:
        sample["_group"] = find(sample["_group"])
        grouped.setdefault(sample["_group"], []).append(sample)
    return grouped


def split_groups(
    samples: list[dict[str, str]],
    validation_ratio: float = 0.02,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Keep source derivatives and equivalent outputs in connected groups."""
    if not samples or not 0 < validation_ratio < 1:
        raise ValueError("Nonempty samples and a validation ratio between 0 and 1 required")
    grouped = connected_groups(samples)
    groups = sorted(grouped)
    RNG.shuffle(groups)
    totals = Counter(s["_category"] for s in samples if not s.get("_train_only"))
    targets = {category: count * validation_ratio for category, count in totals.items()}
    selected: Counter[str] = Counter()
    train, val = [], []
    for group in groups:
        rows = grouped[group]
        counts = Counter(s["_category"] for s in rows)
        before = sum(abs(selected[c] - target) for c, target in targets.items())
        after = sum(abs(selected[c] + counts[c] - target) for c, target in targets.items())
        if not any(s.get("_train_only") for s in rows) and after < before:
            val.extend(rows)
            selected.update(counts)
        else:
            train.extend(rows)
    # Validation should not contain repeated answers, even within one split.
    unique_val = {output_key(s): s for s in val}
    val = list(unique_val.values())
    if not train or not val:
        raise ValueError("Source grouping cannot produce nonempty train and validation splits")
    audit_splits(train, val)
    return train, val


def audit_splits(
    train: list[dict[str, str]],
    val: list[dict[str, str]],
) -> dict[str, int]:
    checks = {
        "exact_record_overlap": len({record_key(s) for s in train} & {record_key(s) for s in val}),
        "normalized_output_overlap": len(
            {output_key(s) for s in train} & {output_key(s) for s in val}
        ),
        "provenance_group_overlap": len({s["_group"] for s in train} & {s["_group"] for s in val}),
        "validation_calibration_rows": sum(bool(s.get("_train_only")) for s in val),
    }
    if any(checks.values()):
        raise ValueError(f"Unsafe dataset split: {checks}")
    return checks


def verify_dataset(directory: Path) -> dict[str, int]:
    """Verify published/staged bytes and provenance, not only in-memory records."""
    manifest = json.loads((directory / "split_manifest.json").read_text(encoding="utf-8"))
    identities: dict[str, dict[str, set[str]]] = {}
    for name in ("train", "val"):
        path = directory / f"{name}.jsonl"
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != manifest["splits"][name]["sha256"]:
                raise ValueError(f"{name}: dataset checksum mismatch")
        keys: dict[str, set[str]] = {
            "record": set(),
            "output": set(),
            "group": set(),
            "source_group": set(),
        }
        counts: Counter[str] = Counter()
        rows = 0
        with (
            path.open(encoding="utf-8") as data,
            (directory / f"{name}.provenance.jsonl").open(encoding="utf-8") as provenance,
        ):
            for line, identity in zip(data, provenance, strict=True):
                row, meta = json.loads(line), json.loads(identity)
                if set(row) != {"instruction", "input", "output"}:
                    raise ValueError(f"{name}: incompatible Alpaca schema")
                record, output = record_key(row), output_key(row)
                if record != meta["record_sha256"] or output != meta["output_sha256"]:
                    raise ValueError(f"{name}: provenance does not match dataset row")
                if name == "val" and meta["training_only"]:
                    raise ValueError("Training-only calibration leaked into validation")
                keys["record"].add(record)
                keys["output"].add(output)
                keys["group"].add(meta["group"])
                keys["source_group"].add(meta["source_group"])
                counts[meta["category"]] += 1
                rows += 1
        expected = manifest["splits"][name]
        if (
            rows != expected["rows"]
            or dict(counts) != expected["categories"]
            or len(keys["group"]) != expected["groups"]
        ):
            raise ValueError(f"{name}: manifest counts mismatch")
        if name == "val" and len(keys["output"]) != rows:
            raise ValueError("Repeated validation outputs")
        identities[name] = keys
    checks = {
        key + "_overlap": len(identities["train"][key] & identities["val"][key])
        for key in identities["train"]
    }
    if any(checks.values()):
        raise ValueError(f"Published datasets overlap: {checks}")
    return checks


def generate_dataset(
    chunks_file: Path,
    output_dir: Path,
    cutoff_date: str,
    seed: int = 3407,
) -> dict[str, Any]:
    RNG.seed(seed)
    general, version, cybersec, total_chunks = _extract_all_samples(chunks_file, cutoff_date)
    train, val = split_groups(general + version + cybersec)
    general_train = [s for s in train if s["_category"] == "general"]
    version_train = [s for s in train if s["_category"] == "version"]
    cyber_train = [s for s in train if s["_category"] == "cybersec"]
    calibration = [s for s in cyber_train if s.get("_train_only")]
    cyber_train += [dict(s) for s in calibration for _ in range(24)]
    version_train, cyber_train = _stratify_and_mix(general_train, version_train, cyber_train)
    train = general_train + version_train + cyber_train
    RNG.shuffle(train)
    RNG.shuffle(val)
    checks = audit_splits(train, val)
    with chunks_file.open("rb") as source:
        input_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "seed": seed,
        "cutoff_date": cutoff_date,
        "input_sha256": input_sha256,
        "input_chunks": total_chunks,
        "validation_target_before_boost": 0.02,
        "lineage": "Regenerated local corpus; not evidence of checkpoint-7500 training lineage.",
        "checks": checks,
        "splits": {},
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rust-lora-datasets-", dir=output_dir.parent) as temp:
        stage = Path(temp)
        for name, rows in (("train", train), ("val", val)):
            path = stage / f"{name}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(
                        json.dumps({k: row[k] for k in ("instruction", "input", "output")}) + "\n"
                    )
            with (stage / f"{name}.provenance.jsonl").open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(
                        json.dumps(
                            {
                                "group": row["_group"],
                                "source_group": row["_source_group"],
                                "category": row["_category"],
                                "training_only": bool(row.get("_train_only")),
                                "record_sha256": record_key(row),
                                "output_sha256": output_key(row),
                            }
                        )
                        + "\n"
                    )
            manifest["splits"][name] = {
                "rows": len(rows),
                "groups": len({s["_group"] for s in rows}),
                "categories": dict(sorted(Counter(s["_category"] for s in rows).items())),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        manifest["actual_validation_fraction"] = len(val) / (len(train) + len(val))
        (stage / "split_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        verify_dataset(stage)
        output_dir.mkdir(parents=True, exist_ok=True)
        names = (
            "train.jsonl",
            "val.jsonl",
            "train.provenance.jsonl",
            "val.provenance.jsonl",
            "split_manifest.json",
        )
        existing = [output_dir / name for name in names if (output_dir / name).exists()]
        if existing:
            snapshot_id = digest(
                "".join(hashlib.sha256(p.read_bytes()).hexdigest() for p in existing)
            )[:16]
            backup = output_dir / "backups" / snapshot_id
            backup.mkdir(parents=True, exist_ok=True)
            for path in existing:
                destination = backup / path.name
                if destination.exists() and destination.read_bytes() != path.read_bytes():
                    raise FileExistsError(f"Snapshot collision: {destination}")
                shutil.copy2(path, destination)
            print(f"Preserved previous dataset artifacts: {backup}")
        for name in names:
            (stage / name).replace(output_dir / name)
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--chunks-file", type=Path, default=BASE_DIR / DEFAULT_CHUNKS_FILE)
    parser.add_argument("--output-dir", type=Path, default=DATASETS_DIR)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_dataset(args.output_dir), indent=2))
    else:
        generate_dataset(args.chunks_file, args.output_dir, get_cutoff_date(), args.seed)


if __name__ == "__main__":
    main()
