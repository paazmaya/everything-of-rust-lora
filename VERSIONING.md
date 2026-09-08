# Versioning Architecture & RAG Pipeline

This document describes how library and documentation versions are collected, preserved, indexed into ChromaDB (for version-aware RAG), and formatted into instruction tuning datasets for the LoRA adapter.

---

## 1. Motivation

When training models on programming languages and fast-evolving ecosystems like Rust:

- **API breaking changes** occur between major/minor versions (e.g. `axum 0.6` vs `axum 0.7`, `clap 3` vs `clap 4`).
- **Migration guidance** is one of the highest-value capabilities of an assistant (e.g. "how do I migrate my handler from axum 0.6 to 0.7?").
- Overwriting older data with newer data leaves the model unable to differentiate version-specific syntax or assist users maintaining legacy codebases.
- Blind deduplication across versions removes documentation sections that are identical between versions, creating gaps in historical docs.

---

## 2. Directory Layout & Persistence

Data across iterations is partitioned by library and version under `data/raw/`:

```
data/
├── raw/
│   ├── docs_rs/
│   │   ├── axum/
│   │   │   ├── 0.6.20/          <-- Preserved historical version
│   │   │   │   └── *.json
│   │   │   └── 0.7.5/           <-- Latest version
│   │   │       └── *.json
│   ├── github/
│   │   ├── tokio-rs_axum/
│   │   │   ├── 0.6.0/           <-- Parsed CHANGELOG section for v0.6.0
│   │   │   ├── 0.7.0/           <-- Parsed CHANGELOG section for v0.7.0
│   │   │   └── 0.7.5/           <-- Tagged README, docs, guides
│   ├── rust_book/
│   │   ├── 2021/                <-- Rust 2021 Edition
│   │   └── 2024/                <-- Rust 2024 Edition
│   └── blogs/
│       └── official/
│           ├── 2023-11/
│           └── 2024-02/
├── processed/
│   └── all_chunks.jsonl         <-- Version-aware composite chunks
└── vectors/                     <-- ChromaDB persistent vector database
```

When new library versions or doc updates are fetched in future runs:

1. New version folders are created alongside existing ones.
2. Older version files are never overwritten or deleted.

---

## 3. Metadata Schema

Every raw document and processed chunk carries explicit version identity:

| Field          | Type   | Description                            | Example                                          |
| -------------- | ------ | -------------------------------------- | ------------------------------------------------ |
| `library`      | `str`  | Canonical crate/source name            | `"axum"`, `"tokio"`, `"rust_book"`               |
| `version`      | `str`  | SemVer, edition, tag, or date          | `"0.7.5"`, `"2021"`, `"2024-02"`                 |
| `version_type` | `str`  | Type of version string                 | `"semver"`, `"edition"`, `"git_tag"`, `"date"`   |
| `is_latest`    | `bool` | Whether this is the latest release     | `true` / `false`                                 |
| `source_type`  | `str`  | Origin category                        | `"api_docs"`, `"github_repo"`, `"official_docs"` |
| `url`          | `str`  | Source URL                             | `"https://docs.rs/axum/0.7.5/axum/"`             |
| `title`        | `str`  | Page or file title                     | `"axum::Router"`                                 |
| `is_changelog` | `bool` | True if section is a version changelog | `true` / `false`                                 |

---

## 4. Collection Strategies

### A. Crate API Docs (`02_collect_docs_rs.py`)

- Resolves exact SemVer from crates.io API (`/api/v1/crates/{name}`) or docs.rs HTTP redirect (`https://docs.rs/{name}/latest/` -> `https://docs.rs/{name}/{version}/`).
- Saves documentation under `data/raw/docs_rs/{name}/{version}/`.
- Supports multi-version pinning in `config/libraries.yaml`:
  ```yaml
  libraries:
    - name: axum
      repo: tokio-rs/axum
      url: https://docs.rs/axum
      versions: ["0.6.20", "0.7.5"] # Optionally tracks multiple versions
  ```

### B. GitHub Repository Docs & Changelogs (`03_collect_github.py`)

- Discovers release tags via GitHub Releases/Tags API.
- **Automated Changelog Partitioning**: `split_changelog()` parses headings such as `## [0.7.0]` in `CHANGELOG.md` or `RELEASES.md`, saving each version section into its respective version directory (`github/{repo}/{section_version}/`).
- This allows direct retrieval of breaking changes and release notes for any version.

### C. Official Rust Docs & ESP-RS (`01_collect_rust_book.py`, `04_collect_esp_rs.py`)

- Detects Rust editions (`2018`, `2021`, `2024`) and toolchain versions from page URLs and metadata.
- Partitions outputs into edition directories under `data/raw/`.

### D. Blogs & News (`05_collect_blogs.py`)

- Extracts publication timestamps and release announcement version strings (e.g., `Announcing Rust 1.75.0`).

---

## 5. Version-Aware Transformation & Deduplication (`06_transform_data.py`)

Standard deduplication creates a content hash `sha256(content)`. If two different versions of a library share unchanged methods or documentation, content-only deduplication deletes the chunk from one of the versions, creating holes in the older or newer version's context.

To solve this, the pipeline uses a **composite key**:
$$\text{hash} = \text{sha256}(\text{source\_type} : \text{library} : \text{version} : \text{content})[:20]$$

- Exact duplicates within the _same_ version are removed.
- Equivalent content across _different_ versions is preserved so each version maintains complete documentation integrity.

---

## 6. ChromaDB Vector Store & Version-Aware RAG (`07_vector_store.py`)

### Stored Metadata

Each vector entry is stored with searchable scalar metadata:

```python
{
    "library": "axum",
    "version": "0.7.5",
    "version_type": "semver",
    "is_latest": True,
    "source": "api_docs",
    "url": "https://docs.rs/axum/0.7.5/axum/struct.Router.html",
    "title": "axum::Router",
    "tokens": 420,
    "is_changelog": False
}
```

**Compound ID**: `{library}_{version}_{hash}` prevents overwrites across versions.

### RAG Query CLI & Python API

#### 1. Filtered Query by Library and Version

```bash
# Query only axum 0.7
uv run python scripts/07_vector_store.py --library axum --version 0.7.5 --query "Router route nesting"

# Query only latest documentation
uv run python scripts/07_vector_store.py --library tokio --query "select! macro"
```

#### 2. Version Migration RAG Query

```bash
# Compare versions and fetch upgrade notes
uv run python scripts/07_vector_store.py --library axum --migrate-from 0.6.20 --migrate-to 0.7.5 --query "extractors and state"
```

#### 3. List Tracked Versions

```bash
uv run python scripts/07_vector_store.py --list-versions --library axum
```

#### 4. Python RAG Usage Example

```python
from scripts.07_vector_store import VectorStore

store = VectorStore()

# 1. Single-version retrieval
results = store.query(
    "How to handle errors in route handlers?",
    library="axum",
    version="0.7.5",
    n_results=4
)

# 2. Migration comparison retrieval
diff = store.query_version_migration(
    library="axum",
    from_version="0.6.20",
    to_version="0.7.5",
    topic="Router::nest"
)
```

---

## 7. Dataset Generation for LoRA Training (`08_create_dataset.py`)

The dataset generator creates version-grounded instructions in Alpaca format:

### Standard Documentation Example:

```json
{
  "instruction": "Explain the Rust crate 'axum' (version 0.7.5) and its usage based on the documentation.",
  "input": "Crate: axum\nVersion: 0.7.5\nTopic: axum::Router",
  "output": "..."
}
```

### Migration / Changelog Example:

```json
{
  "instruction": "Explain the changes and migration details for the Rust crate 'axum' in version 0.7.0.",
  "input": "Crate: axum\nVersion: 0.7.0\nTopic: tokio-rs/axum: CHANGELOG v0.7.0",
  "output": "..."
}
```

### Edition-Specific Example:

```json
{
  "instruction": "Explain this Rust concept or language feature according to the Rust 2024 Edition.",
  "input": "Edition: 2024\nTopic: RPITIT (Return Position Impl Trait in Trait)",
  "output": "..."
}
```

This guarantees the fine-tuned model understands which APIs belong to which version, and can accurately answer migration and version-comparison questions.
