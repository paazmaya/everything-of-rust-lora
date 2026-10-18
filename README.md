# Training a LoRa that knows everything about the Rust language

This project contains the complete pipeline to train a specialized LoRA adapter for Rust programming, covering the language standard, top 100 libraries, best practices, and ESP32/Embedded IoT development.

[![Maintainability issues](https://sonarcloud.io/api/project_badges/measure?project=paazmaya_everything-of-rust-lora&metric=software_quality_maintainability_issues)](https://sonarcloud.io/summary/new_code?id=paazmaya_everything-of-rust-lora)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=paazmaya_everything-of-rust-lora&metric=coverage)](https://sonarcloud.io/summary/new_code?id=paazmaya_everything-of-rust-lora)

## Architecture Overview

1. **Data Collection** (Scripts 01-05): Scrapes Rust Book, docs.rs, GitHub, ESP-RS, and Blogs.
2. **Transformation** (Script 06): Cleans, chunks, and deduplicates the data.
3. **Vector/Graph Storage** (Script 07): Stores data in ChromaDB for easy 6-month incremental updates.
4. **Dataset Creation** (Script 08): Converts chunks into Alpaca instruction-response format.
5. **Training** ([train_k2_horizon.py](train_k2_horizon.py), [train_granite.py](train_granite.py), [train_qwen.py](train_qwen.py), [train_qwen3_8.py](train_qwen3_8.py), [train_qwen3_8_4b.py](train_qwen3_8_4b.py), [train_nemotron3_nano.py](train_nemotron3_nano.py), [train_alodlm.py](train_alodlm.py)): Trains LoRA using Unsloth or Transformers/PEFT.
6. **Agent Serving** (`serve_coding_agent.py`): Serves base model + LoRA over an OpenAI-compatible API for coding agents.
7. **Export** (Script 09): Packages the trained LoRA adapter with metadata and a `Modelfile`.
8. **Merge + GGUF** (Script 10): Merges the adapter with the base model and exports a runnable GGUF bundle for Ollama.

## Phase 1: Environment Setup

### Local Setup (Linux)

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate

# Install all dependencies from pyproject.toml
uv sync

# Optional: Install dev dependencies (ruff for formatting)
uv pip install -e ".[dev]"

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

### Unsloth Studio (Google Colab) Setup

Unsloth Studio is highly recommended if you don't have a local GPU with >24GB VRAM.

1. Go to [Unsloth Studio](https://studio.unsloth.ai/).
2. Create a new notebook.
3. Run the Unsloth installation cell provided in the notebook.

## Phase 2: Data Collection & Processing (Local Machine)

_Note: Do this locally or on a cheap CPU VM. You don't need a GPU for data collection._

### GitHub Token Setup (Optional but Recommended)

The GitHub collection script will work with or without a token, but with higher rate limits when authenticated:

- **With `GITHUB_TOKEN` set**: 0.2s delay between API calls (60 requests/min)
- **Without token**: 0.4s delay between API calls (30 requests/min)

To speed up collection on large repositories, set your GitHub token:

```bash
export GITHUB_TOKEN="ghp_your_personal_access_token"
```

Generate a token at: https://github.com/settings/tokens (requires `repo` and `read:user` scopes)

### Running Data Collection

1. **Run Data Collection Pipeline:**

   ```bash
   uv run python scripts/01_collect_rust_book.py
   uv run python scripts/01b_collect_rust_releases.py
   uv run python scripts/02_collect_docs_rs.py
   uv run python scripts/02b_collect_cybersec.py
   uv run python scripts/03_collect_github.py
   uv run python scripts/04_collect_esp_rs.py
   uv run python scripts/05_collect_blogs.py
   ```

   Collection output appears in `data/raw/` organized by source. Each item is saved as JSON with:
   - `source`: Collection origin (rust_book, rustc_releases, docs_rs, cybersec, github, esp_rs, blogs, best_practices)
   - `source_type`: Category (official_docs, official_releases, api_docs, cybersec, github_repo, blogs, best_practices)
   - `url`: Original URL
   - `title`: Page/file title
   - `content`: Markdown-formatted content
   - `cutoff_date`: Knowledge cutoff date (e.g. for vulnerability intelligence & tooling)
   - `metadata`: Additional context (crate name, edition, version, diffs, etc.)
   - `collected_at`: ISO timestamp

   **Error Handling**: All errors are logged to stdout. If collection is interrupted, you can resume by re-running the same script—already-collected pages are cached and skipped.

   **Version Awareness**: Crate versions, GitHub release tags, changelog sections, and Rust editions are automatically extracted and partitioned under `data/raw/{source}/{version}/`. When new versions are pulled in subsequent runs, previous versions are preserved. See [VERSIONING.md](VERSIONING.md) for full details.

2. **Transform and Chunk Data:**

   ```bash
   uv run python scripts/06_transform_data.py
   ```

   _Outputs to `data/processed/all_chunks.jsonl`_
   **Data Quality**: The transform script applies several validation checks:
   - **Minimum content length**: 200 characters (filters low-value snippets)
   - **Boilerplate detection**: Removes copyright notices, "Follow us on X", "Subscribe", etc. (prevents LLM degradation)
   - **Navigation/footer filtering**: Skips content with >30% link density (nav menus are not useful for training)
   - **Version-aware deduplication**: Deduplicates using `sha256(source_type:library:version:content)`, ensuring shared code between different versions is preserved for each version while removing identical duplicates within the same version.
   - **Token counting**: Uses tiktoken to count tokens for training efficiency

   The script outputs a validation summary showing:
   - How many chunks were accepted vs. filtered at each stage
   - Reasons for rejection (too short, boilerplate, nav/footer)
   - Deduplication statistics and unique `(library, version)` pairs tracked

3. **Store in Vector Database (For Updates & Version-Aware RAG):**

   ```bash
   # Index all chunks into ChromaDB using Google DeepMind EmbeddingGemma-2 (~540MB)
   uv run python scripts/07_vector_store.py

   # Reset and re-index from scratch
   uv run python scripts/07_vector_store.py --reset

   # Query with version filtering
   uv run python scripts/07_vector_store.py --library axum --version 0.7.5 --query "routing and state"

   # Query migration guidance between versions
   uv run python scripts/07_vector_store.py --library axum --migrate-from 0.6.20 --migrate-to 0.7.5 --query "handlers"

   # List tracked libraries and versions
   uv run python scripts/07_vector_store.py --list-versions
   ```

4. **Create Training Dataset:**

   ```bash
   uv run python scripts/08_create_dataset.py
   ```

   _Outputs `data/datasets/train.jsonl` and `val.jsonl` in Alpaca format with version-grounded instructions, consecutive minor version diffs, edition guides, and 10-15% stratified boost._

   ```
   Final Dataset Summary:
   - Total chunks processed: 59099
   - Total samples: 69384
   - Version & Edition representation: 11.0%
   - Cybersecurity & LOTL representation: 8.0%
   - Knowledge Cutoff Date: 2026-10-11
   - Train split (98%): 67996
   - Val split (2%): 1388
   ```

5. **Evaluate Version & Edition Knowledge:**
   ```bash
   # Inspect benchmark questions:
   uv run python scripts/eval_version_knowledge.py --benchmark-only

   # Run benchmark evaluation on base model or trained LoRA adapter:
   uv run python scripts/eval_version_knowledge.py --model-path Qwen/Qwen3.5-4B --adapter-path models/qwen3_5_4b_rust_lora
   ```

## Versioning & Migration Architecture

For complete architectural details on multi-version retention, changelog section extraction, RAG querying, and Alpaca prompt generation, see [VERSIONING.md](VERSIONING.md).

## Code Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for code formatting and linting.

```bash
# Format all Python files
uv run ruff format .

# Check for lint issues
uv run ruff check .

# Fix lint issues automatically
uv run ruff check . --fix
```

## Phase 3: Training with Unsloth Studio

### Option A: Training IBM Granite 4.2 8B

Uses the base HuggingFace model (`ibm-granite/granite-4.2-8b`) or a local model directory. The export script will later merge LoRA and create GGUF.

1. Upload your project folder to Unsloth Studio (or upload `data/datasets/train.jsonl` directly).
2. Open `train_granite.py` in the studio.
3. Ensure the dataset path matches the uploaded location.
4. Run the script:

   **Option 1: Use HuggingFace model (downloads if not cached)**

   ```bash
   uv run python train_granite.py
   ```

   **Option 2: Use local model directory**

   ```bash
   uv run python train_granite.py --model-path /path/to/granite-4.2-8b
   ```

   **Advanced: override training sizing**

   ```bash
   uv run python train_granite.py --max-seq-length 4096 --batch-size 1 --gradient-accumulation-steps 8
   ```

5. Training parameters:
   - **Default sequence length:** `4096`
   - **Default per-device batch size:** `1`
   - **Default gradient accumulation:** `8`
   - **VRAM Requirement:** ~16GB (fits on RTX 3090/4090 or free Colab tier).
   - **Time:** ~2-3 hours for 3 epochs on 50k+ samples.

### Option B: Training Qwen 3.5 4B Instruct

1. Open `train_qwen.py`.
2. Run the script.
   - **VRAM Requirement:** ~12-16GB.
   - **Note:** This script uses the ChatML prompt format native to Qwen.

### Option C: Training NVIDIA Nemotron 3 Nano 4B

The Nemotron 3 Nano training script provides production-grade features including validation, checkpointing, and hardware-aware configuration.

1. **Review the configuration:**

   ```bash
   cat config/nemotron_training_config.yaml
   ```

2. **Validate configuration before training (recommended):**

   ```bash
   uv run python train_nemotron3_nano.py --dry-run
   ```

3. **Run training with default settings (RTX 4070 profile):**

   ```bash
   uv run python train_nemotron3_nano.py
   ```

4. **Alternative: Use different hardware profile:**

   ```bash
   # RTX 6000 (24GB VRAM)
   uv run python train_nemotron3_nano.py --hardware-profile rtx6000

   # A100 (40GB VRAM)
   uv run python train_nemotron3_nano.py --hardware-profile a100

   # Development (minimal resources, fast iteration)
   uv run python train_nemotron3_nano.py --hardware-profile dev
   ```

5. **Resume from checkpoint:**

   ```bash
   # Automatically resume from latest checkpoint:
   uv run python train_nemotron3_nano.py --resume

   # Or resume from a specific checkpoint:
   uv run python train_nemotron3_nano.py --resume-from-checkpoint models/nemotron3_nano_rust_lora/checkpoints/checkpoint-50
   ```

6. **Enable experiment tracking (optional W&B):**
   ```bash
   uv run python train_nemotron3_nano.py --wandb-project my-project --wandb-entity my-team
   ```

**Features:**

- **Validation Pipeline**: Checks CUDA availability, GPU memory, dependencies (including mamba-ssm for remote code), data existence, and JSONL schema
- **Hardware Profiles**: Predefined settings for RTX 4070 (12GB), RTX 6000 (24GB), A100 (40GB), A100 80GB, and dev profiles
- **Checkpointing**: Automatically saves checkpoints every 50 steps, resumable training supported
- **Experiment Tracking**: Saves training metadata to `models/nemotron3_nano_rust_lora/manifest.json` for artifact lineage
- **Flexible Configuration**: YAML-based config with CLI overrides for custom settings
- **Dry-Run Mode**: Validate setup without training

**Training Parameters (RTX 4070 default):**

- **Batch Size:** 1 per device
- **Gradient Accumulation:** 4
- **Sequence Length:** 1024
- **Learning Rate:** 2e-4
- **Warmup Steps:** 10
- **Epochs:** 3
- **VRAM:** ~12GB

**Advanced: Custom hyperparameters:**

```bash
uv run python train_nemotron3_nano.py \
  --batch-size 2 \
  --max-seq-length 2048 \
  --num-epochs 5 \
  --learning-rate 1e-4 \
  --verbose
```

### Option D: Training IFM K2-Horizon 3.7B (Recommended for Coding Agents)

IFM K2-Horizon-3.7B is a 3.7B dense model purpose-built for agentic execution, reasoning, and coding (512K native context). See [K2_HORIZON.md](K2_HORIZON.md) for complete details.

1. **Validate setup before training:**

   ```bash
   uv run python train_k2_horizon.py --dry-run
   ```

2. **Run training with default RTX 3090/4090 profile (bfloat16):**

   ```bash
   uv run python train_k2_horizon.py
   ```

3. **Or run with 12GB GPU profile (RTX 4070 4-bit QLoRA):**

   ```bash
   uv run python train_k2_horizon.py --hardware-profile rtx4070
   ```

4. **Serve base model + LoRA for AI coding agents:**
   ```bash
   uv run python serve_coding_agent.py --port 8000
   ```

### Training Configuration Details

All model recipes share a common core engine in [lib/](lib/) with standardized preflight checks, dual-backend model loading (Unsloth with automatic Hugging Face PEFT fallback), and inline hardware profiles:

- [train_k2_horizon.py](train_k2_horizon.py) (IFM K2-Horizon 3.7B) — See [K2_HORIZON.md](K2_HORIZON.md):
  - **Rank (r):** 16
  - **Alpha:** 16
  - **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  - **Optimizer:** AdamW 8-bit
  - **Scheduler:** Cosine
  - **Batch Size:** 1 per device, gradient accumulation 4 (effective 4, configurable via hardware profiles)
  - **Sequence Length:** 2048 (up to 8192 on A100/H100)
  - **VRAM:** ~12GB (4-bit QLoRA) to 24GB (bfloat16)
  - **Prompt Format:** Native K2-Horizon delimiters (`<|ifm|im_start|>`, `<|ifm|im_end|>`, `<ifm|think>`)
  - **Key Features:** Tailored for coding agents and tool-calling, dual-engine (Unsloth + Transformers/PEFT), dry-run mode, manifest tracking

- [train_granite.py](train_granite.py) (Granite 4.2 8B / 3B) — See [GRANITE.md](GRANITE.md):
  - **Rank (r):** 64
  - **Alpha:** 128
  - **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  - **Optimizer:** AdamW 8-bit
  - **Scheduler:** Cosine
  - **Batch Size:** 1 per device, gradient accumulation 8 (effective 8, configurable via hardware profiles)
  - **Sequence Length:** 4096
  - **VRAM:** ~16GB (4-bit QLoRA default)
  - **Prompt Format:** Alpaca instruction template
  - **Key Features:** Automatic 8B/3B variant detection, GGUF export via [scripts/10_export_ollama.py](scripts/10_export_ollama.py)

- [train_qwen.py](train_qwen.py) (Qwen 3.5 4B) — See [QWEN.md](QWEN.md):
  - **Rank (r):** 16
  - **Alpha:** 16
  - **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  - **Optimizer:** AdamW 8-bit
  - **Scheduler:** Cosine
  - **Batch Size:** 1 per device, gradient accumulation 4 (effective 4, configurable via hardware profiles)
  - **Sequence Length:** 1024
  - **VRAM:** ~12GB (16-bit bf16 recommended)
  - **Prompt Format:** ChatML instruction template
  - **Key Features:** Full 16-bit precision training, hardware profiles, preflight checks

- [train_qwen3_8.py](train_qwen3_8.py) (Qwen 3.8 27B) — See [QWEN3_8.md](QWEN3_8.md):
  - **Rank (r):** 16
  - **Alpha:** 16
  - **Target Modules:** All 12 hybrid attention & MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_a`, `in_proj_b`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`)
  - **Optimizer:** Dynamic (`adamw_torch` on macOS/MPS, `adamw_8bit` on CUDA)
  - **Scheduler:** Cosine
  - **Batch Size:** 1 per device, gradient accumulation 16 on Mac (effective 16)
  - **Sequence Length:** 512 (Mac memory-safe default), up to 4096 on workstation/cluster GPUs
  - **VRAM/RAM:** ~32GB-48GB unified RAM (with offloading) or 24-80GB VRAM
  - **Prompt Format:** Native ChatML template with reasoning `<think>` block support
  - **Key Features:** 27B architecture compatibility, macOS Apple Silicon support, resumable training via `--resume`, automatic step checkpointing

- [train_nemotron3_nano.py](train_nemotron3_nano.py) (Nemotron 3 Nano 4B) — See [NEMOTRON.md](NEMOTRON.md):
  - **Rank (r):** 16
  - **Alpha:** 16
  - **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  - **Optimizer:** AdamW 8-bit
  - **Scheduler:** Cosine
  - **Batch Size:** 1 per device, gradient accumulation 4 (effective 4, configurable per hardware profile)
  - **Sequence Length:** 1024 (configurable, up to 4096 on high-end GPUs)
  - **VRAM:** ~12GB (RTX 4070), scales with hardware profile
  - **Prompt Format:** ChatML-compatible with Rust expert persona
  - **Key Features:** Inline hardware profiles, checkpointing, validation pipeline, W&B integration, dry-run mode

- [train_alodlm.py](train_alodlm.py) (Amazon ALoDLM-8B) — See [ALODLM.md](ALODLM.md):
  - **Rank (r):** 16
  - **Alpha:** 16
  - **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  - **Optimizer:** AdamW 8-bit on CUDA, AdamW torch on MPS/CPU
  - **Scheduler:** Cosine
  - **Batch Size:** 1 per device, gradient accumulation 8 (configurable via hardware profiles)
  - **Sequence Length:** 1024 (macOS) / 2048 (CUDA default)
  - **VRAM/RAM:** ~16GB unified RAM (Mac) or 12GB–24GB VRAM (4-bit QLoRA default on CUDA)
  - **Prompt Format:** Native ChatML instruction format
  - **Key Features:** Qwen3-8B causal backbone alignment for adaptively looped diffusion modeling, dry-run preflight checks, resumable checkpointing

## Phase 4: Serving with AI Coding Agents

Load the trained LoRA adapter directly alongside the base model to use with AI coding agents like Cline, Roo Code, Continue.dev, Aider, and Cursor:

```bash
# Start OpenAI-compatible agent server on port 8000
uv run python serve_coding_agent.py \
  --model-path IFM/K2-Horizon-3.7B \
  --lora-path models/k2_horizon_rust_lora \
  --port 8000
```

In your coding agent settings:

- **Base URL:** `http://localhost:8000/v1`
- **Model ID:** `k2-horizon-3.7b-rust-lora`
- **API Key:** `EMPTY`

See [K2_HORIZON.md](K2_HORIZON.md) for full configuration details and vLLM / SGLang deployment commands.

## Phase 5: Exporting to Ollama

Once training is complete:

1. **Package the trained adapter with Script 09:**

   Script `scripts/09_create_modelfile.py` exports the trained LoRA adapter as an adapter-only package and writes a `Modelfile` with base model metadata and training economics.

   ```bash
   uv run python scripts/09_create_modelfile.py --model models/k2_horizon_rust_lora --name rust-k2-horizon --base k2_horizon

   uv run python scripts/09_create_modelfile.py --model models/granite_rust_lora --name rust-granite --base granite

   uv run python scripts/09_create_modelfile.py --model models/qwen3_5_4b_rust_lora --name rust-qwen --base qwen

   uv run python scripts/09_create_modelfile.py --model models/nemotron3_nano_rust_lora --name rust-nemotron --base nemotron
   ```

   Optionally, provide `--base-model` to override the base model ID or local path used in the generated `Modelfile`.

2. **Merge and export GGUF with Script 10:**

   Use `scripts/10_export_ollama.py` to merge the adapter with the base model and export a runnable GGUF package.

   **Option 1: Standard export (downloads base model from HuggingFace if not cached)**

   ```bash
   # For IFM K2-Horizon 3.7B
   uv run python scripts/10_export_ollama.py --model models/k2_horizon_rust_lora --name rust-k2-horizon --base IFM/K2-Horizon-3.7B

   # For Granite 4.2 8B
   uv run python scripts/10_export_ollama.py --model models/granite_rust_lora --name rust-granite --base ibm-granite/granite-4.2-8b

   # For Qwen 3.5 4B
   uv run python scripts/10_export_ollama.py --model models/qwen3_5_4b_rust_lora --name rust-qwen --base Qwen/Qwen3.5-4B

   # For Nemotron 3 Nano 4B
   uv run python scripts/10_export_ollama.py --model models/nemotron3_nano_rust_lora --name rust-nemotron --base NVIDIA/Nemotron-3-Nano-4B
   ```

   **Option 2: Use local GGUF or model file**

   ```bash
   uv run python scripts/10_export_ollama.py --model models/granite_rust_lora --name rust-granite --base /path/to/granite-4.2-8b-Q4_K_M.gguf
   ```

   _(This skips downloading from HuggingFace and uses your local file directly.)_

   _Script 10 merges the LoRA weights with the specified base model and quantizes the merged model to Q4_K_M._

3. **Import into Ollama:**

   ```bash
   ollama create rust-granite -f models/rust-granite_gguf/Modelfile
   ```

4. **Test the Model:**
   ```bash
   ollama run rust-granite "How do I set up a WiFi connection on an ESP32 using esp-wifi and async Rust?"
   ```

## Phase 6: Quantization & Unsloth Studio Export

For advanced quantization, standalone adapter compression, SVD rank reduction, and Unsloth Studio integration, see [QUANTIZIZE.md](QUANTIZIZE.md).

### Quick Unsloth Studio Export (Script 11)

Use [scripts/11_export_unsloth.py](scripts/11_export_unsloth.py) to export merged Hugging Face weights, standalone adapter bundles, or GGUF files with embedded Jinja chat templates for [Unsloth Studio](https://unsloth.ai/docs/new/studio/chat):

```bash
# 1. Export merged 16-bit model with chat_template.jinja for Unsloth Studio
uv run python scripts/11_export_unsloth.py \
  --model models/granite_rust_lora \
  --base ibm-granite/granite-4.2-8b \
  --mode merged-hf

# 2. Package standalone adapter with tokenizer and studio manifest
uv run python scripts/11_export_unsloth.py \
  --model models/k2_horizon_rust_lora \
  --base IFM/K2-Horizon-3.7B \
  --mode adapter

# 3. Direct GGUF quantization (Q4_K_M)
uv run python scripts/11_export_unsloth.py \
  --model models/qwen3_8_27b_rust_lora \
  --base Qwen/Qwen3.8-27B \
  --mode gguf \
  --quant q4_k_m
```

For the complete guide on creating an extreme 2-bit **$IQ2\_XS$** merge with **Multi-Token Prediction (MTP)** and an **Importance Matrix (imatrix)** for Qwen 3.8 27B, see [QUANTIZIZE.md](QUANTIZIZE.md#4-showcase-extreme-iq2_xs-merge-with-mtp-for-unsloth-studio).

## Phase 7: Maintenance & 6-Month Updates

Because we stored the raw data in ChromaDB (a local vector database), you do not need to re-scrape everything when Rust updates.

### Incremental Update Workflow:

1. **Collect only new data:**

   ```bash
   uv run python scripts/01_collect_rust_book.py  # Gets updated docs
   uv run python scripts/02_collect_docs_rs.py    # Gets new crate versions
   uv run python scripts/02b_collect_cybersec.py  # Gets updated RustSec advisories & tools
   ```

2. **Re-transform the new raw data:**

   ```bash
   uv run python scripts/06_transform_data.py
   ```

3. **Update the Vector Store:**
   Instead of re-indexing everything, update specific sources:

   ```python
   from scripts.07_vector_store import VectorStore
   vs = VectorStore()
   # Deletes old docs_rs chunks, adds new ones
   vs.update_source("docs_rs", "data/processed/docs_rs_chunks.jsonl")
   ```

4. **Re-generate Dataset & Re-train:**
   ```bash
   uv run python scripts/08_create_dataset.py
   # Re-run train_granite.py in Unsloth Studio
   ```

## References: IBM Granite 4.2 8B

- **Unsloth Documentation:** https://unsloth.ai/docs/models/ibm-granite-4.1
- **HuggingFace Model Card (GGUF):** https://huggingface.co/ibm-granite/granite-4.2-8b-GGUF
- **HuggingFace Blog (Granite 4.1):** https://huggingface.co/blog/ibm-granite/granite-4-1
- **IBM Granite Documentation:** https://www.ibm.com/granite/docs/models/granite4-1

## References: Qwen 3.5 4B

- **Unsloth Documentation:** https://unsloth.ai/docs/models/qwen
- **HuggingFace Model Card:** https://huggingface.co/Qwen/Qwen3.5-4B
- **Qwen Documentation:** https://qwenlm.github.io/

## References: NVIDIA Nemotron 3 Nano 4B

- **Unsloth Documentation:** https://unsloth.ai/docs/models/nemotron-3
- **HuggingFace Model Card:** https://huggingface.co/NVIDIA/Nemotron-3-Nano-4B
- **Nemotron 3.5 Lightning Recipe:** https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/lightning35/README.md
- **NVIDIA NeMo Documentation:** https://github.com/NVIDIA-NeMo/Nemotron

## References: IFM K2-Horizon 3.7B

- **HuggingFace Model Card:** https://huggingface.co/IFM/K2-Horizon-3.7B
- **Institute of Foundation Models (IFM):** https://ifm.ai/
- **SGLang Cookbook Recipe:** https://docs.sglang.io/cookbook/autoregressive/IFM/K2-Horizon
- **Detailed K2-Horizon Guide:** [K2_HORIZON.md](K2_HORIZON.md)

## License

[MIT](./LICENSE)
