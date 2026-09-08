# Training a LoRa that knows everything about the Rust language

This project contains the complete pipeline to train a specialized LoRA adapter for Rust programming, covering the language standard, top 100 libraries, best practices, and ESP32/Embedded IoT development.

## Architecture Overview

1. **Data Collection** (Scripts 01-05): Scrapes Rust Book, docs.rs, GitHub, ESP-RS, and Blogs.
2. **Transformation** (Script 06): Cleans, chunks, and deduplicates the data.
3. **Vector/Graph Storage** (Script 07): Stores data in ChromaDB for easy 6-month incremental updates.
4. **Dataset Creation** (Script 08): Converts chunks into Alpaca instruction-response format.
5. **Training** (`train_granite.py`, `train_qwen.py`): Trains LoRA using Unsloth.
6. **Export** (Script 09): Packages the trained LoRA adapter with metadata and a `Modelfile`.
7. **Merge + GGUF** (Script 10): Merges the adapter with the base model and exports a runnable GGUF bundle for Ollama.

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
   uv run python scripts/02_collect_docs_rs.py
   uv run python scripts/03_collect_github.py
   uv run python scripts/04_collect_esp_rs.py
   uv run python scripts/05_collect_blogs.py
   ```

   Collection output appears in `data/raw/` organized by source. Each item is saved as JSON with:
   - `source`: Collection origin (rust_book, docs_rs, github, esp_rs, blogs, best_practices)
   - `source_type`: Category (official_docs, api_docs, github_repo, blogs, best_practices)
   - `url`: Original URL
   - `title`: Page/file title
   - `content`: Markdown-formatted content
   - `metadata`: Additional context (crate name, repo name, etc.)
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
   # Index all chunks into ChromaDB
   uv run python scripts/07_vector_store.py

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
   _Outputs `data/datasets/train.jsonl` and `val.jsonl` in Alpaca format with version-grounded instructions and migration questions._

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

### Training Configuration Details

The current implementation uses three separate LoRA recipes:

- `train_granite.py` (Granite 4.2 8B):
  - **Rank (r):** 64
  - **Alpha:** 128
  - **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  - **Optimizer:** AdamW 8-bit
  - **Scheduler:** Cosine
  - **Batch Size:** 1 per device, gradient accumulation 8 (effective 8)
  - **Sequence Length:** 4096
  - **VRAM:** ~16GB

- `train_qwen.py` (Qwen 3.5 4B):
  - **Rank (r):** 16
  - **Alpha:** 16
  - **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  - **Optimizer:** AdamW 8-bit
  - **Scheduler:** Cosine
  - **Batch Size:** 1 per device, gradient accumulation 4 (effective 4)
  - **Sequence Length:** 1024
  - **VRAM:** ~12GB

- `train_nemotron3_nano.py` (Nemotron 3 Nano 4B):
  - **Rank (r):** 16
  - **Alpha:** 16
  - **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  - **Optimizer:** AdamW 8-bit
  - **Scheduler:** Cosine
  - **Batch Size:** 1 per device, gradient accumulation 4 (effective 4, configurable per hardware profile)
  - **Sequence Length:** 1024 (configurable, up to 4096 on high-end GPUs)
  - **VRAM:** ~12GB (RTX 4070), scales with hardware profile
  - **Prompt Format:** ChatML-compatible with Rust expert persona
  - **Key Features:** Hardware profiles, checkpointing, validation pipeline, W&B integration, dry-run mode

## Phase 4: Exporting to Ollama

Once training is complete (either locally or downloaded from Unsloth Studio):

1. **Package the trained adapter with Script 09:**

   Script `scripts/09_create_modelfile.py` now exports the trained LoRA adapter as an adapter-only package and writes a `Modelfile` with base model metadata and training economics.

   ```bash
   uv run python scripts/09_create_modelfile.py --model models/granite_rust_lora --name rust-granite --base granite

   uv run python scripts/09_create_modelfile.py --model models/qwen3_5_4b_rust_lora --name rust-qwen --base qwen

   uv run python scripts/09_create_modelfile.py --model models/nemotron3_nano_rust_lora --name rust-nemotron --base nemotron
   ```

   Optionally, provide `--base-model` to override the base model ID or local path used in the generated `Modelfile`.

2. **Merge and export GGUF with Script 10:**

   Use `scripts/10_export_ollama.py` to merge the adapter with the base model and export a runnable GGUF package.

   **Option 1: Standard export (downloads base model from HuggingFace if not cached)**

   ```bash
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

## Phase 5: Maintenance & 6-Month Updates

Because we stored the raw data in ChromaDB (a local vector database), you do not need to re-scrape everything when Rust updates.

### Incremental Update Workflow:

1. **Collect only new data:**

   ```bash
   python scripts/01_collect_rust_book.py  # Gets updated docs
   python scripts/02_collect_docs_rs.py    # Gets new crate versions
   ```

2. **Re-transform the new raw data:**

   ```bash
   python scripts/06_transform_data.py
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

## License

[MIT](./LICENSE)
