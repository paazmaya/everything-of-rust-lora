# Training a LoRa that knows everything about the Rust language

This project contains the complete pipeline to train a specialized LoRA adapter for Rust programming, covering the language standard, top 100 libraries, best practices, and ESP32/Embedded IoT development.

## Architecture Overview

1. **Data Collection** (Scripts 01-05): Scrapes Rust Book, docs.rs, GitHub, ESP-RS, Blogs, and StackOverflow.
2. **Transformation** (Script 06): Cleans, chunks, and deduplicates the data.
3. **Vector/Graph Storage** (Script 07): Stores data in ChromaDB for easy 6-month incremental updates.
4. **Dataset Creation** (Script 08): Converts chunks into Alpaca instruction-response format.
5. **Training** (`train_granite.py`, `train_qwen.py`): Trains LoRA using Unsloth.
6. **Export** (Script 09): Merges LoRA and converts to GGUF for Ollama.

---

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

---

## Phase 2: Data Collection & Processing (Local Machine)

*Note: Do this locally or on a cheap CPU VM. You don't need a GPU for data collection.*

1. **Run Data Collection Pipeline:**
   ```bash
   uv run python scripts/01_collect_rust_book.py
   uv run python scripts/02_collect_docs_rs.py
   uv run python scripts/03_collect_github.py
   uv run python scripts/04_collect_esp_rs.py
   uv run python scripts/05_collect_blogs.py
   ```
   *Tip: If GitHub rate limits you, set `export GITHUB_TOKEN="your_token"`.*

2. **Transform and Chunk Data:**
   ```bash
   uv run python scripts/06_transform_data.py
   ```
   *Outputs to `data/processed/all_chunks.jsonl`*

3. **Store in Vector Database (For Updates):**
   ```bash
   uv run python scripts/07_vector_store.py
   ```

4. **Create Training Dataset:**
   ```bash
   uv run python scripts/08_create_dataset.py
   ```
   *Outputs `data/datasets/train.jsonl` and `val.jsonl` in Alpaca format.*

---

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

---

## Phase 3: Training with Unsloth Studio

### Option A: Training IBM Granite 8B Code Instruct

1. Upload your project folder to Unsloth Studio (or upload `data/datasets/train.jsonl` directly).
2. Open `train_granite.py` in the studio.
3. Ensure the dataset path matches the uploaded location.
4. Run the script.
   - **VRAM Requirement:** ~16GB (fits on RTX 3090/4090 or free Colab tier).
   - **Time:** ~2-3 hours for 3 epochs on 50k+ samples.

### Option B: Training Qwen 2.5 7B Instruct

1. Open `train_qwen.py`.
2. Run the script.
   - **VRAM Requirement:** ~16GB.
   - **Note:** This script uses the ChatML prompt format native to Qwen.

### Training Configuration Details
Both scripts use the following optimized LoRA settings:
- **Rank (r):** 64
- **Alpha:** 128
- **Target Modules:** All attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Optimizer:** AdamW 8-bit
- **Scheduler:** Cosine
- **Batch Size:** 4 per device, 4 gradient accumulation steps (Effective batch size = 16).

---

## Phase 4: Exporting to Ollama

Once training is complete (either locally or downloaded from Unsloth Studio):

1. **Export to GGUF:**
   ```bash
   # For Granite
   uv run python scripts/09_export_ollama.py --model models/granite_rust_lora --name rust-granite --base ibm-granite/granite-8b-code-instruct

   # For Qwen
   uv run python scripts/09_export_ollama.py --model models/qwen_rust_lora --name rust-qwen --base Qwen/Qwen2.5-7B-Instruct
   ```
   *This script automatically merges the LoRA weights and quantizes them to `q4_k_m` (4-bit).*

2. **Import into Ollama:**
   ```bash
   ollama create rust-granite -f models/rust-granite_gguf/Modelfile
   ```

3. **Test the Model:**
   ```bash
   ollama run rust-granite "How do I set up a WiFi connection on an ESP32 using esp-wifi and async Rust?"
   ```

---

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
