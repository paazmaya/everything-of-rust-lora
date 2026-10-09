#!/bin/bash
# Master pipeline runner
set -e

echo "========================================"
echo " Starting Rust LoRA Data Pipeline"
echo "========================================"

if [ -z "$GITHUB_TOKEN" ]; then
    echo "ERROR: GITHUB_TOKEN environment variable is required!"
    echo "Set it with: export GITHUB_TOKEN='your_github_token'"
    exit 1
fi

echo "GITHUB_TOKEN found. Proceeding with pipeline."

echo "[1/9] Collecting official Rust documentation..."
uv run python scripts/01_collect_rust_book.py

echo "[2/9] Collecting rustc releases, changelogs, and version diffs..."
uv run python scripts/01b_collect_rust_releases.py

echo "[3/9] Collecting docs.rs crate documentation..."
uv run python scripts/02_collect_docs_rs.py

echo "[4/9] Collecting GitHub READMEs and examples..."
uv run python scripts/03_collect_github.py

echo "[5/9] Collecting ESP-RS and Embedded docs..."
uv run python scripts/04_collect_esp_rs.py

echo "[6/9] Collecting Blogs and Best Practices..."
uv run python scripts/05_collect_blogs.py

echo "[7/9] Transforming and chunking data..."
uv run python scripts/06_transform_data.py

echo "[8/9] Indexing into Vector Database..."
uv run python scripts/07_vector_store.py

echo "[9/9] Creating Training Dataset..."
uv run python scripts/08_create_dataset.py

echo "========================================"
echo " Data Pipeline Complete!"
echo " Next: Train LoRA with:"
echo "   - train_k2_horizon.py (IFM K2-Horizon 3.7B - recommended for coding agents)"
echo "   - train_granite.py (IBM Granite 4.2 8B)"
echo "   - train_qwen.py (Qwen 3.5 4B)"
echo "   - train_qwen3_8.py (Qwen 3.8 27B)"
echo "   - train_nemotron3_nano.py (Nemotron 3 Nano 4B)"
echo "========================================"
