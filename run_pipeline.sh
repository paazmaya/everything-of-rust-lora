#!/bin/bash
# Master pipeline runner
set -e

echo "========================================"
echo " Starting Rust LoRA Data Pipeline"
echo "========================================"

echo "[1/5] Collecting official Rust documentation..."
uv run python scripts/01_collect_rust_book.py

echo "[2/5] Collecting docs.rs crate documentation..."
uv run python scripts/02_collect_docs_rs.py

echo "[3/5] Collecting GitHub READMEs and examples..."
uv run python scripts/03_collect_github.py

echo "[4/5] Collecting ESP-RS and Embedded docs..."
uv run python scripts/04_collect_esp_rs.py

echo "[5/5] Collecting Blogs and Best Practices..."
uv run python scripts/05_collect_blogs.py

echo "[6/6] Transforming and chunking data..."
uv run python scripts/06_transform_data.py

echo "[7/7] Indexing into Vector Database..."
uv run python scripts/07_vector_store.py

echo "[8/8] Creating Training Dataset..."
uv run python scripts/08_create_dataset.py

echo "========================================"
echo " Data Pipeline Complete!"
echo " Next: Upload 'data/datasets/train.jsonl' to Unsloth Studio and run train_granite.py"
echo "========================================"
