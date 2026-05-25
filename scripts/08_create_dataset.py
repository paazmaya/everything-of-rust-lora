#!/usr/bin/env python3
import json
import random
from pathlib import Path

from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATASETS_DIR = BASE_DIR / "data" / "datasets"


def format_alpaca(chunk):
    source = chunk["source_type"]
    content = chunk["content"]
    meta = chunk.get("metadata", {})

    if source == "stack_overflow":
        instruction = "Answer this Rust programming question accurately."
        input_text = f"Question: {meta.get('title', '')}\n\n{meta.get('question_body', '')}"
        output_text = meta.get("answer_body", "")
    elif source == "docs_rs":
        instruction = f"Explain the Rust crate '{meta.get('crate_name', 'unknown')}' and its usage based on the documentation."
        input_text = f"Focus on: {meta.get('title', 'general usage')}"
        output_text = content
    elif source == "esp_rs":
        instruction = "Provide details about ESP32 and embedded Rust development."
        input_text = f"Topic: {meta.get('title', 'ESP32 Rust')}"
        output_text = content
    else:  # Books, blogs, best practices
        instruction = "Explain this Rust concept or best practice."
        input_text = f"Topic: {meta.get('title', 'Rust')}"
        output_text = content

    if len(output_text) < 50:
        return None
    return {"instruction": instruction, "input": input_text, "output": output_text}


def main():
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    chunks_file = PROCESSED_DIR / "all_chunks.jsonl"

    print("Creating Alpaca dataset...")
    dataset = []
    with open(chunks_file) as f:
        for line in tqdm(f):
            chunk = json.loads(line)
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

    print(f"Created {len(dataset)} samples. Train: {split}, Val: {len(dataset) - split}")


if __name__ == "__main__":
    main()
