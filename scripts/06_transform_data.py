#!/usr/bin/env python3
import hashlib
import json
import re
from pathlib import Path

import tiktoken
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


class DataTransformer:
    def __init__(self):
        self.output_dir = PROCESSED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text)) if self.tokenizer else len(text) // 4

    def clean(self, text):
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def process_dir(self, dir_path, source_type):
        chunks = []
        if not dir_path.exists():
            return chunks
        for fp in tqdm(list(dir_path.rglob("*.json")), desc=source_type, leave=False):
            try:
                with open(fp) as f:
                    doc = json.load(f)
                content = self.clean(
                    doc.get("content", "")
                    or doc.get("question_body", "") + doc.get("answer_body", "")
                )
                if len(content) > 100:
                    chunks.append(
                        {
                            "source_type": source_type,
                            "content": content,
                            "metadata": doc,
                            "hash": hashlib.sha256(content.encode()).hexdigest()[:16],
                            "tokens": self.count_tokens(content),
                        }
                    )
            except Exception:
                pass
        return chunks

    def run_all(self):
        print("Transforming data...")
        all_chunks = []
        all_chunks.extend(self.process_dir(RAW_DIR / "rust_book", "rust_book"))
        all_chunks.extend(self.process_dir(RAW_DIR / "docs_rs", "docs_rs"))
        all_chunks.extend(self.process_dir(RAW_DIR / "github", "github"))
        all_chunks.extend(self.process_dir(RAW_DIR / "esp_rs", "esp_rs"))
        all_chunks.extend(self.process_dir(RAW_DIR / "blogs", "blogs"))
        all_chunks.extend(self.process_dir(RAW_DIR / "best_practices", "best_practices"))

        # Deduplicate
        seen = set()
        unique = [c for c in all_chunks if not (c["hash"] in seen or seen.add(c["hash"]))]

        with open(self.output_dir / "all_chunks.jsonl", "w") as f:
            for c in unique:
                f.write(json.dumps(c) + "\n")
        print(f"Saved {len(unique)} unique chunks.")


if __name__ == "__main__":
    DataTransformer().run_all()
