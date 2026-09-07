#!/usr/bin/env python3
import hashlib
import json
import logging
import re
from pathlib import Path

import tiktoken
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("rust_lora.transform")

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

        # Boilerplate patterns to detect and remove
        self.boilerplate_patterns = [
            r"©\s*20\d{2}",  # Copyright notices
            r"copyright",
            r"all rights reserved",
            r"follow us on",
            r"subscribe",
            r"sign up for",
            r"terms of service",
            r"privacy policy",
            r"disclaimer",
            r"made with.*by",
            r"powered by",
        ]

        # Statistics tracking
        self.skipped_too_short = 0
        self.skipped_boilerplate = 0
        self.skipped_low_quality = 0
        self.skipped_invalid = 0
        self.valid_chunks = 0

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text)) if self.tokenizer else len(text) // 4

    def clean(self, text):
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    def has_boilerplate(self, text):
        """Check if text contains boilerplate patterns indicating low-quality content."""
        text_lower = text.lower()
        for pattern in self.boilerplate_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False

    def is_likely_nav_or_footer(self, text):
        """Detect if content is primarily navigation or footer (high link density)."""
        # Count markdown links
        links = len(re.findall(r"\[.*?\]\(.*?\)", text))
        # Count words
        words = len(text.split())

        if words < 50:  # Too short to judge
            return False

        # If more than 30% is links, likely navigation
        link_density = links / max(words / 5, 1)  # Rough: links count as ~5 words
        return link_density > 0.3

    def validate_content(self, doc_data):
        """Validate content and return (is_valid, skip_reason)."""
        content = self.clean(
            doc_data.get("content", "")
            or doc_data.get("question_body", "") + doc_data.get("answer_body", "")
        )

        # Check minimum length (stricter than before)
        if len(content) < 200:
            self.skipped_too_short += 1
            return False, "content_too_short"

        # Check for required fields
        if "source" not in doc_data:
            self.skipped_invalid += 1
            return False, "missing_source"

        if "url" not in doc_data:
            self.skipped_invalid += 1
            return False, "missing_url"

        # Check for boilerplate
        if self.has_boilerplate(content):
            self.skipped_boilerplate += 1
            return False, "boilerplate_detected"

        # Check for navigation/footer dominated content
        if self.is_likely_nav_or_footer(content):
            self.skipped_low_quality += 1
            return False, "nav_footer_content"

        self.valid_chunks += 1
        return True, None

    def count_json_files(self, dir_path):
        if not dir_path.exists():
            return 0
        return sum(1 for _ in dir_path.rglob("*.json") if _.is_file())

    def process_dir(self, dir_path, source_type):
        chunks = []
        if not dir_path.exists():
            return chunks
        for fp in tqdm(list(dir_path.rglob("*.json")), desc=source_type, leave=False):
            try:
                with open(fp) as f:
                    doc = json.load(f)

                # Validate content before adding to chunks
                is_valid, skip_reason = self.validate_content(doc)
                if not is_valid:
                    logger.debug(f"Skipped {fp.name}: {skip_reason}")
                    continue

                content = self.clean(
                    doc.get("content", "")
                    or doc.get("question_body", "") + doc.get("answer_body", "")
                )
                chunks.append(
                    {
                        "source_type": source_type,
                        "content": content,
                        "metadata": doc,
                        "hash": hashlib.sha256(content.encode()).hexdigest()[:16],
                        "tokens": self.count_tokens(content),
                    }
                )
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in {fp}: {e}")
                self.skipped_invalid += 1
            except Exception as e:
                logger.warning(f"Error processing {fp}: {type(e).__name__}: {e}")
                self.skipped_invalid += 1
        return chunks

    def run_all(self):
        print("Transforming data...")
        sources = [
            ("rust_book", "rust_book"),
            ("docs_rs", "docs_rs"),
            ("github", "github"),
            ("esp_rs", "esp_rs"),
            ("blogs", "blogs"),
            ("best_practices", "best_practices"),
        ]
        all_chunks = []
        source_summary = []
        for raw_subdir, source_type in sources:
            raw_path = RAW_DIR / raw_subdir
            raw_count = self.count_json_files(raw_path)
            chunks = self.process_dir(raw_path, source_type)
            all_chunks.extend(chunks)
            source_summary.append((source_type, raw_count, len(chunks)))

        print("\nTransformation results by source:")
        for source_type, raw_count, chunk_count in source_summary:
            print(f"  {source_type}: {raw_count} raw docs -> {chunk_count} valid chunks")

        # Deduplicate
        seen = set()
        unique = [c for c in all_chunks if not (c["hash"] in seen or seen.add(c["hash"]))]

        with open(self.output_dir / "all_chunks.jsonl", "w") as f:
            for c in unique:
                f.write(json.dumps(c) + "\n")

        total_raw = sum(raw_count for _, raw_count, _ in source_summary)
        total_chunks = len(all_chunks)
        total_unique = len(unique)

        print("\nValidation Summary:")
        print(f"  Valid chunks: {self.valid_chunks}")
        print(f"  Skipped (too short): {self.skipped_too_short}")
        print(f"  Skipped (boilerplate): {self.skipped_boilerplate}")
        print(f"  Skipped (nav/footer): {self.skipped_low_quality}")
        print(f"  Skipped (invalid): {self.skipped_invalid}")
        print("\nFinal result:")
        print(
            f"  Saved {total_unique} unique chunks from {total_chunks} validated chunks ({total_raw} raw source docs)."
        )
        if total_chunks > 0:
            dedup_ratio = 100 * (total_chunks - total_unique) / total_chunks
            print(
                f"  Deduplication removed {total_chunks - total_unique} duplicates ({dedup_ratio:.1f}%)."
            )


if __name__ == "__main__":
    DataTransformer().run_all()
