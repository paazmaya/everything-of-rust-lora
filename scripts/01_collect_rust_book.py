#!/usr/bin/env python3
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import html2text
import requests
from bs4 import BeautifulSoup
from cache_utils import CachedSession

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CONFIG_DIR = BASE_DIR / "config"


class RustDocCollector:
    def __init__(self):
        self.output_base = RAW_DIR
        session = requests.Session()
        session.headers.update({"User-Agent": "RustLoRA/1.0"})
        self.session = CachedSession(session)
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_images = True
        self.h2t.body_width = 0
        self.visited = set()

    def _save(self, source, url, title, content, metadata, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256(content.encode()).hexdigest()[:16]
        safe_title = re.sub(r"[^\w\s-]", "", title).strip()[:50].replace(" ", "_")
        with open(out_dir / f"{safe_title}_{h}.json", "w") as f:
            json.dump(
                {
                    "source": source,
                    "url": url,
                    "title": title,
                    "content": content,
                    "metadata": metadata,
                    "collected_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

    def count_output_files(self, out_dir):
        if not out_dir.exists():
            return 0
        return sum(1 for _ in out_dir.rglob("*.json") if _.is_file())

    def collect_site_recursive(self, url, source_name, out_dir, base_domain, base_path):
        """Recursively collect all pages from a site, staying within base_path."""
        normalized_url = urldefrag(url)[0]
        if normalized_url in self.visited:
            return
        self.visited.add(normalized_url)
        sys.stdout.write(".")
        sys.stdout.flush()
        try:
            r = self.session.get(normalized_url, timeout=30)
            if not r:
                return
            s = BeautifulSoup(r.text, "lxml")
            main = s.find("main") or s.find("div", class_="content")
            if main and len(main.text) > 100:
                title = s.find("h1").text.strip() if s.find("h1") else "Untitled"
                md = self.h2t.handle(str(main))
                self._save(source_name, normalized_url, title, md, {"book": source_name}, out_dir)
            # Find and queue all linked pages
            for link in s.find_all("a", href=True):
                href = link["href"]
                if href.startswith("mailto:") or href.startswith("javascript:"):
                    continue
                next_url = urljoin(normalized_url, href)
                parsed = urlparse(next_url)
                if parsed.netloc != base_domain:
                    continue
                if not parsed.path.startswith(base_path):
                    continue
                if parsed.path.endswith(".html") or parsed.path.endswith("/"):
                    self.collect_site_recursive(
                        next_url, source_name, out_dir, base_domain, base_path
                    )
        except Exception:
            pass

    def collect_site(self, base_url, source_name, out_dir):
        output_dir = self.output_base / out_dir
        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc
        base_path = parsed_base.path if parsed_base.path.endswith("/") else parsed_base.path + "/"
        self.collect_site_recursive(base_url, source_name, output_dir, base_domain, base_path)

    def run_all(self):
        print("Collecting official Rust documentation...")
        total_before = sum(
            self.count_output_files(RAW_DIR / part)
            for part in [
                "rust_book",
                "rust_by_example",
                "rust_reference",
                "rustonomicon",
                "cargo_book",
            ]
        )
        self.visited.clear()
        self.collect_site("https://doc.rust-lang.org/book/", "rust_book", "rust_book")
        self.visited.clear()
        self.collect_site(
            "https://doc.rust-lang.org/rust-by-example/", "rust_by_example", "rust_by_example"
        )
        self.visited.clear()
        self.collect_site(
            "https://doc.rust-lang.org/reference/", "rust_reference", "rust_reference"
        )
        self.visited.clear()
        self.collect_site("https://doc.rust-lang.org/nomicon/", "rustonomicon", "rustonomicon")
        self.visited.clear()
        self.collect_site("https://doc.rust-lang.org/cargo/", "cargo_book", "cargo_book")
        total_after = sum(
            self.count_output_files(RAW_DIR / part)
            for part in [
                "rust_book",
                "rust_by_example",
                "rust_reference",
                "rustonomicon",
                "cargo_book",
            ]
        )
        stats = self.session.get_stats()
        print(f"\nCollected {total_after - total_before} new raw documents, {total_after} total.")
        print(
            f"Cache stats: {stats['fetched']} fetched, {stats['skipped']} skipped (out of {stats['total_checked']} total)"
        )


if __name__ == "__main__":
    RustDocCollector().run_all()
