#!/usr/bin/env python3
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import html2text
import requests
from bs4 import BeautifulSoup
from cache_utils import CachedSession
from tqdm import tqdm

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

    def collect_site(self, base_url, source_name, out_dir):
        print(f"\nCollecting {source_name}...")
        output_dir = self.output_base / out_dir
        try:
            resp = self.session.get(base_url, timeout=30)
            if not resp:
                print(f"  {source_name}: Not modified (using cache)")
                return
            soup = BeautifulSoup(resp.text, "lxml")
            links = {
                base_url + link["href"]
                for link in soup.find_all("a", href=True)
                if link["href"].endswith(".html") and not link["href"].startswith("http")
            }
            for url in tqdm(links):
                try:
                    r = self.session.get(url, timeout=30)
                    if not r:  # Content not modified
                        continue
                    s = BeautifulSoup(r.text, "lxml")
                    main = s.find("main") or s.find("div", class_="content")
                    if main and len(main.text) > 100:
                        title = s.find("h1").text.strip() if s.find("h1") else "Untitled"
                        md = self.h2t.handle(str(main))
                        self._save(source_name, url, title, md, {"book": source_name}, output_dir)
                except Exception:
                    pass
        except Exception as e:
            print(f"Error: {e}")

    def run_all(self):
        self.collect_site("https://doc.rust-lang.org/book/", "rust_book", "rust_book")
        self.collect_site(
            "https://doc.rust-lang.org/rust-by-example/", "rust_by_example", "rust_by_example"
        )
        self.collect_site(
            "https://doc.rust-lang.org/reference/", "rust_reference", "rust_reference"
        )
        self.collect_site("https://doc.rust-lang.org/nomicon/", "rustonomicon", "rustonomicon")
        self.collect_site("https://doc.rust-lang.org/cargo/", "cargo_book", "cargo_book")
        stats = self.session.get_stats()
        print(
            f"\nCache stats: {stats['fetched']} fetched, {stats['skipped']} skipped (out of {stats['total_checked']} total)"
        )


if __name__ == "__main__":
    RustDocCollector().run_all()
