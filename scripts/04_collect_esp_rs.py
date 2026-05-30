#!/usr/bin/env python3
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import html2text
import requests
from bs4 import BeautifulSoup
from cache_utils import CachedSession

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"


class ESPCollector:
    def __init__(self):
        self.output_base = RAW_DIR
        session = requests.Session()
        session.headers.update({"User-Agent": "RustLoRA/1.0"})
        self.session = CachedSession(session)
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_images = True
        self.sites = [
            ("https://docs.esp-rs.org/book/", "esp_rs/book"),
            ("https://docs.esp-rs.org/no-std-training/", "esp_rs/no_std"),
            ("https://docs.espressif.com/projects/rust/esp-rust/latest/", "esp_rs/espressif"),
        ]
        self.visited = set()

    def count_files(self, out_dir):
        if not out_dir.exists():
            return 0
        return sum(1 for _ in out_dir.rglob("*.json") if _.is_file())

    def collect_page_recursive(self, url, base_domain, base_path, out_rel):
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
            main = s.find("main") or s.find("article")
            if main and len(main.text) > 100:
                md = self.h2t.handle(str(main))
                h = hashlib.sha256(md.encode()).hexdigest()[:16]
                out_dir = self.output_base / out_rel
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / f"{h}.json", "w") as f:
                    json.dump({"source": "esp_rs", "url": normalized_url, "content": md}, f)
            # Find and queue all linked pages
            for link in s.find_all("a", href=True):
                href = link.get("href", "")
                if href.startswith("mailto:") or href.startswith("javascript:"):
                    continue
                next_url = urljoin(normalized_url, href)
                parsed = urlparse(next_url)
                if parsed.netloc != base_domain:
                    continue
                if not parsed.path.startswith(base_path):
                    continue
                if parsed.path.endswith(".html") or parsed.path.endswith("/"):
                    self.collect_page_recursive(next_url, base_domain, base_path, out_rel)
        except Exception:
            pass

    def collect_site(self, base_url, out_rel):
        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc
        base_path = parsed_base.path if parsed_base.path.endswith("/") else parsed_base.path + "/"
        self.collect_page_recursive(base_url, base_domain, base_path, out_rel)

    def run_all(self):
        print("Collecting ESP-RS...")
        before_count = sum(self.count_files(self.output_base / rel) for _, rel in self.sites)
        for url, rel in self.sites:
            self.collect_site(url, rel)
        after_count = sum(self.count_files(self.output_base / rel) for _, rel in self.sites)
        stats = self.session.get_stats()
        print(f"Collected {after_count - before_count} new documents, {after_count} total.")
        print(
            f"Cache stats: {stats['fetched']} fetched, {stats['skipped']} skipped (out of {stats['total_checked']} total)"
        )


if __name__ == "__main__":
    ESPCollector().run_all()
