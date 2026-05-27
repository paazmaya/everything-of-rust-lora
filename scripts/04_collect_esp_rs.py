#!/usr/bin/env python3
import hashlib
import json
import time
from pathlib import Path

import html2text
import requests
from bs4 import BeautifulSoup
from cache_utils import CachedSession
from tqdm import tqdm

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

    def collect_site(self, base_url, out_rel):
        out_dir = self.output_base / out_rel
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            resp = self.session.get(base_url, timeout=30)
            if not resp:
                print(f"  {out_rel}: Not modified (using cache)")
                return
            soup = BeautifulSoup(resp.text, "lxml")
            links = {
                base_url + link["href"] if not link["href"].startswith("http") else link["href"]
                for link in soup.find_all("a", href=True)
                if link["href"].endswith(".html")
            }
            for url in tqdm(links, desc=out_rel.split("/")[-1], leave=False):
                try:
                    time.sleep(0.1)
                    r = self.session.get(url, timeout=30)
                    if not r:  # Content not modified
                        continue
                    s = BeautifulSoup(r.text, "lxml")
                    main = s.find("main") or s.find("article")
                    if main and len(main.text) > 100:
                        md = self.h2t.handle(str(main))
                        h = hashlib.sha256(md.encode()).hexdigest()[:16]
                        with open(out_dir / f"{h}.json", "w") as f:
                            json.dump({"source": "esp_rs", "url": url, "content": md}, f)
                except Exception:
                    pass
        except Exception:
            pass

    def run_all(self):
        print("Collecting ESP-RS...")
        for url, rel in self.sites:
            self.collect_site(url, rel)
        stats = self.session.get_stats()
        print(
            f"Cache stats: {stats['fetched']} fetched, {stats['skipped']} skipped (out of {stats['total_checked']} total)"
        )


if __name__ == "__main__":
    ESPCollector().run_all()
