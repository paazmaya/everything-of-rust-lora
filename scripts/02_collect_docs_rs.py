#!/usr/bin/env python3
import hashlib
import json
import time
from pathlib import Path

import html2text
import requests
import yaml
from bs4 import BeautifulSoup
from tqdm import tqdm

from cache_utils import CachedSession

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CONFIG_DIR = BASE_DIR / "config"


class DocsRsCollector:
    def __init__(self):
        self.output_base = RAW_DIR
        session = requests.Session()
        session.headers.update({"User-Agent": "RustLoRA/1.0"})
        self.session = CachedSession(session)
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_images = True
        self.h2t.body_width = 0
        with open(CONFIG_DIR / "libraries.yaml") as f:
            self.config = yaml.safe_load(f)
        self.crates = [item for cat in self.config["libraries"].values() for item in cat]

    def collect_crate(self, crate):
        name = crate["name"]
        out_dir = self.output_base / "docs_rs" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        docs = []
        try:
            url = f"https://docs.rs/{name}/latest/{name}/"
            resp = self.session.get(url, timeout=30)
            if not resp or resp.status_code != 200:
                return []
            soup = BeautifulSoup(resp.text, "lxml")
            links = list(
                {
                    f"https://docs.rs{link['href']}"
                    if link["href"].startswith("/")
                    else f"https://docs.rs/{name}/latest/{link['href']}"
                    for link in soup.find_all("a", href=True)
                    if ".html" in link.get("href", "") and "#" not in link.get("href", "")
                }
            )[:30]
            for link in tqdm(links, desc=name, leave=False):
                try:
                    time.sleep(0.1)
                    r = self.session.get(link, timeout=30)
                    if not r:  # Content not modified
                        continue
                    s = BeautifulSoup(r.text, "lxml")
                    main = s.find("div", class_="docblock") or s.find("main")
                    if main and len(main.text) > 50:
                        title = s.find("h1").text.strip() if s.find("h1") else name
                        md = self.h2t.handle(str(main))
                        h = hashlib.sha256(md.encode()).hexdigest()[:16]
                        docs.append(
                            {
                                "crate_name": name,
                                "url": link,
                                "title": title,
                                "content": md,
                                "metadata": crate,
                            }
                        )
                        with open(out_dir / f"{h}.json", "w") as f:
                            json.dump(docs[-1], f)
                except Exception:
                    pass
        except Exception:
            pass
        return docs

    def run_all(self):
        print("Collecting docs.rs...")
        for crate in tqdm(self.crates):
            self.collect_crate(crate)
        stats = self.session.get_stats()
        print(f"Cache stats: {stats['fetched']} fetched, {stats['skipped']} skipped (out of {stats['total_checked']} total)")


if __name__ == "__main__":
    DocsRsCollector().run_all()
