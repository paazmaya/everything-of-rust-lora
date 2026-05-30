#!/usr/bin/env python3
import hashlib
import json
import sys
import time
from pathlib import Path

import feedparser
import html2text
import requests
from bs4 import BeautifulSoup
from cache_utils import CachedSession
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"


class BlogsCollector:
    def __init__(self):
        self.output_base = RAW_DIR
        session = requests.Session()
        session.headers.update({"User-Agent": "RustLoRA/1.0"})
        self.session = CachedSession(session)
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_images = True
        self.feeds = [
            ("https://blog.rust-lang.org/feed.xml", "blogs/official"),
            ("https://www.reddit.com/r/rust/.rss", "blogs/reddit"),
        ]
        self.sites = [
            ("https://rust-lang.github.io/api-guidelines/", "best_practices/api_guidelines"),
            ("https://rust-unofficial.github.io/patterns/", "best_practices/patterns"),
        ]

    def count_files(self, out_dir):
        if not out_dir.exists():
            return 0
        return sum(1 for _ in out_dir.rglob("*.json") if _.is_file())

    def collect_feed(self, url, rel):
        out_dir = self.output_base / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        feed = feedparser.parse(url)
        for entry in tqdm(feed.entries, desc=rel.split("/")[-1], leave=False):
            try:
                sys.stdout.write(".")
                sys.stdout.flush()
                if "reddit" in url:
                    content = entry.get("summary", "")
                else:
                    time.sleep(0.3)
                    r = self.session.get(entry.link, timeout=30)
                    if not r:  # Content not modified
                        continue
                    s = BeautifulSoup(r.text, "lxml")
                    article = s.find("article") or s.find("main")
                    content = self.h2t.handle(str(article)) if article else ""
                if len(content) > 200:
                    h = hashlib.sha256(content.encode()).hexdigest()[:16]
                    with open(out_dir / f"{h}.json", "w") as f:
                        json.dump({"title": entry.title, "content": content}, f)
            except Exception:
                pass

    def collect_site(self, url, rel):
        out_dir = self.output_base / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        visited = set()
        try:
            resp = self.session.get(url, timeout=30)
            if resp:
                soup = BeautifulSoup(resp.text, "lxml")
                links = set()
                for link in soup.find_all("a", href=True):
                    href = link.get("href", "")
                    if href.endswith(".html") and not href.startswith("http"):
                        links.add(url + href if href.startswith("/") else href)
                for link in tqdm(links, leave=False):
                    if link in visited:
                        continue
                    visited.add(link)
                    try:
                        sys.stdout.write(".")
                        sys.stdout.flush()
                        r = self.session.get(link, timeout=30)
                        if not r:  # Content not modified
                            continue
                        s = BeautifulSoup(r.text, "lxml")
                        main = s.find("main")
                        if main:
                            md = self.h2t.handle(str(main))
                            if len(md) > 100:
                                h = hashlib.sha256(md.encode()).hexdigest()[:16]
                                with open(out_dir / f"{h}.json", "w") as f:
                                    json.dump({"content": md}, f)
                    except Exception:
                        pass
        except Exception:
            pass

    def run_all(self):
        print("Collecting Blogs & Best Practices...")
        before_count = sum(
            self.count_files(self.output_base / rel) for _, rel in self.feeds + self.sites
        )
        for url, rel in self.feeds:
            self.collect_feed(url, rel)
        for url, rel in self.sites:
            self.collect_site(url, rel)
        after_count = sum(
            self.count_files(self.output_base / rel) for _, rel in self.feeds + self.sites
        )
        stats = self.session.get_stats()
        print(f"Collected {after_count - before_count} new documents, {after_count} total.")
        print(
            f"Cache stats: {stats['fetched']} fetched, {stats['skipped']} skipped (out of {stats['total_checked']} total)"
        )


if __name__ == "__main__":
    BlogsCollector().run_all()
