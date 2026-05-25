#!/usr/bin/env python3
import hashlib
import json
import time
from pathlib import Path

import feedparser
import html2text
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"


class BlogsCollector:
    def __init__(self):
        self.output_base = RAW_DIR
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "RustLoRA/1.0"})
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

    def collect_feed(self, url, rel):
        out_dir = self.output_base / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        feed = feedparser.parse(url)
        for entry in tqdm(feed.entries[:50], desc=rel.split("/")[-1], leave=False):
            try:
                if "reddit" in url:
                    content = entry.get("summary", "")
                else:
                    time.sleep(0.5)
                    r = self.session.get(entry.link, timeout=30)
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
        try:
            resp = self.session.get(url, timeout=30)
            soup = BeautifulSoup(resp.text, "lxml")
            links = [
                url + link["href"]
                for link in soup.find_all("a", href=True)
                if link["href"].endswith(".html") and not link["href"].startswith("http")
            ]
            for link in tqdm(links, leave=False):
                try:
                    r = self.session.get(link, timeout=30)
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
        for url, rel in self.feeds:
            self.collect_feed(url, rel)
        for url, rel in self.sites:
            self.collect_site(url, rel)


if __name__ == "__main__":
    BlogsCollector().run_all()
