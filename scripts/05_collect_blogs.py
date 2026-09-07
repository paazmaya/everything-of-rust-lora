#!/usr/bin/env python3
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import feedparser
import html2text
import requests
import yaml
from bs4 import BeautifulSoup
from cache_utils import CachedSession
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("rust_lora.blogs")

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CONFIG_DIR = BASE_DIR / "config"


class BlogsCollector:
    def __init__(self):
        self.output_base = RAW_DIR
        session = requests.Session()
        session.headers.update({"User-Agent": "RustLoRA/1.0"})
        self.session = CachedSession(session)
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_images = True

        # Load blogs and best_practices sources
        with open(CONFIG_DIR / "sources.yaml") as f:
            config = yaml.safe_load(f)

        # Feeds are RSS feeds, sites are web pages
        self.feeds = [
            (source["url"], source["output_dir"].replace("raw/", ""), source["name"])
            for source in config.get("blogs", [])
        ]

        self.sites = [
            (source["url"], source["output_dir"].replace("raw/", ""), source["name"])
            for source in config.get("best_practices", [])
        ]

    def count_files(self, out_dir):
        if not out_dir.exists():
            return 0
        return sum(1 for _ in out_dir.rglob("*.json") if _.is_file())

    def collect_feed(self, url, rel, name=None):

        out_dir = self.output_base / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        feed = feedparser.parse(url)

        for entry in tqdm(feed.entries, desc=name or rel.split("/")[-1], leave=False):
            try:
                sys.stdout.write(".")
                sys.stdout.flush()
                time.sleep(0.3)
                r = self.session.get(entry.link, timeout=30)
                if not r:  # Content not modified
                    continue
                s = BeautifulSoup(r.text, "lxml")
                article = s.find("article") or s.find("main")
                content = self.h2t.handle(str(article)) if article else ""
                if len(content) > 300:
                    h = hashlib.sha256(content.encode()).hexdigest()[:16]
                    with open(out_dir / f"{h}.json", "w") as f:
                        json.dump(
                            {
                                "source": "blogs",
                                "source_type": "blogs",
                                "url": entry.link,
                                "title": entry.title,
                                "content": content,
                                "metadata": {"feed_url": url},
                                "collected_at": datetime.now().isoformat(),
                            },
                            f,
                            indent=2,
                        )
            except requests.Timeout as e:
                logger.warning(f"Timeout fetching {entry.link}: {e}")
            except Exception as e:
                logger.warning(f"Error processing {entry.link}: {type(e).__name__}: {e}")

    def _extract_site_links(self, soup, base_url):
        """Extract all HTML links from site index."""
        links = set()
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            if href.endswith(".html") and not href.startswith("http"):
                full_url = base_url + href if href.startswith("/") else href
                links.add(full_url)
        return links

    def _process_site_link(self, link, out_dir, base_url, rel):
        """Process a single site link and save if valid."""
        sys.stdout.write(".")
        sys.stdout.flush()
        r = self.session.get(link, timeout=30)
        if not r:
            return
        s = BeautifulSoup(r.text, "lxml")
        main = s.find("main")
        if not main:
            return
        title = s.find("h1")
        title_text = title.text.strip() if title else rel.split("/")[-1]
        md = self.h2t.handle(str(main))
        if len(md) <= 200:
            return
        h = hashlib.sha256(md.encode()).hexdigest()[:16]
        with open(out_dir / f"{h}.json", "w") as f:
            json.dump(
                {
                    "source": "best_practices",
                    "source_type": "best_practices",
                    "url": link,
                    "title": title_text,
                    "content": md,
                    "metadata": {"site_url": base_url},
                    "collected_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

    def _fetch_site_index(self, url):
        """Fetch and parse site index page."""
        resp = self.session.get(url, timeout=30)
        if not resp:
            return None
        return BeautifulSoup(resp.text, "lxml")

    def collect_site(self, url, rel, name=None):
        out_dir = self.output_base / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        visited = set()
        logger.info(f"Collecting {name or rel} from {url}")
        try:
            soup = self._fetch_site_index(url)
            if not soup:
                return
            links = self._extract_site_links(soup, url)
            for link in tqdm(links, leave=False):
                if link in visited:
                    continue
                visited.add(link)
                try:
                    self._process_site_link(link, out_dir, url, rel)
                except requests.Timeout as e:
                    logger.warning(f"Timeout fetching {link}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing {link}: {type(e).__name__}: {e}")
        except requests.Timeout as e:
            logger.warning(f"Timeout fetching site index {url}: {e}")
        except Exception as e:
            logger.warning(f"Error collecting from {url}: {type(e).__name__}: {e}")

    def run_all(self):
        print("Collecting Blogs & Best Practices...")
        before_count = sum(
            self.count_files(self.output_base / rel) for _, rel, _ in self.feeds + self.sites
        )
        for url, rel, name in self.feeds:
            self.collect_feed(url, rel, name)
        for url, rel, name in self.sites:
            self.collect_site(url, rel, name)
        after_count = sum(
            self.count_files(self.output_base / rel) for _, rel, _ in self.feeds + self.sites
        )
        stats = self.session.get_stats()
        print(f"Collected {after_count - before_count} new documents, {after_count} total.")
        print(
            f"Cache stats: {stats['fetched']} fetched, {stats['skipped']} skipped, {stats.get('errors', 0)} errors"
        )
        if self.session.get_errors():
            print(f"\nCollection warnings: {len(self.session.get_errors())} URLs had issues.")
            logger.info("See logs for details on failed URLs.")


if __name__ == "__main__":
    BlogsCollector().run_all()
