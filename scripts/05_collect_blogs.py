#!/usr/bin/env python3
import hashlib
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

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


def extract_feed_version(entry: Any) -> tuple[str, str]:
    """Extract release or date-based version from feed entry."""
    title = str(entry.get("title", ""))
    # Check for Rust release announcements, e.g. "Announcing Rust 1.75.0"
    m = re.search(r"Rust\s+(\d+\.\d+(?:\.\d+)?)", title, re.IGNORECASE)
    if m:
        return m.group(1), "semver"

    # Try pubdate / year-month
    if entry.get("published_parsed"):
        tm = entry["published_parsed"]
        return f"{tm.tm_year}-{tm.tm_mon:02d}", "date"

    pub = entry.get("published") or entry.get("updated")
    if pub:
        m_year = re.search(r"(\d{4})", str(pub))
        if m_year:
            return m_year.group(1), "date"
    return "recent", "date"


def extract_site_version(link: str) -> tuple[str, str]:
    """Extract edition or version from URL."""
    m = re.search(r"(?:edition-guide/|rust-|edition/)?(2018|2021|2024)", link)
    if m and any(e in link for e in ["2018", "2021", "2024"]):
        return m.group(1), "edition"
    m_ver = re.search(r"/v?(\d+\.\d+(?:\.\d+)?)/", link)
    if m_ver:
        return m_ver.group(1), "semver"
    return "current", "edition"


class BlogsCollector:
    def __init__(self) -> None:
        self.output_base: Path = RAW_DIR
        session = requests.Session()
        session.headers.update({"User-Agent": "RustLoRA/1.0"})
        self.session: CachedSession = CachedSession(session)
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_images = True

        # Load blogs and best_practices sources
        with open(CONFIG_DIR / "sources.yaml") as f:
            config = yaml.safe_load(f)

        # Feeds are RSS feeds, sites are web pages
        self.feeds: list[tuple[str, str, str]] = [
            (source["url"], source["output_dir"].replace("raw/", ""), source["name"])
            for source in config.get("blogs", [])
        ]

        self.sites: list[tuple[str, str, str]] = [
            (source["url"], source["output_dir"].replace("raw/", ""), source["name"])
            for source in config.get("best_practices", [])
        ]

    def count_files(self, out_dir: Path) -> int:
        if not out_dir.exists():
            return 0
        return sum(1 for _ in out_dir.rglob("*.json") if _.is_file())

    def collect_feed(self, url: str, rel: str, name: str | None = None) -> None:
        source_name = name or rel.split("/")[-1]
        feed = feedparser.parse(url)

        for entry in tqdm(feed.entries, desc=source_name, leave=False):
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
                    version, version_type = extract_feed_version(entry)
                    pub_date = str(entry.get("published") or entry.get("updated") or "")
                    h = hashlib.sha256(
                        f"blogs:{source_name}:{version}:{entry.link}:{content}".encode()
                    ).hexdigest()[:16]
                    target_dir = self.output_base / rel / version
                    target_dir.mkdir(parents=True, exist_ok=True)
                    with open(target_dir / f"{h}.json", "w") as f:
                        json.dump(
                            {
                                "source": "blogs",
                                "source_type": "blogs",
                                "library": source_name,
                                "version": version,
                                "version_type": version_type,
                                "is_latest": True,
                                "url": entry.link,
                                "title": entry.title,
                                "content": content,
                                "metadata": {
                                    "feed_url": url,
                                    "source": source_name,
                                    "published": pub_date,
                                    "version": version,
                                    "version_type": version_type,
                                },
                                "collected_at": datetime.now().isoformat(),
                            },
                            f,
                            indent=2,
                        )
            except requests.Timeout as e:
                logger.warning(f"Timeout fetching {entry.link}: {e}")
            except Exception as e:
                logger.warning(f"Error processing {entry.link}: {type(e).__name__}: {e}")

    def _extract_site_links(self, soup: BeautifulSoup, base_url: str) -> set[str]:
        """Extract all HTML links from site index."""
        links: set[str] = set()
        for link in soup.find_all("a", href=True):
            href = str(link.get("href", ""))
            if href.endswith(".html") and not href.startswith("http"):
                full_url = base_url + href if href.startswith("/") else href
                links.add(full_url)
        return links

    def _process_site_link(
        self,
        link: str,
        out_dir: Path,
        base_url: str,
        rel: str,
        source_name: str,
    ) -> None:
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
        title_tag = s.find("h1")
        if title_tag:
            title_text = title_tag.get_text(strip=True)
        else:
            title_text = rel.split("/")[-1]
        md = self.h2t.handle(str(main))
        if len(md) <= 200:
            return
        version, version_type = extract_site_version(link)
        target_dir = out_dir / version
        target_dir.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256(
            f"best_practices:{source_name}:{version}:{link}:{md}".encode()
        ).hexdigest()[:16]
        with open(target_dir / f"{h}.json", "w") as f:
            json.dump(
                {
                    "source": "best_practices",
                    "source_type": "best_practices",
                    "library": source_name,
                    "version": version,
                    "version_type": version_type,
                    "is_latest": True,
                    "url": link,
                    "title": title_text,
                    "content": md,
                    "metadata": {
                        "site_url": base_url,
                        "source": source_name,
                        "version": version,
                        "version_type": version_type,
                    },
                    "collected_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

    def _fetch_site_index(self, url: str) -> BeautifulSoup | None:
        """Fetch and parse site index page."""
        resp = self.session.get(url, timeout=30)
        if not resp:
            return None
        return BeautifulSoup(resp.text, "lxml")

    def collect_site(self, url: str, rel: str, name: str | None = None) -> None:
        out_dir = self.output_base / rel
        source_name = name or rel.split("/")[-1]
        visited: set[str] = set()
        logger.info(f"Collecting {source_name} from {url}")
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
                    self._process_site_link(link, out_dir, url, rel, source_name)
                except requests.Timeout as e:
                    logger.warning(f"Timeout fetching {link}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing {link}: {type(e).__name__}: {e}")
        except requests.Timeout as e:
            logger.warning(f"Timeout fetching site index {url}: {e}")
        except Exception as e:
            logger.warning(f"Error collecting from {url}: {type(e).__name__}: {e}")

    def run_all(self) -> None:
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
