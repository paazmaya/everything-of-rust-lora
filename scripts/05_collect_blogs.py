#!/usr/bin/env python3
import hashlib
import json
import logging
import re
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse

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
    m = re.search(r"(?:edition-guide/|rust-|edition/)?(2015|2018|2021|2024)", link)
    if m and any(e in link for e in ["2015", "2018", "2021", "2024"]):
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

        # Load blogs, best_practices, security, and web_frameworks sources
        with open(CONFIG_DIR / "sources.yaml") as f:
            config = yaml.safe_load(f)

        # Feeds are RSS feeds, sites are web pages
        self.feeds: list[tuple[str, str, str]] = [
            (source["url"], source["output_dir"].replace("raw/", ""), source["name"])
            for source in config.get("blogs", [])
        ]

        self.sites: list[tuple[str, str, str]] = [
            (source["url"], source["output_dir"].replace("raw/", ""), source["name"])
            for section in ["best_practices", "security", "web_frameworks"]
            for source in config.get(section, [])
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
                article = (
                    s.find("article") or s.find("main") or s.find("div", class_="content") or s.body
                )
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

    def collect_github_repo(self, url: str, rel: str, name: str | None = None) -> None:
        source_name = name or rel.split("/")[-1]
        out_dir = self.output_base / rel
        logger.info(f"Collecting GitHub repo {source_name} from {url}")
        parsed = urlparse(url)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 2:
            return
        repo = f"{parts[0]}/{parts[1]}"
        for branch in ["main", "master"]:
            raw_url = f"https://raw.githubusercontent.com/{repo}/{branch}/README.md"
            try:
                r = self.session.get(raw_url, timeout=30)
                if not r:
                    continue
                content = r.text
                if len(content) > 200:
                    target_dir = out_dir / "current"
                    target_dir.mkdir(parents=True, exist_ok=True)
                    h = hashlib.sha256(
                        f"best_practices:{source_name}:current:{raw_url}:{content}".encode()
                    ).hexdigest()[:16]
                    with open(target_dir / f"{h}.json", "w") as f:
                        json.dump(
                            {
                                "source": "best_practices",
                                "source_type": "best_practices",
                                "library": source_name,
                                "version": "current",
                                "version_type": "git_branch",
                                "is_latest": True,
                                "url": url,
                                "title": source_name.replace("_", " ").title(),
                                "content": content,
                                "metadata": {
                                    "site_url": url,
                                    "source": source_name,
                                    "repo": repo,
                                    "branch": branch,
                                    "version": "current",
                                },
                                "collected_at": datetime.now().isoformat(),
                            },
                            f,
                            indent=2,
                        )
                    break
            except Exception as e:
                logger.warning(f"Error fetching repo {repo}: {e}")

    def _is_valid_site_url(self, url: str, base_domain: str, base_path: str) -> bool:
        parsed = urlparse(url)
        if parsed.netloc != base_domain:
            return False
        if not parsed.path.startswith(base_path):
            return False
        lower = parsed.path.lower()
        if any(
            lower.endswith(ext)
            for ext in [
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".svg",
                ".webp",
                ".ico",
                ".css",
                ".js",
                ".pdf",
                ".zip",
                ".tar.gz",
                ".xml",
            ]
        ):
            return False
        return True

    def _extract_site_content(self, soup: BeautifulSoup) -> Any:
        return (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", class_="content")
            or soup.find("div", class_="page")
            or soup.find("div", class_="theme-doc-markdown")
            or soup.find("div", id="content")
            or soup.body
        )

    def _process_site_link(
        self,
        soup: BeautifulSoup,
        link: str,
        out_dir: Path,
        base_url: str,
        rel: str,
        source_name: str,
    ) -> None:
        main = self._extract_site_content(soup)
        if not main:
            return
        title_tag = soup.find("h1")
        if title_tag:
            title_text = title_tag.get_text(strip=True)
        elif soup.title and soup.title.string:
            title_text = soup.title.string.strip()
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

    def _compute_base_path(self, parsed_path: str) -> str:
        parts = [p for p in parsed_path.split("/") if p]
        if parts and parts[-1].endswith((".html", ".htm")):
            parts = parts[:-1]
        return "/" + "/".join(parts) + "/" if parts else "/"

    def _crawl_page(
        self,
        current_url: str,
        out_dir: Path,
        base_url: str,
        rel: str,
        source_name: str,
        base_domain: str,
        base_path: str,
        visited: set[str],
        queue: deque[str],
    ) -> None:
        r = self.session.get(current_url, timeout=30)
        if not r:
            return
        soup = BeautifulSoup(r.text, "lxml")
        self._process_site_link(soup, current_url, out_dir, base_url, rel, source_name)
        for a in soup.find_all("a", href=True):
            href = str(a.get("href", "")).strip()
            if not href or href.startswith(("mailto:", "javascript:", "#")):
                continue
            next_url = urldefrag(urljoin(current_url, href))[0]
            if next_url not in visited and self._is_valid_site_url(
                next_url, base_domain, base_path
            ):
                queue.append(next_url)

    def collect_site(self, url: str, rel: str, name: str | None = None) -> None:
        out_dir = self.output_base / rel
        source_name = name or rel.split("/")[-1]

        parsed = urlparse(url)
        if parsed.netloc == "github.com":
            self.collect_github_repo(url, rel, name)
            return

        base_url = urldefrag(url)[0]
        base_domain = parsed.netloc
        base_path = self._compute_base_path(parsed.path)

        visited: set[str] = set()
        queue: deque[str] = deque([base_url])
        max_pages = 500

        logger.info(f"Collecting {source_name} from {url}")
        while queue and len(visited) < max_pages:
            current_url = queue.popleft()
            if current_url in visited:
                continue
            visited.add(current_url)
            sys.stdout.write(".")
            sys.stdout.flush()

            try:
                self._crawl_page(
                    current_url,
                    out_dir,
                    base_url,
                    rel,
                    source_name,
                    base_domain,
                    base_path,
                    visited,
                    queue,
                )
            except requests.Timeout as e:
                logger.warning(f"Timeout fetching {current_url}: {e}")
            except Exception as e:
                logger.warning(f"Error processing {current_url}: {type(e).__name__}: {e}")

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
