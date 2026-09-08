#!/usr/bin/env python3
import hashlib
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import ParseResult, urldefrag, urljoin, urlparse

import html2text
import requests
import yaml
from bs4 import BeautifulSoup
from cache_utils import CachedSession

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("rust_lora.esp_rs")

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CONFIG_DIR = BASE_DIR / "config"


class ESPCollector:
    def __init__(self):
        self.output_base = RAW_DIR
        session = requests.Session()
        session.headers.update({"User-Agent": "RustLoRA/1.0"})
        self.session = CachedSession(session)
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_images = True
        self.visited: set[str] = set()

        # Load embedded documentation sources
        with open(CONFIG_DIR / "sources.yaml") as f:
            config = yaml.safe_load(f)

        embedded = config.get("embedded", [])
        self.sites = [
            (source["url"], source["output_dir"].replace("raw/", ""), source["name"])
            for source in embedded
        ]

    def count_files(self, out_dir: Path) -> int:
        if not out_dir.exists():
            return 0
        return sum(1 for _ in out_dir.rglob("*.json") if _.is_file())

    def _is_valid_link_esp(self, href: str) -> bool:
        """Check if a link should be followed."""
        return not href.startswith(("mailto:", "javascript:"))

    def _is_valid_page_url_esp(
        self, parsed_url: ParseResult, base_domain: str, base_path: str
    ) -> bool:
        """Check if parsed URL is within boundaries."""
        if parsed_url.netloc != base_domain:
            return False
        if not parsed_url.path.startswith(base_path):
            return False
        return parsed_url.path.endswith((".html", "/"))

    def _extract_version(self, url: str) -> str:
        """Extract version string from URL if present."""
        m = re.search(r"/v?(\d+\.\d+(?:\.\d+)?)/", url)
        if m:
            return m.group(1)
        return "current"

    def _extract_main_content(self, soup: BeautifulSoup) -> Any:
        """Extract main content from page."""
        return (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", id="content")
            or soup.find("body")
        )

    def _save_page_content(self, soup: BeautifulSoup, normalized_url: str, out_rel: str) -> None:
        """Extract and save page content if valid."""
        main = self._extract_main_content(soup)
        if not main or len(main.text) <= 200:
            return
        title = soup.find("h1")
        title_text = title.get_text(strip=True) if title is not None else "esp-rs documentation"
        version = self._extract_version(normalized_url)
        md = self.h2t.handle(str(main))
        h = hashlib.sha256(
            f"esp_rs:{out_rel}:{version}:{normalized_url}:{md}".encode()
        ).hexdigest()[:16]
        out_dir = self.output_base / out_rel / version
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{h}.json", "w") as f:
            json.dump(
                {
                    "source": "esp_rs",
                    "source_type": "official_docs",
                    "library": "esp-rs",
                    "version": version,
                    "version_type": "doc_version",
                    "is_latest": (version == "current"),
                    "url": normalized_url,
                    "title": title_text,
                    "content": md,
                    "metadata": {
                        "section": out_rel,
                        "library": "esp-rs",
                        "version": version,
                    },
                    "collected_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

    def _queue_linked_pages_esp(
        self,
        soup: BeautifulSoup,
        normalized_url: str,
        base_domain: str,
        base_path: str,
        out_rel: str,
    ) -> None:
        """Extract and queue linked pages for collection."""
        for link in soup.find_all("a", href=True):
            href = str(link.get("href", ""))
            if not self._is_valid_link_esp(href):
                continue
            next_url = urljoin(normalized_url, href)
            parsed = urlparse(next_url)
            if not self._is_valid_page_url_esp(parsed, base_domain, base_path):
                continue
            self.collect_page_recursive(next_url, base_domain, base_path, out_rel)

    def collect_page_recursive(
        self, url: str, base_domain: str, base_path: str, out_rel: str
    ) -> None:
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
            self._save_page_content(s, normalized_url, out_rel)
            self._queue_linked_pages_esp(s, normalized_url, base_domain, base_path, out_rel)
        except requests.Timeout as e:
            logger.warning(f"Timeout collecting {normalized_url}: {e}")
        except Exception as e:
            logger.warning(f"Error collecting {normalized_url}: {type(e).__name__}: {e}")

    def collect_site(self, base_url: str, out_rel: str) -> None:
        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc
        base_path = parsed_base.path if parsed_base.path.endswith("/") else parsed_base.path + "/"
        self.collect_page_recursive(base_url, base_domain, base_path, out_rel)

    def run_all(self) -> None:
        print("Collecting ESP-RS and embedded documentation...")
        before_count = sum(self.count_files(self.output_base / rel) for _, rel, _ in self.sites)
        for url, rel, name in self.sites:
            logger.info(f"Collecting {name} from {url}")
            self.collect_site(url, rel)
        after_count = sum(self.count_files(self.output_base / rel) for _, rel, _ in self.sites)
        stats = self.session.get_stats()
        print(f"Collected {after_count - before_count} new documents, {after_count} total.")
        print(
            f"Cache stats: {stats['fetched']} fetched, {stats['skipped']} skipped, {stats.get('errors', 0)} errors"
        )
        if self.session.get_errors():
            print(f"\nCollection warnings: {len(self.session.get_errors())} URLs had issues.")
            logger.info("See logs for details on failed URLs.")


if __name__ == "__main__":
    ESPCollector().run_all()
