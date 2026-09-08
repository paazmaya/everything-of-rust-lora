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
logger = logging.getLogger("rust_lora.rust_book")

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CONFIG_DIR = BASE_DIR / "config"


def extract_doc_version(url: str, soup: BeautifulSoup | None = None) -> tuple[str, str]:
    """Extract version or edition from URL/HTML. Returns (version_str, version_type)."""
    m = re.search(r"(?:edition-guide/|rust-|edition/)?(2018|2021|2024)", url)
    if m and any(e in url for e in ["2018", "2021", "2024"]):
        return m.group(1), "edition"

    m_ver = re.search(r"/(\d+\.\d+(?:\.\d+)?)/", url)
    if m_ver:
        return m_ver.group(1), "semver"

    if soup:
        meta_ver = soup.find("meta", {"name": "rust-version"}) or soup.find(
            "meta", {"name": "doc-version"}
        )
        if meta_ver and meta_ver.get("content"):
            return str(meta_ver["content"]), "semver"

    return "current", "edition"


class RustDocCollector:
    def __init__(self) -> None:
        self.output_base: Path = RAW_DIR
        session = requests.Session()
        session.headers.update({"User-Agent": "RustLoRA/1.0"})
        self.session: CachedSession = CachedSession(session)
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_images = True
        self.h2t.body_width = 0
        self.visited: set[str] = set()

        # Load official documentation and blog sources
        with open(CONFIG_DIR / "sources.yaml") as f:
            config = yaml.safe_load(f)
        self.sources: list[dict[str, Any]] = config.get("sources", []) + config.get("blogs", [])

    def _save(
        self,
        source: str,
        url: str,
        title: str,
        content: str,
        metadata: dict[str, Any],
        out_dir: Path,
        version: str,
        version_type: str,
    ) -> None:
        target_dir = out_dir / version
        target_dir.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256(f"{source}:{version}:{url}:{content}".encode()).hexdigest()[:16]
        safe_title = re.sub(r"[^\w\s-]", "", title).strip()[:50].replace(" ", "_")
        with open(target_dir / f"{safe_title}_{h}.json", "w") as f:
            json.dump(
                {
                    "source": source,
                    "source_type": "official_docs",
                    "library": source,
                    "version": version,
                    "version_type": version_type,
                    "is_latest": (version in ["current", "2024"]),
                    "url": url,
                    "title": title,
                    "content": content,
                    "metadata": {
                        **metadata,
                        "source": source,
                        "version": version,
                        "version_type": version_type,
                    },
                    "collected_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

    def count_output_files(self, out_dir: Path) -> int:
        if not out_dir.exists():
            return 0
        return sum(1 for _ in out_dir.rglob("*.json") if _.is_file())

    def _is_valid_link(self, href: str) -> bool:
        """Check if a link should be followed."""
        return not href.startswith(("mailto:", "javascript:"))

    def _is_valid_page_url(self, parsed_url: ParseResult, base_domain: str, base_path: str) -> bool:
        """Check if parsed URL is within boundaries."""
        if parsed_url.netloc != base_domain:
            return False
        if not parsed_url.path.startswith(base_path):
            return False
        return parsed_url.path.endswith((".html", "/"))

    def _process_page_content(
        self, soup: BeautifulSoup, normalized_url: str, source_name: str, out_dir: Path
    ) -> None:
        """Extract and save page content if valid."""
        main = soup.find("main") or soup.find("div", class_="content")
        if not main or len(main.text) <= 200:
            return
        h1_tag = soup.find("h1")
        title = h1_tag.get_text(strip=True) if h1_tag is not None else "Untitled"
        version, version_type = extract_doc_version(normalized_url, soup)
        md = self.h2t.handle(str(main))
        self._save(
            source_name,
            normalized_url,
            title,
            md,
            {"book": source_name, "version": version},
            out_dir,
            version,
            version_type,
        )

    def _queue_linked_pages(
        self,
        soup: BeautifulSoup,
        normalized_url: str,
        source_name: str,
        out_dir: Path,
        base_domain: str,
        base_path: str,
    ) -> None:
        """Extract and queue linked pages for collection."""
        for link in soup.find_all("a", href=True):
            href = str(link["href"])
            if not self._is_valid_link(href):
                continue
            next_url = urljoin(normalized_url, href)
            parsed = urlparse(next_url)
            if not self._is_valid_page_url(parsed, base_domain, base_path):
                continue
            self.collect_site_recursive(next_url, source_name, out_dir, base_domain, base_path)

    def collect_site_recursive(
        self,
        url: str,
        source_name: str,
        out_dir: Path,
        base_domain: str,
        base_path: str,
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
            self._process_page_content(s, normalized_url, source_name, out_dir)
            self._queue_linked_pages(
                s, normalized_url, source_name, out_dir, base_domain, base_path
            )
        except requests.Timeout as e:
            logger.warning(f"Timeout collecting {normalized_url}: {e}")
        except Exception as e:
            logger.warning(f"Error collecting {normalized_url}: {type(e).__name__}: {e}")

    def collect_site(self, base_url: str, source_name: str, out_dir: str) -> None:
        output_dir = self.output_base / out_dir
        parsed_base = urlparse(base_url)
        base_domain = parsed_base.netloc
        base_path = parsed_base.path if parsed_base.path.endswith("/") else parsed_base.path + "/"
        self.collect_site_recursive(base_url, source_name, output_dir, base_domain, base_path)

    def run_all(self) -> None:
        print("Collecting official Rust documentation...")

        # Get all source directories before collection
        source_dirs = [s["output_dir"].replace("raw/", "") for s in self.sources]
        total_before = sum(self.count_output_files(RAW_DIR / part) for part in source_dirs)

        # Collect from each source
        for source in self.sources:
            url = source["url"]
            output_dir = source["output_dir"].replace("raw/", "")
            source_name = source["name"]

            logger.info(f"Collecting {source_name} from {url}")
            self.visited.clear()
            self.collect_site(url, source_name, output_dir)

        total_after = sum(self.count_output_files(RAW_DIR / part) for part in source_dirs)
        stats = self.session.get_stats()
        print(f"\nCollected {total_after - total_before} new raw documents, {total_after} total.")
        print(
            f"Cache stats: {stats['fetched']} fetched, {stats['skipped']} skipped, {stats.get('errors', 0)} errors"
        )
        if self.session.get_errors():
            print(f"\nCollection warnings: {len(self.session.get_errors())} URLs had issues.")
            logger.info("See logs for details on failed URLs.")


if __name__ == "__main__":
    RustDocCollector().run_all()
