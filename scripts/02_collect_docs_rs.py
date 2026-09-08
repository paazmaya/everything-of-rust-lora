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
from urllib.parse import urldefrag, urljoin

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
logger = logging.getLogger("rust_lora.docs_rs")

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
        # Load all crates from flat array
        self.crates = self.config["libraries"]

    def count_docs(self, out_dir: Path) -> int:
        if not out_dir.exists():
            return 0
        return sum(1 for _ in out_dir.rglob("*.json") if _.is_file())

    def _resolve_crate_version(self, name: str) -> str:
        """Resolve the latest version of a crate from crates.io API or docs.rs redirect."""
        # Try crates.io API first
        try:
            resp = self.session.session.get(f"https://crates.io/api/v1/crates/{name}", timeout=10)
            if resp.status_code == 200:
                data: dict[str, Any] = resp.json()
                max_version = data.get("crate", {}).get("max_version")
                if max_version:
                    return str(max_version)
        except Exception as e:
            logger.debug(f"Failed to query crates.io for {name}: {e}")

        # Fallback to docs.rs redirect
        try:
            head_resp = self.session.session.head(
                f"https://docs.rs/{name}/latest/{name}/",
                allow_redirects=True,
                timeout=10,
            )
            match = re.search(rf"docs\.rs/{re.escape(name)}/([^/]+)/", head_resp.url)
            if match and match.group(1) != "latest":
                return match.group(1)
        except Exception as e:
            logger.debug(f"Failed to query docs.rs redirect for {name}: {e}")

        return "latest"

    def _extract_doc_links(
        self, soup: BeautifulSoup, name: str, version: str, base_url: str
    ) -> set[str]:
        """Extract all documentation links from crate index for specific version."""
        links: set[str] = set()
        for link in soup.find_all("a", href=True):
            href = str(link.get("href", ""))
            if ".html" in href and "#" not in href:
                full_url = urljoin(base_url, href)
                full_url = urldefrag(full_url)[0]
                # Ensure the link stays within this crate and version namespace
                if f"/{name}/{version}/" in full_url or (
                    version == "latest" and f"/{name}/latest/" in full_url
                ):
                    links.add(full_url)
                elif f"/{name}/" in full_url and ".html" in full_url:
                    links.add(full_url)
        return links

    def _process_doc_link(
        self,
        link: str,
        crate: dict[str, Any],
        name: str,
        version: str,
        is_latest: bool,
        out_dir: Path,
    ) -> dict[str, Any] | None:
        """Process a single documentation link and save if valid."""
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(0.05)
        r = self.session.get(link, timeout=30)
        if not r:
            return None
        s = BeautifulSoup(r.text, "lxml")
        main = s.find("div", class_="docblock") or s.find("main")
        if not main or len(main.text) <= 200:
            return None
        h1_tag = s.find("h1")
        title = h1_tag.get_text(strip=True) if h1_tag is not None else name
        md = self.h2t.handle(str(main))
        h = hashlib.sha256(f"{name}:{version}:{link}:{md}".encode()).hexdigest()[:16]
        doc: dict[str, Any] = {
            "source": "docs_rs",
            "source_type": "api_docs",
            "library": name,
            "version": version,
            "version_type": "semver",
            "is_latest": is_latest,
            "url": link,
            "title": title,
            "content": md,
            "metadata": {
                **crate,
                "crate_name": name,
                "version": version,
                "is_latest": is_latest,
            },
            "collected_at": datetime.now().isoformat(),
        }
        with open(out_dir / f"{h}.json", "w") as f:
            json.dump(doc, f, indent=2)
        return doc

    def _fetch_crate_index(self, name: str, version: str) -> tuple[BeautifulSoup | None, str]:
        """Fetch and parse crate index page for a specific version."""
        url = f"https://docs.rs/{name}/{version}/{name}/"
        resp = self.session.get(url, timeout=30)
        if not resp or resp.status_code != 200:
            logger.warning(
                f"Failed to fetch crate index for {name} ({version}): status {resp.status_code if resp else 'no response'}"
            )
            return None, url
        final_url = getattr(resp, "url", url)
        return BeautifulSoup(resp.text, "lxml"), final_url

    def collect_crate_version(
        self, crate: dict[str, Any], version: str, is_latest: bool = False
    ) -> list[dict[str, Any]]:
        name = str(crate["name"])
        out_dir = self.output_base / "docs_rs" / name / version
        out_dir.mkdir(parents=True, exist_ok=True)
        docs: list[dict[str, Any]] = []
        try:
            soup, base_url = self._fetch_crate_index(name, version)
            if not soup:
                return []
            links = self._extract_doc_links(soup, name, version, base_url)
            for link in tqdm(links, desc=f"{name}@{version}", leave=False):
                try:
                    doc = self._process_doc_link(link, crate, name, version, is_latest, out_dir)
                    if doc:
                        docs.append(doc)
                except requests.Timeout as e:
                    logger.warning(f"Timeout fetching {link}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing {link}: {type(e).__name__}: {e}")
        except requests.Timeout as e:
            logger.warning(f"Timeout fetching crate index for {name}@{version}: {e}")
        except Exception as e:
            logger.warning(
                f"Error collecting docs for crate {name}@{version}: {type(e).__name__}: {e}"
            )
        return docs

    def collect_crate(self, crate: dict[str, Any]) -> list[dict[str, Any]]:
        name = str(crate["name"])
        configured_versions = crate.get("versions")
        if configured_versions and isinstance(configured_versions, list):
            docs: list[dict[str, Any]] = []
            for idx, ver in enumerate(configured_versions):
                is_latest = idx == len(configured_versions) - 1
                docs.extend(self.collect_crate_version(crate, str(ver), is_latest=is_latest))
            return docs
        elif crate.get("version"):
            return self.collect_crate_version(crate, str(crate["version"]), is_latest=True)
        else:
            resolved_version = self._resolve_crate_version(name)
            return self.collect_crate_version(crate, resolved_version, is_latest=True)

    def run_all(self) -> None:
        print("Collecting docs.rs...")
        before_count = self.count_docs(self.output_base / "docs_rs")
        for crate in tqdm(self.crates):
            self.collect_crate(crate)
        after_count = self.count_docs(self.output_base / "docs_rs")
        stats = self.session.get_stats()
        print(f"Collected {after_count - before_count} new docs, {after_count} total.")
        print(
            f"Cache stats: {stats['fetched']} fetched, {stats['skipped']} skipped, {stats.get('errors', 0)} errors"
        )
        if self.session.get_errors():
            print(f"\nCollection warnings: {len(self.session.get_errors())} URLs had issues.")
            logger.info("See logs for details on failed URLs.")


if __name__ == "__main__":
    DocsRsCollector().run_all()
