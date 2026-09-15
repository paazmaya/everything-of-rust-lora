#!/usr/bin/env python3
import hashlib
import io
import json
import logging
import re
import sys
import tarfile
import time
import zipfile
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


USER_AGENT = "RustLoRA/1.0"
HTML_EXT = ".html"


class DocsRsCollector:
    def __init__(self):
        self.output_base = RAW_DIR
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
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
        """Resolve the latest full stable version of a crate from crates.io API."""
        # Try crates.io API
        try:
            resp = self.session.session.get(
                f"https://crates.io/api/v1/crates/{name}",
                headers={"User-Agent": USER_AGENT},
                timeout=10,
            )
            if resp.status_code == 200:
                data: dict[str, Any] = resp.json()
                crate_info = data.get("crate", {})
                max_stable = crate_info.get("max_stable_version")
                default_ver = crate_info.get("default_version")
                max_ver = crate_info.get("max_version")
                # Avoid alpha/beta/rc if stable is available
                if max_ver and re.search(r"-(alpha|beta|rc|dev|pre)", str(max_ver), re.IGNORECASE):
                    chosen = max_stable or default_ver or max_ver
                else:
                    chosen = max_stable or max_ver or default_ver
                if chosen:
                    return str(chosen)
        except Exception as e:
            logger.debug(f"Failed to query crates.io for {name}: {e}")

        # Fallback to docs.rs redirect
        try:
            head_resp = self.session.session.head(
                f"https://docs.rs/{name}/latest/",
                allow_redirects=True,
                timeout=10,
            )
            match = re.search(rf"docs\.rs/{re.escape(name)}/([^/]+)/", head_resp.url)
            if match and match.group(1) != "latest":
                return match.group(1)
        except Exception as e:
            logger.debug(f"Failed to query docs.rs redirect for {name}: {e}")

        return "latest"

    def _clean_title(self, h1_tag: Any, fallback: str) -> str:
        """Extract clean text from h1 tag without copy buttons."""
        if h1_tag is None:
            return fallback
        for btn in h1_tag.find_all(class_="copy-path"):
            btn.decompose()
        return h1_tag.get_text(strip=True) or fallback

    def _is_useful_doc_path(self, path: str) -> bool:
        """Filter out source views, assets, trivial constants/types, and secondary target duplicates."""
        if not path.endswith(HTML_EXT):
            return False
        # Filter source code views
        if "/src/" in path or path.startswith("src/"):
            return False
        # Filter UI boilerplate
        if any(path.endswith(b) for b in ["help.html", "settings.html", "all.html"]):
            return False
        # Filter secondary target architecture trees (default target lives at the zip root)
        first_segment = path.split("/")[0]
        if re.match(
            r"^(x86_64|i686|aarch64|arm|riscv|wasm32|wasm64|mips|s390x|powerpc|sparc)-",
            first_segment,
        ):
            return False
        # Filter trivial single-item declaration subpages
        base = path.split("/")[-1]
        if any(base.startswith(p) for p in ["constant.", "type.", "primitive."]):
            return False
        return True

    def _extract_doc_links(
        self, soup: BeautifulSoup, name: str, version: str, base_url: str
    ) -> set[str]:
        """Extract all documentation links from crate index for specific version."""
        links: set[str] = set()
        for link in soup.find_all("a", href=True):
            href = str(link.get("href", ""))
            if HTML_EXT in href and "#" not in href:
                base_name = href.split("/")[-1]
                if any(base_name.startswith(p) for p in ["constant.", "type.", "primitive."]):
                    continue
                full_url = urljoin(base_url, href)
                full_url = urldefrag(full_url)[0]
                # Ensure the link stays within this crate and version namespace
                if (
                    f"/{name}/{version}/" in full_url
                    or (version == "latest" and f"/{name}/latest/" in full_url)
                    or (f"/{name}/" in full_url and HTML_EXT in full_url)
                ):
                    links.add(full_url)
        return links

    def _save_doc(
        self,
        name: str,
        version: str,
        is_latest: bool,
        url: str,
        title: str,
        content: str,
        crate: dict[str, Any],
        out_dir: Path,
    ) -> dict[str, Any]:
        """Save extracted markdown document to JSON file."""
        h = hashlib.sha256(f"{name}:{version}:{url}:{content}".encode()).hexdigest()[:16]
        doc: dict[str, Any] = {
            "source": "docs_rs",
            "source_type": "api_docs",
            "library": name,
            "version": version,
            "version_type": "semver",
            "is_latest": is_latest,
            "url": url,
            "title": title,
            "content": content,
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

    def _parse_and_save_zip_entry(
        self,
        z: zipfile.ZipFile,
        entry_name: str,
        name: str,
        resolved_version: str,
        is_latest: bool,
        crate: dict[str, Any],
        target_out_dir: Path,
    ) -> dict[str, Any] | None:
        """Parse a single HTML file from zip and save if valid."""
        raw = z.read(entry_name).decode("utf-8", errors="replace")
        s = BeautifulSoup(raw, "lxml")
        main = s.find("div", class_="docblock") or s.find("main")
        if not main or len(main.text) <= 200:
            return None
        title = self._clean_title(s.find("h1"), name)
        md = self.h2t.handle(str(main))
        if len(md) <= 200:
            return None
        entry_url = f"https://docs.rs/{name}/{resolved_version}/{entry_name}"
        return self._save_doc(
            name=name,
            version=resolved_version,
            is_latest=is_latest,
            url=entry_url,
            title=title,
            content=md,
            crate=crate,
            out_dir=target_out_dir,
        )

    def _extract_from_zip(
        self,
        crate: dict[str, Any],
        name: str,
        version: str,
        is_latest: bool,
    ) -> list[dict[str, Any]] | None:
        """Download and extract documentation directly from docs.rs ZIP archive."""
        url = f"https://docs.rs/crate/{name}/{version}/download"
        try:
            r = self.session.session.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=45,
            )
            if r.status_code != 200 or len(r.content) < 500:
                return None

            # Extract exact version from Content-Disposition if available
            disp = r.headers.get("content-disposition", "")
            m = re.search(
                r"rustdoc-.*?-(\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9.]+)?)\.zip",
                disp,
            )
            resolved_version = m.group(1) if m else version
            target_out_dir = self.output_base / "docs_rs" / name / resolved_version
            target_out_dir.mkdir(parents=True, exist_ok=True)

            docs: list[dict[str, Any]] = []
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                for entry_name in z.namelist():
                    if not self._is_useful_doc_path(entry_name):
                        continue
                    try:
                        doc = self._parse_and_save_zip_entry(
                            z,
                            entry_name,
                            name,
                            resolved_version,
                            is_latest,
                            crate,
                            target_out_dir,
                        )
                        if doc:
                            docs.append(doc)
                    except Exception as e:
                        logger.debug(f"Error parsing zip entry {entry_name}: {e}")

            if docs:
                logger.info(
                    f"Successfully extracted {len(docs)} docs from docs.rs ZIP for {name} ({resolved_version})"
                )
                return docs
        except Exception as e:
            logger.debug(f"Zip download failed for {name} ({version}): {e}")

        return None

    def _extract_from_crate_tarball(
        self,
        crate: dict[str, Any],
        name: str,
        version: str,
        is_latest: bool,
    ) -> list[dict[str, Any]]:
        """Fallback: download source .crate from crates.io and extract README/CHANGELOG/docs."""
        resolved_version = self._resolve_crate_version(name) if version == "latest" else version
        if resolved_version == "latest":
            return []
        url = f"https://static.crates.io/crates/{name}/{name}-{resolved_version}.crate"
        try:
            r = self.session.session.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=30,
            )
            if r.status_code != 200:
                return []

            target_out_dir = self.output_base / "docs_rs" / name / resolved_version
            target_out_dir.mkdir(parents=True, exist_ok=True)
            docs: list[dict[str, Any]] = []

            with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".md"):
                        f = tar.extractfile(member)
                        if f:
                            content = f.read().decode("utf-8", errors="replace")
                            if len(content) > 200:
                                fname = member.name.split("/")[-1]
                                title = f"{name}: {fname}"
                                file_url = (
                                    f"https://crates.io/crates/{name}/{resolved_version}#{fname}"
                                )
                                doc = self._save_doc(
                                    name=name,
                                    version=resolved_version,
                                    is_latest=is_latest,
                                    url=file_url,
                                    title=title,
                                    content=content,
                                    crate=crate,
                                    out_dir=target_out_dir,
                                )
                                docs.append(doc)
            if docs:
                logger.info(
                    f"Fallback extracted {len(docs)} markdown files from .crate package for {name} ({resolved_version})"
                )
            return docs
        except Exception as e:
            logger.debug(f"Tarball fallback failed for {name}: {e}")
            return []

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
        title = self._clean_title(s.find("h1"), name)
        md = self.h2t.handle(str(main))
        if len(md) <= 200:
            return None
        return self._save_doc(
            name=name,
            version=version,
            is_latest=is_latest,
            url=link,
            title=title,
            content=md,
            crate=crate,
            out_dir=out_dir,
        )

    def _fetch_crate_index(self, name: str, version: str) -> tuple[BeautifulSoup | None, str]:
        """Fetch and parse crate index page for a specific version with proper redirect handling."""
        mod_name = name.replace("-", "_")
        urls_to_try = [
            f"https://docs.rs/{name}/{version}/",
            f"https://docs.rs/{name}/{version}/{mod_name}/",
        ]
        for url in urls_to_try:
            try:
                resp = self.session.get(url, timeout=30)
                if resp and resp.status_code == 200:
                    final_url = getattr(resp, "url", url)
                    return BeautifulSoup(resp.text, "lxml"), final_url
            except Exception as e:
                logger.debug(f"Failed fetching {url}: {e}")

        logger.warning(f"Failed to fetch crate index for {name} ({version})")
        return None, urls_to_try[0]

    def _crawl_crate_index(
        self,
        crate: dict[str, Any],
        name: str,
        version: str,
        is_latest: bool,
        out_dir: Path,
    ) -> list[dict[str, Any]]:
        """Crawl individual HTML pages if zip download is not available."""
        docs: list[dict[str, Any]] = []
        try:
            soup, base_url = self._fetch_crate_index(name, version)
            if not soup:
                return docs
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

    def collect_crate_version(
        self, crate: dict[str, Any], version: str, is_latest: bool = False
    ) -> list[dict[str, Any]]:
        name = str(crate["name"])
        out_dir = self.output_base / "docs_rs" / name / version
        out_dir.mkdir(parents=True, exist_ok=True)

        # Strategy 1: Try docs.rs ZIP download (fastest, clean, official)
        zip_docs = self._extract_from_zip(crate, name, version, is_latest)
        if zip_docs is not None and len(zip_docs) > 0:
            return zip_docs

        # Strategy 2: If zip failed, crawl docs.rs HTML pages
        docs = self._crawl_crate_index(crate, name, version, is_latest, out_dir)
        if docs:
            return docs

        # Strategy 3: Fallback to .crate package from crates.io (for crates with failed docs.rs builds)
        return self._extract_from_crate_tarball(crate, name, version, is_latest)

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
