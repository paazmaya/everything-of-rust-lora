#!/usr/bin/env python3
import base64
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml
from cache_utils import CachedSession
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("rust_lora.github")

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CONFIG_DIR = BASE_DIR / "config"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")


def split_changelog(content: str) -> list[tuple[str, str]]:
    """
    Split a changelog into (version_str, section_text) pairs.
    Matches headings like:
    ## [0.7.0] - 2023-11-27
    ## v1.2.3
    # 0.5.0
    """
    header_pattern = re.compile(
        r"^(#{1,3})\s+\[?v?(\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9\.]+)?)\]?.*$",
        re.MULTILINE,
    )
    matches = list(header_pattern.finditer(content))
    if not matches:
        return []

    sections: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        ver = match.group(2)
        start_idx = match.start()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sec_content = content[start_idx:end_idx].strip()
        if len(sec_content) > 40:
            sections.append((ver, sec_content))
    return sections


class GitHubCollector:
    def __init__(self):
        self.output_base = RAW_DIR
        self.github_token = os.environ.get("GITHUB_TOKEN", "")
        if not self.github_token:
            logger.warning(
                "GITHUB_TOKEN environment variable not set. Using unauthenticated GitHub API (slower rate limits). "
                "Set it with: export GITHUB_TOKEN='your_token'"
            )
        else:
            logger.info("Using authenticated GitHub API with GITHUB_TOKEN")

        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "RustLoRA/1.0",
                "Authorization": f"token {self.github_token}" if self.github_token else "",
            }
        )
        self.session = CachedSession(session)
        with open(CONFIG_DIR / "libraries.yaml") as f:
            self.config = yaml.safe_load(f)
        # Map repo to crate name
        self.repo_to_name = {
            item["repo"]: item["name"]
            for item in self.config["libraries"]
            if "repo" in item and "name" in item
        }
        self.repos = [item["repo"] for item in self.config["libraries"] if "repo" in item]

    def count_files(self, out_dir: Path) -> int:
        if not out_dir.exists():
            return 0
        return sum(1 for _ in out_dir.rglob("*.json") if _.is_file())

    def _resolve_repo_version(self, repo: str) -> tuple[str, str]:
        """Fetch latest release tag or commit SHA for a GitHub repo."""
        # Try latest release
        try:
            r = self.session.get(f"https://api.github.com/repos/{repo}/releases/latest", timeout=15)
            if r and r.status_code == 200:
                data = r.json()
                tag = data.get("tag_name")
                if tag:
                    clean_ver = tag.lstrip("v")
                    return str(tag), str(clean_ver)
        except Exception as e:
            logger.debug(f"Failed to fetch latest release for {repo}: {e}")

        # Try tags
        try:
            r = self.session.get(f"https://api.github.com/repos/{repo}/tags?per_page=1", timeout=15)
            if r and r.status_code == 200:
                data = r.json()
                if data and isinstance(data, list) and len(data) > 0:
                    tag = data[0].get("name")
                    if tag:
                        clean_ver = tag.lstrip("v")
                        return str(tag), str(clean_ver)
        except Exception as e:
            logger.debug(f"Failed to fetch tags for {repo}: {e}")

        return "main", "main"

    def get_file(self, repo: str, path: str, ref: str | None = None) -> str | None:
        try:
            url = f"https://api.github.com/repos/{repo}/contents/{path}"
            if ref:
                url += f"?ref={ref}"
            r = self.session.get(url, timeout=30)
            if r and r.status_code == 200:
                data = r.json()
                if data.get("encoding") == "base64":
                    return base64.b64decode(data["content"]).decode("utf-8")
        except Exception:
            pass
        return None

    def collect_repo(self, repo: str) -> None:
        library_name = self.repo_to_name.get(repo, repo.split("/")[-1])
        tag, version = self._resolve_repo_version(repo)
        repo_slug = repo.replace("/", "_")

        out_dir = self.output_base / "github" / repo_slug / version
        out_dir.mkdir(parents=True, exist_ok=True)

        files_to_collect = [
            ("README.md", "README"),
            ("CHANGELOG.md", "CHANGELOG"),
            ("CONTRIBUTING.md", "CONTRIBUTING"),
            ("ARCHITECTURE.md", "ARCHITECTURE"),
            ("DESIGN.md", "DESIGN"),
            ("SECURITY.md", "SECURITY"),
            ("PERFORMANCE.md", "PERFORMANCE"),
            ("docs/guide.md", "docs_guide"),
        ]

        for fname, label in files_to_collect:
            try:
                content = self.get_file(repo, fname, ref=tag if tag != "main" else None)
                if not content:
                    # Fallback without ref
                    content = self.get_file(repo, fname)

                if content and len(content) > 200:
                    # Special handling for CHANGELOG: also extract per-version sections
                    if label == "CHANGELOG":
                        sections = split_changelog(content)
                        for sec_ver, sec_content in sections:
                            sec_dir = self.output_base / "github" / repo_slug / sec_ver
                            sec_dir.mkdir(parents=True, exist_ok=True)
                            sec_h = hashlib.sha256(
                                f"{repo}:{sec_ver}:{sec_content}".encode()
                            ).hexdigest()[:16]
                            with open(sec_dir / f"CHANGELOG_v{sec_ver}_{sec_h}.json", "w") as f:
                                json.dump(
                                    {
                                        "source": "github",
                                        "source_type": "github_repo",
                                        "library": library_name,
                                        "repo": repo,
                                        "version": sec_ver,
                                        "version_type": "semver",
                                        "is_latest": (sec_ver == version),
                                        "url": f"https://github.com/{repo}/blob/{tag}/{fname}",
                                        "title": f"{repo}: CHANGELOG v{sec_ver}",
                                        "content": sec_content,
                                        "metadata": {
                                            "repo": repo,
                                            "file": fname,
                                            "library": library_name,
                                            "version": sec_ver,
                                            "is_changelog_section": True,
                                        },
                                        "collected_at": datetime.now().isoformat(),
                                    },
                                    f,
                                    indent=2,
                                )

                    # Save main document in the current version folder
                    h = hashlib.sha256(f"{repo}:{version}:{fname}:{content}".encode()).hexdigest()[
                        :16
                    ]
                    with open(out_dir / f"{label}_{h}.json", "w") as f:
                        json.dump(
                            {
                                "source": "github",
                                "source_type": "github_repo",
                                "library": library_name,
                                "repo": repo,
                                "version": version,
                                "version_type": "git_tag" if tag != "main" else "branch",
                                "is_latest": True,
                                "url": f"https://github.com/{repo}/blob/{tag}/{fname}",
                                "title": f"{repo}: {label}",
                                "content": content,
                                "metadata": {
                                    "repo": repo,
                                    "file": fname,
                                    "library": library_name,
                                    "version": version,
                                    "tag": tag,
                                    "is_latest": True,
                                },
                                "collected_at": datetime.now().isoformat(),
                            },
                            f,
                            indent=2,
                        )
                elif not content:
                    logger.debug(f"File not found or empty: {repo}/{fname}")
            except Exception as e:
                logger.warning(f"Error collecting {repo}/{fname}: {type(e).__name__}: {e}")

        # Adaptive rate limiting based on token
        delay = 0.2 if self.github_token else 0.4
        time.sleep(delay)

    def run_all(self) -> None:
        print("Collecting GitHub...")
        before_count = self.count_files(self.output_base / "github")
        for repo in tqdm(self.repos):
            self.collect_repo(repo)
        after_count = self.count_files(self.output_base / "github")
        stats = self.session.get_stats()
        print(f"Collected {after_count - before_count} new files, {after_count} total.")
        print(
            f"Cache stats: {stats['fetched']} fetched, {stats['skipped']} skipped, {stats.get('errors', 0)} errors"
        )
        if self.session.get_errors():
            print(f"\nCollection warnings: {len(self.session.get_errors())} URLs had issues.")
            logger.info("See logs for details on failed URLs.")


if __name__ == "__main__":
    GitHubCollector().run_all()
