#!/usr/bin/env python3
import base64
import hashlib
import json
import logging
import os
import re
import sys
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


def split_changelog(content: str) -> list[tuple[str, str]]:
    """
    Split a changelog into (version_str, section_text) pairs.
    Matches headings like:
    ## [0.7.0] - 2023-11-27
    ## v1.2.3
    # 0.5.0
    """
    header_pattern = re.compile(
        r"^(#{1,3})\s+\[?v?(\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9.]+)?)\b[^\n]*$",
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
        self.github_token = os.environ.get("GITHUB_TOKEN", "").strip()
        if not self.github_token:
            logger.error(
                "GITHUB_TOKEN environment variable is required. "
                "GitHub collection cannot proceed without authentication due to rate limits.\n"
                "Please set it with: export GITHUB_TOKEN='your_personal_access_token'\n"
                "Generate a token at: https://github.com/settings/tokens (requires 'repo' and 'read:user' scopes)"
            )
            sys.exit(1)

        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "RustLoRA/1.0",
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github+json",
            }
        )
        self.session = CachedSession(session)
        self._validate_token()

        with open(CONFIG_DIR / "libraries.yaml") as f:
            self.config = yaml.safe_load(f)
        # Map repo to crate name
        self.repo_to_name = {
            item["repo"]: item["name"]
            for item in self.config["libraries"]
            if "repo" in item and "name" in item
        }
        self.repos = [item["repo"] for item in self.config["libraries"] if "repo" in item]

    def _validate_token(self) -> None:
        """Verify that GITHUB_TOKEN is valid before proceeding with collection."""
        try:
            r = self.session.session.get("https://api.github.com/user", timeout=10)
            if r.status_code == 401:
                logger.error(
                    "GitHub API authentication failed (HTTP 401 Unauthorized).\n"
                    "The provided GITHUB_TOKEN is invalid, revoked, or expired.\n"
                    "Please update it with: export GITHUB_TOKEN='ghp_...'"
                )
                sys.exit(1)
            elif r.status_code == 200:
                user = r.json().get("login", "authenticated user")
                logger.info(f"Using authenticated GitHub API as '{user}'")
            else:
                logger.warning(
                    f"GitHub token verification returned unexpected status {r.status_code}"
                )
        except requests.RequestException:
            logger.exception("Failed to connect to GitHub API to verify token")
            sys.exit(1)

    def _check_auth_response(self, resp: requests.Response | None) -> None:
        """Abort immediately if the API returns 401 Unauthorized."""
        if resp is not None and resp.status_code == 401:
            logger.error(
                "GitHub API returned HTTP 401 Unauthorized. GITHUB_TOKEN is invalid or expired."
            )
            sys.exit(1)

    def count_files(self, out_dir: Path) -> int:
        if not out_dir.exists():
            return 0
        return sum(1 for _ in out_dir.rglob("*.json") if _.is_file())

    def _get_latest_release_tag(self, repo: str) -> str | None:
        try:
            r = self.session.get(f"https://api.github.com/repos/{repo}/releases/latest", timeout=15)
            self._check_auth_response(r)
            if r and r.status_code == 200:
                tag = r.json().get("tag_name")
                if tag:
                    return str(tag)
        except Exception as e:
            logger.debug(f"Failed to fetch latest release for {repo}: {e}")
        return None

    def _get_first_tag(self, repo: str) -> str | None:
        try:
            r = self.session.get(f"https://api.github.com/repos/{repo}/tags?per_page=1", timeout=15)
            self._check_auth_response(r)
            if r and r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    tag = data[0].get("name")
                    if tag:
                        return str(tag)
        except Exception as e:
            logger.debug(f"Failed to fetch tags for {repo}: {e}")
        return None

    def _resolve_repo_version(self, repo: str) -> tuple[str, str]:
        """Fetch latest release tag or commit SHA for a GitHub repo."""
        tag = self._get_latest_release_tag(repo) or self._get_first_tag(repo)
        if tag:
            clean_ver = tag.lstrip("v")
            return tag, clean_ver
        return "main", "main"

    def get_file(self, repo: str, path: str, ref: str | None = None) -> str | None:
        try:
            url = f"https://api.github.com/repos/{repo}/contents/{path}"
            if ref:
                url += f"?ref={ref}"
            r = self.session.get(url, timeout=30)
            self._check_auth_response(r)
            if r and r.status_code == 200:
                data = r.json()
                if data.get("encoding") == "base64":
                    return base64.b64decode(data["content"]).decode("utf-8")
        except Exception:
            pass
        return None

    def _save_changelog_sections(
        self,
        repo: str,
        repo_slug: str,
        library_name: str,
        content: str,
        tag: str,
        fname: str,
        latest_version: str,
    ) -> None:
        sections = split_changelog(content)
        for sec_ver, sec_content in sections:
            sec_dir = self.output_base / "github" / repo_slug / sec_ver
            sec_dir.mkdir(parents=True, exist_ok=True)
            sec_h = hashlib.sha256(f"{repo}:{sec_ver}:{sec_content}".encode()).hexdigest()[:16]
            with open(sec_dir / f"CHANGELOG_v{sec_ver}_{sec_h}.json", "w") as f:
                json.dump(
                    {
                        "source": "github",
                        "source_type": "github_repo",
                        "library": library_name,
                        "repo": repo,
                        "version": sec_ver,
                        "version_type": "semver",
                        "is_latest": (sec_ver == latest_version),
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

    def _save_repo_doc(
        self,
        repo: str,
        library_name: str,
        version: str,
        tag: str,
        fname: str,
        label: str,
        content: str,
        out_dir: Path,
    ) -> None:
        h = hashlib.sha256(f"{repo}:{version}:{fname}:{content}".encode()).hexdigest()[:16]
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
                    content = self.get_file(repo, fname)

                if content and len(content) > 200:
                    if label == "CHANGELOG":
                        self._save_changelog_sections(
                            repo, repo_slug, library_name, content, tag, fname, version
                        )
                    self._save_repo_doc(
                        repo, library_name, version, tag, fname, label, content, out_dir
                    )
                elif not content:
                    logger.debug(f"File not found or empty: {repo}/{fname}")
            except Exception as e:
                logger.warning(f"Error collecting {repo}/{fname}: {type(e).__name__}: {e}")

        time.sleep(0.2)

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
