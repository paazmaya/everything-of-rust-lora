#!/usr/bin/env python3
import base64
import hashlib
import json
import logging
import os
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
        # Load all repo references from flat array
        self.repos = [item["repo"] for item in self.config["libraries"] if "repo" in item]

    def count_files(self, out_dir):
        if not out_dir.exists():
            return 0
        return sum(1 for _ in out_dir.rglob("*") if _.is_file())

    def get_file(self, repo, path):
        try:
            r = self.session.get(f"https://api.github.com/repos/{repo}/contents/{path}", timeout=30)
            if r and r.status_code == 200:
                data = r.json()
                if data.get("encoding") == "base64":
                    return base64.b64decode(data["content"]).decode("utf-8")
        except Exception:
            pass
        return None

    def collect_repo(self, repo):
        import hashlib
        import json
        from datetime import datetime
        
        out_dir = self.output_base / "github" / repo.replace("/", "_")
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
                content = self.get_file(repo, fname)
                if content and len(content) > 200:
                    h = hashlib.sha256(content.encode()).hexdigest()[:16]
                    with open(out_dir / f"{label}_{h}.json", "w") as f:
                        json.dump(
                            {
                                "source": "github",
                                "source_type": "github_repo",
                                "url": f"https://github.com/{repo}/blob/main/{fname}",
                                "title": f"{repo}: {label}",
                                "content": content,
                                "metadata": {"repo": repo, "file": fname},
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

    def run_all(self):
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
            logger.info(f"See logs for details on failed URLs.")


if __name__ == "__main__":
    GitHubCollector().run_all()
