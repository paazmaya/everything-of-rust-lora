#!/usr/bin/env python3
import base64
import os
import time
from pathlib import Path

import requests
import yaml
from cache_utils import CachedSession
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CONFIG_DIR = BASE_DIR / "config"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")


class GitHubCollector:
    def __init__(self):
        self.output_base = RAW_DIR
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "RustLoRA/1.0",
                "Authorization": f"token {GITHUB_TOKEN}" if GITHUB_TOKEN else "",
            }
        )
        self.session = CachedSession(session)
        with open(CONFIG_DIR / "libraries.yaml") as f:
            self.config = yaml.safe_load(f)
        self.repos = [
            item["repo"]
            for cat in self.config["libraries"].values()
            for item in cat
            if "repo" in item
        ]

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
        out_dir = self.output_base / "github" / repo.replace("/", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        for fname in ["README.md", "CHANGELOG.md"]:
            content = self.get_file(repo, fname)
            if content and len(content) > 100:
                with open(out_dir / fname, "w") as f:
                    f.write(content)
        time.sleep(0.5)

    def run_all(self):
        print("Collecting GitHub...")
        if not GITHUB_TOKEN:
            print("Warning: No GITHUB_TOKEN set. Rate limits apply.")
        for repo in tqdm(self.repos):
            self.collect_repo(repo)
        stats = self.session.get_stats()
        print(
            f"Cache stats: {stats['fetched']} fetched, {stats['skipped']} skipped (out of {stats['total_checked']} total)"
        )


if __name__ == "__main__":
    GitHubCollector().run_all()
