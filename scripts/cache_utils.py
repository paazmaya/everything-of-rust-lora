#!/usr/bin/env python3
"""
HTTP cache utility for tracking remote content versions.
Stores ETags and Last-Modified headers to avoid re-downloading unchanged content.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


class CacheMetadata:
    """Manages cache metadata for fetched URLs."""

    def __init__(self, cache_dir: Path = None):
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / "data" / ".cache"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "fetch_metadata.json"
        self.metadata = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        """Load existing cache metadata."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self):
        """Save cache metadata to disk."""
        with open(self.cache_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def get(self, url: str) -> dict[str, str] | None:
        """Get cached metadata for a URL."""
        return self.metadata.get(url)

    def should_fetch(self, url: str, response_headers: dict) -> bool:
        """
        Check if URL should be fetched based on cache.
        Returns True if:
        - URL not in cache
        - ETag differs
        - Last-Modified is newer
        """
        cached = self.get(url)
        if not cached:
            return True

        # Check ETag (exact match means no change)
        etag = response_headers.get("etag")
        if etag and cached.get("etag") == etag:
            return False

        # Check Last-Modified (if remote is newer, fetch it)
        last_modified = response_headers.get("last-modified")
        if last_modified and cached.get("last-modified") == last_modified:
            return False

        return True

    def update(self, url: str, response_headers: dict):
        """Update cache metadata after a successful fetch."""
        headers_to_cache = {}
        if "etag" in response_headers:
            headers_to_cache["etag"] = response_headers["etag"]
        if "last-modified" in response_headers:
            headers_to_cache["last-modified"] = response_headers["last-modified"]

        self.metadata[url] = {
            **headers_to_cache,
            "fetched_at": datetime.now().isoformat(),
        }
        self._save()


class CachedSession:
    """Session wrapper that handles conditional requests."""

    def __init__(self, session: requests.Session, cache_dir: Path = None):
        self.session = session
        self.cache = CacheMetadata(cache_dir)
        self.skipped_count = 0
        self.fetched_count = 0

    def get(self, url: str, **kwargs) -> requests.Response | None:
        """
        Fetch URL with conditional request support.
        Returns None if content hasn't changed (304 Not Modified).
        Returns response if content was fetched or 200 OK.
        """
        cached = self.cache.get(url)

        # Add conditional headers if we have them cached
        headers = kwargs.pop("headers", {})
        if cached:
            if cached.get("etag"):
                headers["If-None-Match"] = cached["etag"]
            if cached.get("last-modified"):
                headers["If-Modified-Since"] = cached["last-modified"]

        headers["User-Agent"] = headers.get("User-Agent", "RustLoRA/1.0")

        try:
            resp = self.session.get(url, headers=headers, **kwargs)

            if resp.status_code == 304:
                # Not Modified - content unchanged
                self.skipped_count += 1
                return None

            if resp.status_code == 200:
                # Update cache metadata
                self.cache.update(url, resp.headers)
                self.fetched_count += 1
                return resp

            # Other status codes
            return resp

        except Exception:
            return None

    def get_stats(self) -> dict[str, int]:
        """Return fetch statistics."""
        return {
            "fetched": self.fetched_count,
            "skipped": self.skipped_count,
            "total_checked": self.fetched_count + self.skipped_count,
        }
