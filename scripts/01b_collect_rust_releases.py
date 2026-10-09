#!/usr/bin/env python3
"""
Collect comprehensive rustc release history and changelogs:
1. Technical release notes (RELEASES.md) from rust-lang/rust for all versions >= 1.0.0.
2. Narrative release blog posts with code examples from rust-lang/blog.rust-lang.org.
3. Pre-computed consecutive minor version diffs (e.g. 1.80.0 -> 1.81.0) for migration Q&A.

Outputs to data/raw/rustc_releases/{version}/
"""

import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from packaging.version import InvalidVersion, Version
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("rust_lora.rustc_releases")

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = RAW_DIR / "rustc_releases"

USER_AGENT = "RustLoRA/1.0"
RELEASES_MD_URL = "https://raw.githubusercontent.com/rust-lang/rust/master/RELEASES.md"
BLOG_TREE_URL = (
    "https://api.github.com/repos/rust-lang/blog.rust-lang.org/git/trees/main?recursive=1"
)
BLOG_RAW_BASE = "https://raw.githubusercontent.com/rust-lang/blog.rust-lang.org/main/"


def get_http_headers() -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        # Test if token is valid or expired
        try:
            test_resp = requests.get(
                "https://api.github.com/user",
                headers={"Authorization": f"token {token}", "User-Agent": USER_AGENT},
                timeout=5,
            )
            if test_resp.status_code == 200:
                headers["Authorization"] = f"token {token}"
            else:
                logger.debug(
                    "Provided GITHUB_TOKEN unauthorized; proceeding with unauthenticated requests."
                )
        except Exception:
            pass
    return headers


def normalize_version(v_str: str) -> str:
    """Normalize version strings like '1.80' to '1.80.0'."""
    parts = v_str.split(".")
    if len(parts) == 2:
        return f"{v_str}.0"
    return v_str


def is_valid_stable_release(v_str: str) -> bool:
    """Check if version is a valid stable release >= 1.0.0 without pre-release suffixes."""
    try:
        norm = normalize_version(v_str)
        v = Version(norm)
        if v.is_prerelease or v.is_devrelease:
            return False
        return v >= Version("1.0.0")
    except InvalidVersion:
        return False


def _parse_version_sections(body: str) -> dict[str, str]:
    """Extract named sections from release markdown body."""
    sections: dict[str, str] = {}
    section_pattern = re.compile(
        r"(?:<a id=\"[^\"]+\"></a>\s*)?(?:###?\s+([A-Za-z0-9 /&_-]+)|^([A-Za-z0-9 /&_-]+)\n[-~]{3,})",
        re.MULTILINE,
    )
    sec_matches = list(section_pattern.finditer(body))
    if not sec_matches:
        sections["Details"] = body
        return sections

    first_sec_start = sec_matches[0].start()
    preamble = body[:first_sec_start].strip()
    if preamble:
        sections["Overview"] = preamble

    for s_idx, s_match in enumerate(sec_matches):
        title = (s_match.group(1) or s_match.group(2) or "Section").strip()
        s_start = s_match.end()
        s_end = sec_matches[s_idx + 1].start() if s_idx + 1 < len(sec_matches) else len(body)
        s_text = body[s_start:s_end].strip()
        if s_text:
            sections[title] = s_text

    return sections


def _extract_single_release(
    match: re.Match[str], next_start: int, content: str
) -> dict[str, Any] | None:
    """Extract a release dict from a regex match if it is a valid stable release."""
    version = match.group(1)
    release_date = match.group(2).strip()
    if not is_valid_stable_release(version):
        return None

    body = content[match.end() : next_start].strip()
    return {
        "version": version,
        "release_date": release_date,
        "content": body,
        "sections": _parse_version_sections(body),
    }


def parse_releases_md(content: str) -> list[dict[str, Any]]:
    """Parse RELEASES.md into individual version records."""
    pattern = re.compile(
        r"^Version\s+(\d+\.\d+\.\d+)\s*\(([^)]+)\)\s*\n=+",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(content))
    if not matches:
        logger.error("No version sections found in RELEASES.md")
        return []

    releases: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
        rel = _extract_single_release(match, next_start, content)
        if rel:
            releases.append(rel)

    releases.sort(key=lambda r: Version(r["version"]))
    for i in range(len(releases)):
        releases[i]["previous_version"] = releases[i - 1]["version"] if i > 0 else None

    return releases


def _parse_blog_frontmatter(raw_text: str, norm_v: str) -> tuple[str, str, str]:
    """Extract title, date, and clean body from blog post markdown."""
    title = f"Announcing Rust {norm_v}"
    date_str = ""
    content_body = raw_text

    if raw_text.startswith("+++"):
        parts = raw_text.split("+++", 2)
        if len(parts) >= 3:
            frontmatter = parts[1]
            content_body = parts[2].strip()
            m_title = re.search(r'title\s*=\s*"([^"]+)"', frontmatter)
            if m_title:
                title = m_title.group(1)
            m_date = re.search(r'path\s*=\s*"(\d{4}/\d{2}/\d{2})', frontmatter)
            if m_date:
                date_str = m_date.group(1).replace("/", "-")
    elif raw_text.startswith("---"):
        parts = raw_text.split("---", 2)
        if len(parts) >= 3:
            frontmatter = parts[1]
            content_body = parts[2].strip()
            m_title = re.search(r'title:\s*["\']?([^"\']+)["\']?', frontmatter)
            if m_title:
                title = m_title.group(1)

    return title, date_str, content_body


def _discover_blog_post_paths(headers: dict[str, str]) -> dict[str, str]:
    """Fetch tree and map version strings to repo paths."""
    try:
        r = requests.get(BLOG_TREE_URL, headers=headers, timeout=30)
        if r.status_code != 200:
            logger.warning(f"Could not retrieve blog tree: HTTP {r.status_code}")
            return {}
        data = r.json()
        tree = data.get("tree", [])
    except Exception as e:
        logger.warning(f"Failed to query blog tree: {e}")
        return {}

    post_paths: dict[str, str] = {}
    for item in tree:
        path = str(item.get("path", ""))
        m = re.search(
            r"content/.*?Rust[_-](\d+\.\d+(?:\.\d+)?)(?:\.md|/index\.md)$", path, re.IGNORECASE
        )
        if m:
            norm_v = normalize_version(m.group(1))
            if is_valid_stable_release(norm_v):
                post_paths[norm_v] = path
    return post_paths


def fetch_blog_announcements(headers: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Fetch blog release announcements from rust-lang/blog.rust-lang.org."""
    logger.info("Discovering release announcement posts from blog.rust-lang.org...")
    post_paths = _discover_blog_post_paths(headers)
    logger.info(f"Found {len(post_paths)} release blog announcement posts.")

    announcements: dict[str, dict[str, Any]] = {}
    for norm_v, path in tqdm(post_paths.items(), desc="Downloading blog posts", leave=False):
        raw_url = f"{BLOG_RAW_BASE}{path}"
        try:
            resp = requests.get(raw_url, headers={"User-Agent": USER_AGENT}, timeout=30)
            if resp.status_code != 200:
                continue

            title, date_str, content_body = _parse_blog_frontmatter(resp.text, norm_v)
            announcements[norm_v] = {
                "version": norm_v,
                "title": title,
                "date": date_str,
                "content": content_body,
                "url": f"https://blog.rust-lang.org/{path.replace('content/', '').replace('.md', '.html')}",
            }
            time.sleep(0.05)
        except Exception as e:
            logger.warning(f"Error fetching blog post for {norm_v} ({path}): {e}")

    return announcements


def build_consecutive_diff_summary(
    prev_release: dict[str, Any], curr_release: dict[str, Any]
) -> dict[str, Any]:
    """Generate a structured delta between two consecutive minor releases."""
    prev_v = prev_release["version"]
    curr_v = curr_release["version"]

    curr_sections = curr_release.get("sections", {})
    language_changes = curr_sections.get("Language", "")
    compiler_changes = curr_sections.get("Compiler", "")
    lib_changes = curr_sections.get("Stabilized APIs", "") or curr_sections.get("Libraries", "")
    compat_notes = curr_sections.get("Compatibility Notes", "") or curr_sections.get(
        "Compatibility notes", ""
    )
    cargo_changes = curr_sections.get("Cargo", "")

    summary_lines = [
        f"# Rust {prev_v} to {curr_v} Upgrade and Migration Summary",
        "",
        f"This document outlines all technical changes, language advancements, compiler updates, and newly stabilized APIs between Rust {prev_v} and Rust {curr_v}.",
        "",
    ]

    if language_changes:
        summary_lines.append(f"## Language Changes in {curr_v}\n{language_changes}\n")
    if lib_changes:
        summary_lines.append(f"## Stabilized APIs & Library Additions in {curr_v}\n{lib_changes}\n")
    if compiler_changes:
        summary_lines.append(f"## Compiler & Tooling Updates in {curr_v}\n{compiler_changes}\n")
    if cargo_changes:
        summary_lines.append(f"## Cargo Changes in {curr_v}\n{cargo_changes}\n")
    if compat_notes:
        summary_lines.append(
            f"## Compatibility Notes & Potential Breaking Changes\n{compat_notes}\n"
        )

    diff_content = "\n".join(summary_lines)

    return {
        "title": f"Rust {prev_v} to {curr_v} Upgrade Summary",
        "from_version": prev_v,
        "to_version": curr_v,
        "content": diff_content,
        "language_changes": language_changes,
        "lib_changes": lib_changes,
        "compat_notes": compat_notes,
    }


def main() -> None:
    headers = get_http_headers()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching RELEASES.md from rust-lang/rust...")
    resp = requests.get(RELEASES_MD_URL, headers={"User-Agent": "RustLoRA/1.0"}, timeout=60)
    if resp.status_code != 200:
        logger.error(f"Failed to fetch RELEASES.md: HTTP {resp.status_code}")
        sys.exit(1)

    releases = parse_releases_md(resp.text)
    logger.info(f"Parsed {len(releases)} stable releases >= 1.0.0.")

    latest_version = releases[-1]["version"] if releases else "1.0.0"

    # Fetch blog release announcements
    blog_posts = fetch_blog_announcements(headers)

    # Save documents per version
    saved_count = 0
    # Map to help build consecutive minor releases
    minor_releases: dict[str, dict[str, Any]] = {}

    for rel in tqdm(releases, desc="Saving release records"):
        v = rel["version"]
        target_dir = OUTPUT_DIR / v
        target_dir.mkdir(parents=True, exist_ok=True)

        is_latest = v == latest_version

        # 1. Save RELEASES.md technical release record
        rel_json_path = target_dir / f"releases_md_{v}.json"
        with open(rel_json_path, "w") as f:
            json.dump(
                {
                    "source": "rustc_releases",
                    "source_type": "official_releases",
                    "library": "rustc",
                    "version": v,
                    "version_type": "semver",
                    "previous_version": rel.get("previous_version"),
                    "release_date": rel["release_date"],
                    "is_latest": is_latest,
                    "is_changelog": True,
                    "url": f"https://github.com/rust-lang/rust/blob/master/RELEASES.md#version-{v.replace('.', '')}",
                    "title": f"Rust {v} Technical Release Notes",
                    "content": rel["content"],
                    "sections": rel["sections"],
                    "metadata": {
                        "library": "rustc",
                        "version": v,
                        "previous_version": rel.get("previous_version"),
                        "release_date": rel["release_date"],
                        "is_changelog": True,
                        "is_latest": is_latest,
                        "source": "rustc_releases",
                        "version_type": "semver",
                    },
                    "collected_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )
        saved_count += 1

        # Track .0 releases for minor diffs (e.g. 1.80.0, 1.81.0)
        if v.endswith(".0"):
            minor_releases[v] = rel

        # 2. Save blog announcement if available
        if v in blog_posts:
            bp = blog_posts[v]
            bp_json_path = target_dir / f"blog_announcement_{v}.json"
            with open(bp_json_path, "w") as f:
                json.dump(
                    {
                        "source": "rustc_releases",
                        "source_type": "official_releases",
                        "library": "rustc",
                        "version": v,
                        "version_type": "semver",
                        "previous_version": rel.get("previous_version"),
                        "is_latest": is_latest,
                        "is_changelog": True,
                        "url": bp["url"],
                        "title": bp["title"],
                        "content": bp["content"],
                        "metadata": {
                            "library": "rustc",
                            "version": v,
                            "title": bp["title"],
                            "date": bp["date"],
                            "source": "rustc_releases",
                            "is_changelog": True,
                            "is_latest": is_latest,
                            "post_type": "release_announcement",
                            "version_type": "semver",
                        },
                        "collected_at": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
            saved_count += 1

    # 3. Generate and save consecutive minor version diffs (1.N-1.0 -> 1.N.0)
    sorted_minors = sorted(minor_releases.keys(), key=lambda x: Version(x))
    diff_count = 0
    for idx in range(1, len(sorted_minors)):
        prev_v = sorted_minors[idx - 1]
        curr_v = sorted_minors[idx]

        diff_data = build_consecutive_diff_summary(minor_releases[prev_v], minor_releases[curr_v])
        diff_path = OUTPUT_DIR / curr_v / f"diff_{prev_v}_to_{curr_v}.json"

        with open(diff_path, "w") as f:
            json.dump(
                {
                    "source": "rustc_releases",
                    "source_type": "official_releases",
                    "library": "rustc",
                    "version": curr_v,
                    "version_type": "semver",
                    "from_version": prev_v,
                    "to_version": curr_v,
                    "is_latest": (curr_v == latest_version),
                    "is_changelog": True,
                    "url": f"https://doc.rust-lang.org/nightly/releases.html#version-{curr_v.replace('.', '')}",
                    "title": diff_data["title"],
                    "content": diff_data["content"],
                    "metadata": {
                        "library": "rustc",
                        "version": curr_v,
                        "from_version": prev_v,
                        "to_version": curr_v,
                        "is_migration_guide": True,
                        "is_changelog": True,
                        "source": "rustc_releases",
                        "version_type": "semver",
                    },
                    "collected_at": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )
        diff_count += 1

    logger.info(
        f"Collection complete! Saved {saved_count} release & blog documents and {diff_count} consecutive version diffs under {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
