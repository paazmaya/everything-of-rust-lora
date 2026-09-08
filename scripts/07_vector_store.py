#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import Metadata, QueryResult, Where
from chromadb.config import Settings

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
VECTORS_DIR = BASE_DIR / "data" / "vectors"


class VectorStore:
    def __init__(self) -> None:
        VECTORS_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(VECTORS_DIR), settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection("rust_docs")

    def index(self, filepath: Path) -> None:
        print(f"Indexing {filepath.name} into ChromaDB...")
        before_count = self.collection.count()
        batch_ids: list[str] = []
        batch_docs: list[str] = []
        batch_metas: list[Metadata] = []
        input_count = 0
        with open(filepath) as f:
            for line in f:
                input_count += 1
                c: dict[str, Any] = json.loads(line)
                lib = str(c.get("library", "unknown"))
                ver = str(c.get("version", "current"))
                compound_id = f"{lib}_{ver}_{c['hash']}"

                batch_ids.append(compound_id)
                batch_docs.append(str(c["content"])[:50000])
                batch_metas.append(
                    {
                        "library": lib,
                        "version": ver,
                        "version_type": str(c.get("version_type", "static")),
                        "is_latest": bool(c.get("is_latest", True)),
                        "source": str(c.get("source_type", "")),
                        "url": str(c.get("url", "")),
                        "title": str(c.get("title", "")),
                        "tokens": int(c.get("tokens", 0)),
                        "is_changelog": bool(
                            c.get("metadata", {}).get("is_changelog_section", False)
                        ),
                    }
                )
                if len(batch_ids) == 100:
                    self.collection.upsert(
                        ids=batch_ids, documents=batch_docs, metadatas=batch_metas
                    )
                    batch_ids, batch_docs, batch_metas = [], [], []
        if batch_ids:
            self.collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
        after_count = self.collection.count()
        net_change = after_count - before_count
        print(
            f"Indexed {input_count} chunks into vector storage. Vector store now contains {after_count} documents (was {before_count}, net change {net_change})."
        )

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        library: str | None = None,
        version: str | None = None,
        is_latest: bool | None = None,
    ) -> QueryResult:
        """
        Query vector database with optional version and library filtering for RAG.
        """
        filters: list[Where] = []
        if library:
            filters.append({"library": library})
        if version:
            filters.append({"version": version})
        if is_latest is not None:
            filters.append({"is_latest": is_latest})

        where_clause: Where | None = None
        if len(filters) == 1:
            where_clause = filters[0]
        elif len(filters) > 1:
            where_clause = {"$and": filters}  # type: ignore[dict-item]

        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_clause,
        )

    def query_version_migration(
        self,
        library: str,
        from_version: str,
        to_version: str,
        topic: str | None = None,
        n_results: int = 3,
    ) -> dict[str, Any]:
        """
        Specialized RAG query for comparing versions and finding migration/breaking change guidance.
        """
        query_str = f"{library} migration breaking changes upgrade {topic or ''}".strip()

        # Query changelogs and docs for the target version
        target_docs = self.collection.query(
            query_texts=[query_str],
            n_results=n_results,
            where={"$and": [{"library": library}, {"version": to_version}]},  # type: ignore[dict-item]
        )

        # Query previous version context
        from_docs = self.collection.query(
            query_texts=[query_str],
            n_results=n_results,
            where={"$and": [{"library": library}, {"version": from_version}]},  # type: ignore[dict-item]
        )

        return {
            "library": library,
            "from_version": from_version,
            "to_version": to_version,
            "from_version_chunks": from_docs,
            "to_version_chunks": target_docs,
        }

    def list_tracked_versions(self, library: str | None = None) -> dict[str, list[str]]:
        """List distinct libraries and versions stored in the vector database."""
        result = self.collection.get(include=["metadatas"])
        all_metas = result.get("metadatas") or []
        libs: dict[str, set[str]] = {}
        for m in all_metas:
            lib = str(m.get("library", "unknown"))
            ver = str(m.get("version", "current"))
            if library and lib != library:
                continue
            libs.setdefault(lib, set()).add(ver)
        return {k: sorted(v) for k, v in libs.items()}

    def update_source(self, source_type: str, chunks_file: Path) -> None:
        print(f"Updating {source_type}...")
        get_result = self.collection.get()
        ids_list = get_result.get("ids") or []
        metas_list = get_result.get("metadatas") or []
        ids = [
            id
            for id, meta in zip(ids_list, metas_list, strict=False)
            if meta.get("source") == source_type
        ]
        if ids:
            self.collection.delete(ids=ids)
        self.index(chunks_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vector store indexer and RAG query CLI")
    parser.add_argument(
        "--index", action="store_true", help="Index all processed chunks into ChromaDB"
    )
    parser.add_argument("--query", type=str, help="Run a semantic query")
    parser.add_argument("--library", type=str, help="Filter query by library name")
    parser.add_argument("--version", type=str, help="Filter query by library version")
    parser.add_argument("--migrate-from", type=str, help="Source version for migration comparison")
    parser.add_argument("--migrate-to", type=str, help="Target version for migration comparison")
    parser.add_argument(
        "--list-versions", action="store_true", help="List all tracked libraries and versions"
    )

    args = parser.parse_args()
    store = VectorStore()

    if args.list_versions:
        versions = store.list_tracked_versions(library=args.library)
        print("Tracked libraries and versions in Vector Store:")
        for lib, vers in sorted(versions.items()):
            print(f"  {lib}: {', '.join(vers)}")
    elif args.migrate_from and args.migrate_to and args.library:
        results = store.query_version_migration(
            library=args.library,
            from_version=args.migrate_from,
            to_version=args.migrate_to,
            topic=args.query,
        )
        print(f"\nMigration Context: {args.library} {args.migrate_from} -> {args.migrate_to}")
        print("\n--- Target Version Context (New APIs / Changelogs) ---")
        to_docs = results["to_version_chunks"].get("documents")
        if to_docs and len(to_docs) > 0:
            for doc in to_docs[0]:
                print(str(doc)[:300] + "...\n")
    elif args.query:
        results = store.query(
            args.query,
            library=args.library,
            version=args.version,
        )
        docs_list = results.get("documents")
        metas_list = results.get("metadatas")
        if docs_list and metas_list and len(docs_list) > 0 and len(metas_list) > 0:
            print(f"Found {len(docs_list[0])} matching chunks:")
            for idx, (doc, meta) in enumerate(zip(docs_list[0], metas_list[0], strict=False)):
                print(
                    f"\n[{idx + 1}] {meta.get('library')} @ {meta.get('version')} ({meta.get('title')}) - {meta.get('url')}"
                )
                print(str(doc)[:250] + "...")
    else:
        store.index(PROCESSED_DIR / "all_chunks.jsonl")


if __name__ == "__main__":
    main()
