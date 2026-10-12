#!/usr/bin/env python3
import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, cast

import chromadb
import torch
from chromadb.api.types import (
    EmbeddingFunction,
    Embeddings,
    Metadata,
    QueryResult,
    Where,
)
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from lib.config import DEFAULT_CHUNKS_FILE, DEFAULT_PROCESSED_DIR

logger = logging.getLogger("rust_lora.vector_store")

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / DEFAULT_PROCESSED_DIR
VECTORS_DIR = BASE_DIR / "data" / "vectors"

DEFAULT_EMBEDDING_MODEL = "google/embeddinggemma-2"


class EmbeddingGemmaEmbeddingFunction(EmbeddingFunction[Any]):
    """
    ChromaDB embedding function using Google DeepMind's EmbeddingGemma-2 via SentenceTransformers.
    Features hardware acceleration (Apple Silicon Metal / CUDA / CPU),
    task instruction prefixes ('task: search result | query: ' and 'title: {title} | text: {content}'),
    and memory-efficient text-only encoder loading (disabling unused vision/audio towers).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
        truncate_dim: int | None = None,
        **kwargs: Any,
    ) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        torch_dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.model_name = model_name
        self.device = device
        self.truncate_dim = truncate_dim

        print(f"Loading embedding model '{model_name}' on device '{device}'...")
        self.model = SentenceTransformer(
            model_name_or_path=model_name,
            device=device,
            model_kwargs={"torch_dtype": torch_dtype},
            config_kwargs={"vision_config": None, "audio_config": None},
        )

    @staticmethod
    def name() -> str:
        return "embeddinggemma_2"

    def _prepare_text(self, text: str) -> str:
        if text.startswith(("task:", "title:")):
            return text
        return f"title: none | text: {text}"

    def __call__(self, input: Any) -> Embeddings:
        processed = [self._prepare_text(str(doc)) for doc in input]
        encode_kwargs: dict[str, Any] = {
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        if self.truncate_dim is not None:
            encode_kwargs["truncate_dim"] = self.truncate_dim

        embeddings = self.model.encode(processed, **encode_kwargs)
        return cast(Embeddings, [emb.tolist() for emb in embeddings])


# Backwards compatibility alias
NomicGGUFEmbeddingFunction = EmbeddingGemmaEmbeddingFunction


class VectorStore:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
        truncate_dim: int | None = None,
        reset: bool = False,
        model_filename: str | None = None,
        model_path: str | None = None,
    ) -> None:
        VECTORS_DIR.mkdir(parents=True, exist_ok=True)
        self.embedding_fn = EmbeddingGemmaEmbeddingFunction(
            model_name=model_name,
            device=device,
            truncate_dim=truncate_dim,
        )
        self.client = chromadb.PersistentClient(
            path=str(VECTORS_DIR), settings=Settings(anonymized_telemetry=False)
        )
        if reset:
            try:
                self.client.delete_collection("rust_docs")
            except Exception:
                pass
            self.collection = self.client.create_collection(
                name="rust_docs",
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            return

        try:
            self.collection = self.client.get_collection(
                name="rust_docs", embedding_function=self.embedding_fn
            )
        except ValueError as e:
            if "Embedding function conflict" in str(e):
                old_coll = self.client.get_collection(name="rust_docs")
                if old_coll.count() == 0:
                    self.client.delete_collection(name="rust_docs")
                    self.collection = self.client.create_collection(
                        name="rust_docs",
                        embedding_function=self.embedding_fn,
                        metadata={"hnsw:space": "cosine"},
                    )
                else:
                    raise ValueError(
                        f"Existing 'rust_docs' collection was created with a different embedding function ({e}). "
                        "Run with '--reset' to re-index the collection with the new embedding model."
                    ) from e
            else:
                raise
        except Exception:
            self.collection = self.client.create_collection(
                name="rust_docs",
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

    def reset(self) -> None:
        """Clear and recreate the collection."""
        try:
            self.client.delete_collection("rust_docs")
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name="rust_docs",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        print("Vector store collection 'rust_docs' has been reset.")

    def _parse_chunk_record(self, raw_line: str) -> tuple[str, str, Metadata] | None:
        if not raw_line.strip():
            return None
        c: dict[str, Any] = json.loads(raw_line)
        lib = str(c.get("library", "unknown"))
        ver = str(c.get("version", "current"))
        chunk_hash = str(c.get("hash") or "")
        if not chunk_hash:
            content = str(c.get("content", ""))
            src = str(c.get("source_type", ""))
            chunk_hash = hashlib.sha256(f"{src}:{lib}:{ver}:{content}".encode()).hexdigest()[:20]

        compound_id = f"{lib}_{ver}_{chunk_hash}"
        raw_content = str(c.get("content", ""))[:20000]
        title = str(c.get("title", "")).strip()
        title_prefix = f"title: {title}" if title else "title: none"
        doc_text = f"{title_prefix} | text: {raw_content}"
        meta: Metadata = {
            "library": lib,
            "version": ver,
            "version_type": str(c.get("version_type", "static")),
            "is_latest": bool(c.get("is_latest", True)),
            "source": str(c.get("source_type", "")),
            "url": str(c.get("url", "")),
            "title": str(c.get("title", "")),
            "tokens": int(c.get("tokens", 0)),
            "is_changelog": bool(c.get("metadata", {}).get("is_changelog_section", False)),
        }
        return compound_id, doc_text, meta

    def _get_existing_ids(self) -> set[str]:
        if self.collection.count() == 0:
            return set()
        res = self.collection.get(include=[])
        ids = set(res.get("ids", []))
        print(f"Found {len(ids):,} existing records in ChromaDB. Skipping already added chunks...")
        return ids

    def index(self, filepath: Path, batch_size: int = 32) -> None:
        if not filepath.exists():
            print(f"Error: {filepath} does not exist.")
            return

        with open(filepath, "rb") as f:
            total_lines = sum(1 for _ in f)

        print(f"Indexing {filepath.name} ({total_lines:,} chunks) into ChromaDB...")
        existing_ids = self._get_existing_ids()

        batch_ids: list[str] = []
        batch_docs: list[str] = []
        batch_metas: list[Metadata] = []
        added_count = 0
        skipped_count = 0

        pbar = tqdm(total=total_lines, desc="ChromaDB Index", unit="chunk")

        with open(filepath, encoding="utf-8") as f:
            for line in f:
                parsed = self._parse_chunk_record(line)
                if not parsed:
                    pbar.update(1)
                    continue

                compound_id, doc_text, meta = parsed
                if compound_id in existing_ids:
                    skipped_count += 1
                    pbar.update(1)
                    continue

                batch_ids.append(compound_id)
                batch_docs.append(doc_text)
                batch_metas.append(meta)

                if len(batch_ids) >= batch_size:
                    self.collection.upsert(
                        ids=batch_ids, documents=batch_docs, metadatas=batch_metas
                    )
                    existing_ids.update(batch_ids)
                    added_count += len(batch_ids)
                    pbar.update(len(batch_ids))
                    batch_ids, batch_docs, batch_metas = [], [], []

        if batch_ids:
            self.collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
            existing_ids.update(batch_ids)
            added_count += len(batch_ids)
            pbar.update(len(batch_ids))

        pbar.close()

        after_count = self.collection.count()
        print(
            f"\nIndexing finished: {added_count:,} new chunks added, {skipped_count:,} already present (skipped). "
            f"Vector store now contains {after_count:,} documents."
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

        formatted_query = (
            query_text
            if query_text.startswith(("task:", "title:"))
            else f"task: search result | query: {query_text}"
        )

        return self.collection.query(
            query_texts=[formatted_query],
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
        query_str = f"task: search result | query: {library} migration breaking changes upgrade {topic or ''}".strip()

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
        print(f"Updating source '{source_type}'...")
        get_result = self.collection.get()
        ids_list = get_result.get("ids") or []
        metas_list = get_result.get("metadatas") or []
        ids = [
            id
            for id, meta in zip(ids_list, metas_list, strict=False)
            if meta.get("source") == source_type
        ]
        if ids:
            print(f"Removing {len(ids)} outdated chunks for source '{source_type}'...")
            self.collection.delete(ids=ids)
        self.index(chunks_file)


def _handle_list_versions(store: VectorStore, library: str | None) -> None:
    versions = store.list_tracked_versions(library=library)
    print("Tracked libraries and versions in Vector Store:")
    for lib, vers in sorted(versions.items()):
        print(f"  {lib}: {', '.join(vers)}")


def _handle_migration_query(store: VectorStore, args: argparse.Namespace) -> None:
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


def _handle_search_query(store: VectorStore, args: argparse.Namespace) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Vector store indexer and RAG query CLI")
    parser.add_argument(
        "--index", action="store_true", help="Index all processed chunks into ChromaDB"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Embedding model name or HF repo (default: {DEFAULT_EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: 'mps', 'cuda', or 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--truncate-dim",
        type=int,
        default=None,
        choices=[128, 256, 512, 768],
        help="Matryoshka representation truncation dimension (default: 768 full)",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default=None,
        help="Legacy model file name (ignored, preserved for backward compatibility)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Explicit local path or repo ID for embedding model",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for embedding (default: 32)"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Reset ChromaDB collection before indexing"
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
    model_target = args.model_path or args.model
    store = VectorStore(
        model_name=model_target,
        device=args.device,
        truncate_dim=args.truncate_dim,
        reset=args.reset,
    )

    if args.list_versions:
        _handle_list_versions(store, args.library)
    elif args.migrate_from and args.migrate_to and args.library:
        _handle_migration_query(store, args)
    elif args.query:
        _handle_search_query(store, args)
    else:
        store.index(BASE_DIR / DEFAULT_CHUNKS_FILE, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
