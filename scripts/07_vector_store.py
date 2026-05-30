#!/usr/bin/env python3
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
VECTORS_DIR = BASE_DIR / "data" / "vectors"


class VectorStore:
    def __init__(self):
        VECTORS_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(VECTORS_DIR), settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection("rust_docs")

    def index(self, filepath):
        print(f"Indexing {filepath.name} into ChromaDB...")
        before_count = self.collection.count()
        batch_ids, batch_docs, batch_metas = [], [], []
        input_count = 0
        with open(filepath) as f:
            for line in f:
                input_count += 1
                c = json.loads(line)
                batch_ids.append(c["hash"])
                batch_docs.append(c["content"][:50000])
                batch_metas.append({"source": c["source_type"], "tokens": c["tokens"]})
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

    def update_source(self, source_type, chunks_file):
        print(f"Updating {source_type}...")
        ids = [
            id
            for id, meta in zip(
                self.collection.get()["ids"], self.collection.get()["metadatas"], strict=False
            )
            if meta["source"] == source_type
        ]
        if ids:
            self.collection.delete(ids=ids)
        self.index(chunks_file)


if __name__ == "__main__":
    VectorStore().index(PROCESSED_DIR / "all_chunks.jsonl")
