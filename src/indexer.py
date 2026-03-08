"""
Indexer: builds ChromaDB vector store + BM25 index from processed chunks.
Supports incremental rebuild and persistence.
"""
import json
import pickle
from pathlib import Path
from typing import Any

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from rank_bm25 import BM25Okapi
from openai import OpenAI

from src.chunker import build_chunk_document

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
BM25_PATH = Path(__file__).parent.parent / "data" / "bm25_index.pkl"
CHUNKS_PATH = Path(__file__).parent.parent / "data" / "chunks.json"
COLLECTION_NAME = "tesla_reports"


class DashScopeEmbeddingFunction(EmbeddingFunction):
    """Custom ChromaDB embedding function compatible with DashScope & OpenAI."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        self.model = model
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def __call__(self, input: Documents) -> Embeddings:
        """Embed in sub-batches of 10 to satisfy DashScope's limit, with retry."""
        import time
        texts = list(input)
        results = []
        sub_batch = 10
        for i in range(0, len(texts), sub_batch):
            batch = texts[i:i + sub_batch]
            for attempt in range(5):
                try:
                    response = self.client.embeddings.create(model=self.model, input=batch)
                    results.extend(item.embedding for item in response.data)
                    break
                except Exception as e:
                    if attempt == 4:
                        raise
                    wait = 2 ** attempt
                    print(f"\n  [retry {attempt+1}/4] {e} — waiting {wait}s...")
                    time.sleep(wait)
        return results


def get_embedding_function(
    api_key: str,
    model: str | None = None,
    base_url: str | None = None,
) -> DashScopeEmbeddingFunction:
    """Return ChromaDB-compatible embedding function (DashScope or OpenAI)."""
    import os
    model = model or os.environ.get("EMBEDDING_MODEL", "text-embedding-v3")
    base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    return DashScopeEmbeddingFunction(api_key=api_key, model=model, base_url=base_url)


def build_index(chunks: list[dict[str, Any]], api_key: str, embed_batch: int = 10, insert_batch: int = 100):
    """
    Build ChromaDB collection and BM25 index from chunks list.
    Pre-computes embeddings in small batches (DashScope max=10),
    then inserts with explicit embeddings to skip ChromaDB's internal call.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save chunks for retrieval metadata
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks to {CHUNKS_PATH}")

    # Build BM25
    tokenized = [build_chunk_document(c).lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"Saved BM25 index to {BM25_PATH}")

    # Pre-compute all embeddings in small batches
    ef = get_embedding_function(api_key)
    documents = [build_chunk_document(c) for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = []
    for c in chunks:
        metadatas.append({
            "filename": c.get("filename", ""),
            "year": c.get("year", ""),
            "quarter": c.get("quarter", ""),
            "period": c.get("period", ""),
            "doc_type": c.get("doc_type", ""),
            "section": c.get("section", ""),
            "chunk_type": c.get("chunk_type", ""),
            "page": c.get("page", 0),
        })

    print(f"Computing embeddings for {len(documents)} documents (batch={embed_batch})...")
    all_embeddings: list[list[float]] = []
    for i in range(0, len(documents), embed_batch):
        batch = documents[i:i + embed_batch]
        embs = ef(batch)
        all_embeddings.extend(embs)
        print(f"  Embedded {min(i + embed_batch, len(documents))}/{len(documents)}")

    # Build ChromaDB (no embedding function — we pass embeddings directly)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Insert in batches with precomputed embeddings
    print(f"Inserting into ChromaDB...")
    for i in range(0, len(documents), insert_batch):
        collection.add(
            documents=documents[i:i + insert_batch],
            embeddings=all_embeddings[i:i + insert_batch],
            ids=ids[i:i + insert_batch],
            metadatas=metadatas[i:i + insert_batch],
        )
        print(f"  Inserted {min(i + insert_batch, len(documents))}/{len(documents)}")

    print("Index build complete.")
    return collection


def load_index(api_key: str):
    """Load existing ChromaDB collection and BM25 index."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Load without embedding function — we pass query embeddings directly at query time
    collection = client.get_collection(name=COLLECTION_NAME)

    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return collection, bm25, chunks


def index_exists() -> bool:
    """Check if index files exist."""
    return CHUNKS_PATH.exists() and BM25_PATH.exists() and CHROMA_DIR.exists()
