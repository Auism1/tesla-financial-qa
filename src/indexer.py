"""
索引器：从处理后的块构建 ChromaDB 向量存储 + BM25 索引。
支持增量重建和持久化。
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
    """与 DashScope 和 OpenAI 兼容的自定义 ChromaDB 嵌入函数。"""

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        self.model = model
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def __call__(self, input: Documents) -> Embeddings:
        """以 10 个为一批进行嵌入以满足 DashScope 的限制，带重试。"""
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
                    print(f"\n  [重试 {attempt+1}/4] {e} — 等待 {wait}秒...")
                    time.sleep(wait)
        return results


def get_embedding_function(
    api_key: str,
    model: str | None = None,
    base_url: str | None = None,
) -> DashScopeEmbeddingFunction:
    """返回 ChromaDB 兼容的嵌入函数（DashScope 或 OpenAI）。"""
    import os
    model = model or os.environ.get("EMBEDDING_MODEL", "text-embedding-v3")
    base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    return DashScopeEmbeddingFunction(api_key=api_key, model=model, base_url=base_url)


def build_index(chunks: list[dict[str, Any]], api_key: str, embed_batch: int = 10, insert_batch: int = 100):
    """
    从块列表构建 ChromaDB 集合和 BM25 索引。
    以小批量预计算嵌入（DashScope 最大=10），
    然后使用显式嵌入插入以跳过 ChromaDB 的内部调用。
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 保存块以用于检索元数据
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(chunks)} 个块到 {CHUNKS_PATH}")

    # 构建 BM25
    tokenized = [build_chunk_document(c).lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"已保存 BM25 索引到 {BM25_PATH}")

    # 以小批量预计算所有嵌入
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

    print(f"为 {len(documents)} 个文档计算嵌入（批次={embed_batch}）...")
    all_embeddings: list[list[float]] = []
    for i in range(0, len(documents), embed_batch):
        batch = documents[i:i + embed_batch]
        embs = ef(batch)
        all_embeddings.extend(embs)
        print(f"  已嵌入 {min(i + embed_batch, len(documents))}/{len(documents)}")

    # 构建 ChromaDB（无嵌入函数 — 我们直接传递嵌入）
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
