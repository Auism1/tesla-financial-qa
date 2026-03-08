"""
Hybrid retriever: combines BM25 keyword search + ChromaDB vector search.
Supports metadata filtering (year, quarter, doc_type).
"""
import os
import re
from typing import Any

from openai import OpenAI
from src.chunker import build_chunk_document


def embed_query(query: str) -> list[float]:
    """Embed a single query string using the configured embedding API."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-v3")
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    resp = client.embeddings.create(model=model, input=[query])
    return resp.data[0].embedding

# Reciprocal Rank Fusion constant
RRF_K = 60

# Sections containing legal boilerplate that pollutes retrieval for substantive queries
SECTION_BLOCKLIST = {
    "EXHIBITS",
    "OTHER INFORMATION",
    "PART IV",
    "SECURITY OWNERSHIP",
    "CERTIFICATIONS",
}


def extract_query_filters(query: str) -> dict[str, list[str]]:
    """
    Extract year/quarter/doc_type filters from natural language query.
    e.g. "2022 Q3 revenue" → {"years": ["2022"], "quarters": ["Q3"]}
    e.g. "2021 10-K annual report" → {"years": ["2021"], "doc_type": "annual_report"}
    """
    years = re.findall(r"\b(202[0-9])\b", query)
    quarters = re.findall(r"\b(Q[1-3])\b", query, re.IGNORECASE)  # Only Q1-Q3 exist as 10-Q
    quarters = [q.upper() for q in quarters]

    doc_type: str | None = None
    # Q4 / fourth quarter / full-year → annual 10-K (no Q4 quarterly report exists)
    if re.search(r"\b(Q4|fourth quarter|full.?year|annual report|annual filing|10-?K|10K|FY20\d{2})\b", query, re.IGNORECASE):
        doc_type = "annual_report"
    elif re.search(r"\b(10-?Q|quarterly report|quarterly filing)\b", query, re.IGNORECASE):
        doc_type = "quarterly_report"

    result: dict[str, list[str]] = {
        "years": list(set(years)),
        "quarters": list(set(quarters)),
    }
    if doc_type:
        result["doc_type"] = doc_type  # type: ignore[assignment]
    return result


def build_chroma_where(filters: dict[str, list[str]]) -> dict | None:
    """Build ChromaDB where clause from filters.
    NOTE: This function is currently unused. hybrid_search() builds its own
    where conditions inline with proper $in/$eq handling.
    """
    years = filters.get("years", [])
    quarters = filters.get("quarters", [])

    conditions = []
    if years:
        if len(years) == 1:
            conditions.append({"year": {"$eq": years[0]}})
        else:
            conditions.append({"year": {"$in": years}})
    if quarters:
        if len(quarters) == 1:
            conditions.append({"quarter": {"$eq": quarters[0]}})
        else:
            conditions.append({"quarter": {"$in": quarters}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def bm25_search(
    query: str,
    bm25,
    chunks: list[dict],
    top_k: int = 20,
    year_filter: list[str] | None = None,
    quarter_filter: list[str] | None = None,
    section_blocklist: set[str] | None = None,
) -> list[tuple[int, float]]:
    """Return (chunk_index, bm25_score) sorted by score descending."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Apply metadata filters
    filtered = []
    for idx, score in enumerate(scores):
        c = chunks[idx]
        if year_filter and c.get("year") not in year_filter:
            continue
        if quarter_filter and c.get("quarter") not in quarter_filter:
            continue
        if section_blocklist and c.get("section") in section_blocklist:
            continue
        filtered.append((idx, float(score)))

    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[:top_k]


def vector_search(
    query: str,
    collection,
    top_k: int = 20,
    where: dict | None = None,
) -> list[tuple[str, float]]:
    """Return (chunk_id, distance) sorted by distance ascending (lower=better)."""
    query_embedding = embed_query(query)
    kwargs: dict[str, Any] = {"query_embeddings": [query_embedding], "n_results": top_k}
    if where:
        kwargs["where"] = where
    results = collection.query(**kwargs)
    ids = results["ids"][0]
    distances = results["distances"][0]
    return list(zip(ids, distances))


def rrf_fusion(
    bm25_results: list[tuple[int, float]],
    vector_results: list[tuple[str, float]],
    chunks: list[dict],
) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion of BM25 and vector results.
    Returns merged list of chunks sorted by combined RRF score.
    """
    # Build chunk_id → index map
    id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}

    rrf_scores: dict[str, float] = {}

    # BM25 ranks
    for rank, (idx, _) in enumerate(bm25_results):
        cid = chunks[idx]["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (RRF_K + rank + 1)

    # Vector ranks
    for rank, (cid, _) in enumerate(vector_results):
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (RRF_K + rank + 1)

    # Sort by RRF score
    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    result = []
    for cid, score in sorted_ids:
        if cid in id_to_idx:
            c = dict(chunks[id_to_idx[cid]])
            c["rrf_score"] = round(score, 6)
            result.append(c)
    return result


def hybrid_search(
    query: str,
    collection,
    bm25,
    chunks: list[dict],
    top_k: int = 8,
    year_filter: list[str] | None = None,
    quarter_filter: list[str] | None = None,
    doc_type_filter: str | None = None,
    section_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Full hybrid search pipeline:
    1. Extract filters from query (augmented by explicit filters).
    2. Run BM25 + vector search.
    3. RRF fusion.
    4. Return top_k chunks.
    """
    # Auto-extract filters from query
    auto_filters = extract_query_filters(query)
    years = year_filter or auto_filters.get("years") or []
    quarters = quarter_filter or auto_filters.get("quarters") or []
    # Use explicit doc_type_filter or auto-detected one
    if not doc_type_filter:
        doc_type_filter = auto_filters.get("doc_type")  # type: ignore[assignment]

    # Build chroma where clause
    where_conditions = []
    if years:
        where_conditions.append({"year": {"$in": years}} if len(years) > 1 else {"year": {"$eq": years[0]}})
    if quarters:
        where_conditions.append({"quarter": {"$in": quarters}} if len(quarters) > 1 else {"quarter": {"$eq": quarters[0]}})
    if doc_type_filter:
        where_conditions.append({"doc_type": {"$eq": doc_type_filter}})
    if section_filter:
        where_conditions.append({"section": {"$eq": section_filter}})

    if len(where_conditions) == 0:
        where = None
    elif len(where_conditions) == 1:
        where = where_conditions[0]
    else:
        where = {"$and": where_conditions}

    # BM25 (with section blocklist to suppress boilerplate sections)
    # When section_filter is set, override blocklist with an allowlist approach
    effective_blocklist = None if section_filter else SECTION_BLOCKLIST
    bm25_res = bm25_search(
        query, bm25, chunks, top_k=20,
        year_filter=years or None,
        quarter_filter=quarters or None,
        section_blocklist=effective_blocklist,
    )
    # If section_filter specified, restrict BM25 results to that section
    if section_filter:
        bm25_res = [(idx, score) for idx, score in bm25_res if chunks[idx].get("section") == section_filter]

    # Vector
    try:
        vec_res = vector_search(query, collection, top_k=20, where=where)
    except Exception:
        # Fallback without filter if ChromaDB filter fails
        vec_res = vector_search(query, collection, top_k=20)

    # Fuse and remove blocked sections from final results
    merged = rrf_fusion(bm25_res, vec_res, chunks)
    merged = [c for c in merged if c.get("section") not in SECTION_BLOCKLIST]
    return merged[:top_k]
