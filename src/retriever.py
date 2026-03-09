"""
混合检索器：结合 BM25 关键词搜索 + ChromaDB 向量搜索。
支持元数据过滤（年份、季度、文档类型）。
"""
import os
import re
from typing import Any

from openai import OpenAI
from src.chunker import build_chunk_document


def embed_query(query: str) -> list[float]:
    """使用配置的嵌入 API 嵌入单个查询字符串。"""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-v3")
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    resp = client.embeddings.create(model=model, input=[query])
    return resp.data[0].embedding

# 倒数排名融合常数
RRF_K = 60

# 包含法律样板文本的章节，会污染实质性查询的检索
SECTION_BLOCKLIST = {
    "EXHIBITS",
    "OTHER INFORMATION",
    "PART IV",
    "SECURITY OWNERSHIP",
    "CERTIFICATIONS",
}


def extract_query_filters(query: str) -> dict[str, list[str]]:
    """
    从自然语言查询中提取年份/季度/文档类型过滤器。
    例如 "2022 Q3 revenue" → {"years": ["2022"], "quarters": ["Q3"]}
    例如 "2021 10-K annual report" → {"years": ["2021"], "doc_type": "annual_report"}
    """
    years = re.findall(r"\b(202[0-9])\b", query)
    quarters = re.findall(r"\b(Q[1-3])\b", query, re.IGNORECASE)  # 仅 Q1-Q3 作为 10-Q 存在
    quarters = [q.upper() for q in quarters]

    doc_type: str | None = None
    # Q4 / 第四季度 / 全年 → 年度 10-K（不存在 Q4 季度报告）
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
    """从过滤器构建 ChromaDB where 子句。
    注意：此函数当前未使用。hybrid_search() 使用正确的 $in/$eq 处理
    内联构建自己的 where 条件。
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
    """返回按分数降序排序的 (chunk_index, bm25_score)。"""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # 应用元数据过滤器
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
    """返回按距离升序排序的 (chunk_id, distance)（越低越好）。"""
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
    BM25 和向量结果的倒数排名融合。
    返回按组合 RRF 分数排序的块合并列表。
    """
    # 构建 chunk_id → index 映射
    id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}

    rrf_scores: dict[str, float] = {}

    # BM25 排名
    for rank, (idx, _) in enumerate(bm25_results):
        cid = chunks[idx]["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (RRF_K + rank + 1)

    # 向量排名
    for rank, (cid, _) in enumerate(vector_results):
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (RRF_K + rank + 1)

    # 按 RRF 分数排序
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
    完整的混合搜索流水线：
    1. 从查询中提取过滤器（由显式过滤器增强）。
    2. 运行 BM25 + 向量搜索。
    3. RRF 融合。
    4. 返回 top_k 块。
    """
    # 从查询中自动提取过滤器
    auto_filters = extract_query_filters(query)
    years = year_filter or auto_filters.get("years") or []
    quarters = quarter_filter or auto_filters.get("quarters") or []
    # 使用显式 doc_type_filter 或自动检测的过滤器
    if not doc_type_filter:
        doc_type_filter = auto_filters.get("doc_type")  # type: ignore[assignment]

    # 构建 chroma where 子句
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

    # BM25（使用章节黑名单来抑制样板章节）
    # 当设置 section_filter 时，用白名单方法覆盖黑名单
    effective_blocklist = None if section_filter else SECTION_BLOCKLIST
    bm25_res = bm25_search(
        query, bm25, chunks, top_k=20,
        year_filter=years or None,
        quarter_filter=quarters or None,
        section_blocklist=effective_blocklist,
    )
    # 如果指定了 section_filter，将 BM25 结果限制为该章节
    if section_filter:
        bm25_res = [(idx, score) for idx, score in bm25_res if chunks[idx].get("section") == section_filter]

    # 向量搜索
    try:
        vec_res = vector_search(query, collection, top_k=20, where=where)
    except Exception:
        # 如果 ChromaDB 过滤器失败，回退到无过滤器
        vec_res = vector_search(query, collection, top_k=20)

    # 融合并从最终结果中删除被阻止的章节
    merged = rrf_fusion(bm25_res, vec_res, chunks)
    merged = [c for c in merged if c.get("section") not in SECTION_BLOCKLIST]
    return merged[:top_k]
