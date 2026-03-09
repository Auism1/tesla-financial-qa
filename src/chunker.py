"""
特斯拉财报块的语义分块器。
策略：
  1. 表格保持为独立的原子块（从不拆分）。
  2. 章节内的文本块合并至 MAX_CHARS。
  3. 长文本块使用句子感知重叠（OVERLAP_CHARS）拆分。
  4. 每个块保留完整元数据：年份、季度、章节、页码、文档类型。
"""
import re
from typing import Any

MAX_CHARS = 1200       # 文本块的软最大值
OVERLAP_CHARS = 150    # 连续文本块之间的重叠


def split_sentences(text: str) -> list[str]:
    """粗略的句子拆分器。"""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text_block(text: str, meta: dict, section: str, page: int, subsection: str = "") -> list[dict[str, Any]]:
    """将长文本块拆分为重叠的块。"""
    if len(text) <= MAX_CHARS:
        return [{
            "text": text,
            "chunk_type": "text",
            "section": section,
            "subsection": subsection,
            "page": page,
            **meta,
        }]
    sentences = split_sentences(text)
    result = []
    current = ""
    prev_tail = ""
    for sent in sentences:
        candidate = (prev_tail + " " + current + " " + sent).strip()
        if len(candidate) > MAX_CHARS and current:
            result.append({
                "text": current.strip(),
                "chunk_type": "text",
                "section": section,
                "subsection": subsection,
                "page": page,
                **meta,
            })
            prev_tail = current[-OVERLAP_CHARS:].strip()
            current = sent
        else:
            current = (current + " " + sent).strip()
    if current:
        result.append({
            "text": (prev_tail + " " + current).strip(),
            "chunk_type": "text",
            "section": section,
            "subsection": subsection,
            "page": page,
            **meta,
        })
    return result


def merge_and_rechunk(raw_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    1. 保持表格块不变。
    2. 合并来自同一文档 + 章节的连续文本块。
    3. 如果合并后的文本 > MAX_CHARS，则重新拆分。
    4. 分配唯一的 chunk_id。
    """
    result: list[dict[str, Any]] = []
    chunk_id = 0

    # 按 (filename, section) 分组以合并文本
    groups: list[list[dict]] = []
    current_group: list[dict] = []

    for chunk in raw_chunks:
        if chunk["chunk_type"] == "table":
            # 刷新当前组
            if current_group:
                groups.append(current_group)
                current_group = []
            groups.append([chunk])
        else:
            if not current_group:
                current_group.append(chunk)
            else:
                prev = current_group[-1]
                # 相同文档 + 章节 + 子章节 → 合并
                if (prev["filename"] == chunk["filename"] and
                        prev["section"] == chunk["section"] and
                        prev.get("subsection", "") == chunk.get("subsection", "")):
                    current_group.append(chunk)
                else:
                    groups.append(current_group)
                    current_group = [chunk]
    if current_group:
        groups.append(current_group)

    for group in groups:
        if len(group) == 1 and group[0]["chunk_type"] == "table":
            c = dict(group[0])
            c["chunk_id"] = f"chunk_{chunk_id:05d}"
            result.append(c)
            chunk_id += 1
            continue

        # 合并文本
        merged_text = " ".join(c["text"] for c in group).strip()
        base_meta = {k: v for k, v in group[0].items()
                     if k not in ("text", "chunk_type", "section", "subsection", "page")}
        section = group[0]["section"]
        subsection = group[0].get("subsection", "")
        page = group[0]["page"]

        sub_chunks = chunk_text_block(merged_text, base_meta, section, page, subsection)
        for sc in sub_chunks:
            sc["chunk_id"] = f"chunk_{chunk_id:05d}"
            result.append(sc)
            chunk_id += 1

    return result


def build_chunk_document(chunk: dict[str, Any]) -> str:
    """
    构建将被嵌入的字符串。
    添加元数据前缀以更好地检索时间特定查询。
    """
    section_label = chunk["section"]
    if chunk.get("subsection"):
        section_label = f"{section_label} > {chunk['subsection']}"
    prefix = f"[{chunk['period']} | {section_label}] "
    return prefix + chunk["text"]
