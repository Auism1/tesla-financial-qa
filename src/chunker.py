"""
Semantic chunker for Tesla financial report chunks.
Strategy:
  1. Tables are kept as individual atomic chunks (never split).
  2. Text chunks within a section are merged up to MAX_CHARS.
  3. Long text chunks are split with sentence-aware overlap (OVERLAP_CHARS).
  4. Each chunk retains full metadata: year, quarter, section, page, doc_type.
"""
import re
from typing import Any

MAX_CHARS = 1200       # soft max for text chunks
OVERLAP_CHARS = 150    # overlap between consecutive text chunks


def split_sentences(text: str) -> list[str]:
    """Rough sentence splitter."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text_block(text: str, meta: dict, section: str, page: int, subsection: str = "") -> list[dict[str, Any]]:
    """Split a long text block into overlapping chunks."""
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
    1. Keep table chunks as-is.
    2. Merge consecutive text chunks from the same document + section.
    3. Re-split merged text if > MAX_CHARS.
    4. Assign a unique chunk_id.
    """
    result: list[dict[str, Any]] = []
    chunk_id = 0

    # Group by (filename, section) for merging text
    groups: list[list[dict]] = []
    current_group: list[dict] = []

    for chunk in raw_chunks:
        if chunk["chunk_type"] == "table":
            # Flush current group
            if current_group:
                groups.append(current_group)
                current_group = []
            groups.append([chunk])
        else:
            if not current_group:
                current_group.append(chunk)
            else:
                prev = current_group[-1]
                # Same doc + section + subsection → merge
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

        # Merge text
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
    Build the string that will be embedded.
    Prefix with metadata for better retrieval of time-specific queries.
    """
    section_label = chunk["section"]
    if chunk.get("subsection"):
        section_label = f"{section_label} > {chunk['subsection']}"
    prefix = f"[{chunk['period']} | {section_label}] "
    return prefix + chunk["text"]
