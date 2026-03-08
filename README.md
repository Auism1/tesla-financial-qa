# Tesla SEC Filings Q&A System

An end-to-end Retrieval-Augmented Generation (RAG) system for querying Tesla's SEC filings (10-K annual reports and 10-Q quarterly reports) from 2021 to 2025.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key in .env
echo "OPENAI_API_KEY=sk-your-key" > .env
echo "OPENAI_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1" >> .env
echo "CHAT_MODEL=qwen-plus" >> .env
echo "EMBEDDING_MODEL=text-embedding-v3" >> .env

# 3. Build index (takes ~5–10 minutes for embeddings)
python ingest.py

# 4. Launch UI
python app.py
# → http://localhost:7860
```

---

## Data Overview

### Successfully Processed Documents

| Year | 10-K (Annual) | 10-Q Q1 | 10-Q Q2 | 10-Q Q3 |
|------|--------------|---------|---------|---------|
| 2021 | Tesla-10K-2021.json | Tesla-Q1-2021.json | Tesla-Q2-2021.json | Tesla-Q3-2021.json |
| 2022 | Tesla-10K-2022.json | Tesla-Q1-2022.json | Tesla-Q2-2022.json | Tesla-Q3-2022.json |
| 2023 | Tesla-10K-2023.json | Tesla-Q1-2023.json | Tesla-Q2-2023.json | Tesla-Q3-2023.json |
| 2024 | Tesla-10K-2024.json | Tesla-Q1-2024.json | Tesla-Q2-2024.json | Tesla-Q3-2024.json |
| 2025 | Tesla-10K-2025.json | Tesla-Q1-2025.json | Tesla-Q2-2025.json | Tesla-Q3-2025.json |

**Total: 20 files** (5 annual 10-K + 15 quarterly 10-Q). Note: Q4 reports are not separately filed; the 10-K covers the full fiscal year including Q4.

### Index Statistics

| Metric | Value |
|--------|-------|
| Total chunks indexed | 6,153 |
| Table chunks | 1,010 |
| Text chunks | 5,143 |
| Quarterly report chunks | 3,530 |
| Annual report chunks | 2,623 |
| Periods covered | 20 (FY2021–FY2025, Q1–Q3 per year) |

### Data Source Format

All documents are **MinerU JSON output** from PDF parsing (`_version_name: 2.7.5`). Each JSON file contains:
- `pdf_info`: array of page objects with `para_blocks`
- Block types: `title`, `text`, `list`, `table`, `table_caption`, `table_footnote`, `header`, `page_number`
- Tables embedded as HTML in `blocks[].lines[].spans[].html`

---

## System Design Decisions

### 1. Document Parsing (`src/json_parser.py`)

**What**: Reads MinerU JSON output directly, extracting structured blocks per page.

**How**:
- `title` blocks → section header detection via `SECTION_MAP` (exact match) + substring keyword fallback + SEC Item number patterns (e.g. "Item 1A" → RISK FACTORS)
- `text` and `list` blocks → plain text chunks with current section metadata
- `table` blocks → HTML extracted from nested span → BeautifulSoup parse → pipe-delimited markdown table (preserving all column headers, handling `colspan`)
- `table_caption` → buffered and prepended to the next table's label
- `table_footnote` → appended to the most recently emitted table chunk
- `header`, `page_number`, `ref_text` → skipped entirely

**Why JSON over Markdown/OCR**: MinerU JSON provides structured block-level access, making table extraction reliable and column-header-preserving. OCR-based approaches (pdf2image + Tesseract) lose column relationships. Plain markdown conversion loses block-type metadata needed for section detection.

### 2. Chunking Strategy (`src/chunker.py`)

**Section-based semantic chunking**:
- Text chunks within the same `(filename, section)` pair are merged before splitting, ensuring each chunk carries coherent meaning (e.g., all MD&A text for a given quarter stays together)
- Tables are **atomic** — each table is one indivisible chunk. Splitting a financial table would destroy the column-header relationship
- Maximum text chunk size: **1,200 characters** with **150-character overlap** between consecutive chunks

**Metadata per chunk**: `chunk_id`, `filename`, `year`, `quarter`, `period`, `doc_type`, `section`, `chunk_type`, `page`

**Embedding string**: `[{period} | {section}] {text}` — the prefix encodes temporal and topical context into each embedding, improving retrieval precision for time-specific queries.

**Why this strategy**: SEC filings have well-defined section boundaries (PART I Item 1, MD&A, Risk Factors, etc.). Respecting these boundaries means a query for "automotive gross margin" targets the right section rather than pulling fragments from unrelated sections.

### 3. Embedding (`src/indexer.py`)

- Provider: **Alibaba Cloud DashScope** (Singapore), OpenAI-compatible API
- Model: `text-embedding-v3` (1536-dim)
- DashScope batch limit = 10 inputs per API call; chunked internally
- Vector distance: cosine similarity in ChromaDB

### 4. Hybrid Retrieval (`src/retriever.py`)

```
Query → [BM25 top-20] + [Vector top-20] → RRF fusion (k=60) → top-k results
```

- **BM25** (BM25Okapi) ensures exact financial term matches ("Free Cash Flow", "Q3 2022", specific dollar amounts) are not missed by semantic drift
- **Vector search** (ChromaDB cosine) handles paraphrasing and concept-level queries
- **Reciprocal Rank Fusion** (k=60) combines both rankings without requiring score normalization

**Auto-filtering**: Year, quarter, and doc_type (`annual_report`/`quarterly_report`) are auto-extracted from the query via regex and applied as ChromaDB `where` clauses and BM25 post-filtering.

### 5. Multi-Step QA Pipeline (`src/qa_system.py`)

1. **Query analysis**: LLM classifies complexity (simple/complex), extracts years/quarters/topics, generates sub-queries for multi-hop problems
2. **Multi-hop retrieval**: Retrieves for main question + each sub-query; deduplicates by `chunk_id`
3. **Table-specific retrieval**: If query requires text+table join, explicitly re-searches for table chunks
4. **Context assembly**: Top-12 chunks by RRF score, formatted with `period | section | page` citations
5. **LLM generation**: `qwen-plus` with financial analyst system prompt requiring cited answers

---

## Test Set & Results

### Complex Test Questions (v2 — Post-Fix Results)

| # | Question | Type | Result | Key Finding |
|---|----------|------|--------|-------------|
| Q1 | Compare how 2021 10-K and 2023 10-K describe China market risks. What changed? | Cross-doc comparison | **Failure** | FINANCIAL STATEMENTS still retrieved; 10-K RISK FACTORS section labeling gap in parser |
| Q2 | Total R&D expense across all four quarters of 2022? Compare to FY2021. | Numerical calculation | **Success** | Found $3,075M (FY2022) vs $2,593M (FY2021); doc_type filter worked correctly |
| Q3 | Which quarter (2021–2024) had the lowest automotive gross margin? What does MD&A say? | Text + table join | **Partial** | Found Q1-2024 = 18.5%; automotive GM table mislabeled as SERVICES AND OTHER SEGMENT |
| Q4 | Tesla's free cash flow fluctuation quarter by quarter from 2021 to 2023. | Temporal multi-doc | **Partial (Improved)** | FCF derivation prompt added; retrieves LIQUIDITY sections; some YTD midpoints still missing |
| Q5 | How did supply chain risk disclosures evolve from Q1-2021 to Q3-2023? Cite quarters. | Multi-doc synthesis | **Partial (Improved)** | EXHIBITS blocklist restored 2021 RISK FACTORS retrieval; 2022–2023 still sparse |

**Overall rate**: 1 Success / 3 Partial / 1 Failure (same count; Q4 and Q5 answer quality significantly improved).

### Fixes Applied Between v1 and v2

| Fix | File | Change | Impact |
|-----|------|--------|--------|
| EXHIBITS blocklist | `src/retriever.py` | Block EXHIBITS/OTHER INFORMATION/PART IV from BM25 and post-RRF | Q5: 0 → 3 RISK FACTORS sources retrieved |
| Number normalization | `src/json_parser.py` | Normalize LaTeX `\$`, spaced decimals, space-thousands in extracted spans | Reduces "$1237 billion" type artifacts |
| doc_type auto-filter removed | `src/qa_system.py` | Only propagate `annual_report` from analysis; never auto-restrict to `quarterly_report` | Q3: 10-K annual data no longer excluded |
| Section-targeted retrieval | `src/retriever.py` + `src/qa_system.py` | Extra retrieval pass for RISK FACTORS / MD&A on narrative queries | Q5: improved; Q1 still fails due to parser labeling gap |
| FCF derivation prompt | `src/qa_system.py` | Instruct LLM to subtract prior YTD from current YTD for quarterly FCF | Q4: model now shows correct arithmetic |

### Remaining Systemic Issues

- **10-K RISK FACTORS section labeling**: Annual report RISK FACTORS pages are mislabeled during parsing (as FINANCIAL STATEMENTS or OVERVIEW), making section-targeted retrieval ineffective for 10-K risk queries
- **Automotive segment section mismatch**: Tables on pages titled "Automotive & Services and Other Segment" get labeled `SERVICES AND OTHER SEGMENT` instead of `AUTOMOTIVE SEGMENT`
- **Incomplete YTD coverage for FCF**: Multi-year FCF queries require 9 YTD midpoints simultaneously; not all fit in the top-12 context window

See [FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md) for detailed per-case root cause analysis and concrete improvement proposals.

---

## File Structure

```
tesla/
├── src/
│   ├── json_parser.py   # MinerU JSON parser (primary)
│   ├── chunker.py       # Semantic chunking
│   ├── indexer.py       # ChromaDB + BM25 index builder/loader
│   ├── retriever.py     # Hybrid BM25+vector with RRF fusion
│   └── qa_system.py     # Multi-step QA pipeline
├── dataset/             # 20 MinerU JSON files (2021–2025)
│   ├── 2021/
│   ├── 2022/
│   ├── 2023/
│   ├── 2024/
│   └── 2025/
├── data/
│   ├── chunks.json      # 6,153 final chunks
│   └── bm25_index.pkl   # BM25Okapi serialized
├── chroma_db/           # ChromaDB persistent vector store
├── app.py               # Gradio web UI (port 7860)
├── ingest.py            # Ingestion pipeline
├── requirements.txt
└── .env                 # API keys (not committed)
```

## CLI Reference

```bash
python ingest.py              # Full parse + index build
python ingest.py --index-only # Rebuild index from existing chunks.json only
python app.py                 # Launch Gradio UI at http://localhost:7860
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| chromadb | 0.4.22 | Vector store |
| openai | 1.12.0 | API client (OpenAI-compatible) |
| rank-bm25 | 0.2.2 | BM25 keyword search |
| gradio | 4.19.2 | Web UI |
| beautifulsoup4 | 4.12.3 | HTML table parsing |
| python-dotenv | 1.0.1 | Environment config |
| numpy | 1.26.4 | Numerical operations |
| tqdm | 4.66.1 | Progress bars |
