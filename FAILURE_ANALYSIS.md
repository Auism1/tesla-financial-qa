# Failure Analysis — Tesla SEC Filings Q&A System (v2)

Deep-dive analysis of 5 test cases run against the actual system (JSON dataset, 6,153 chunks, after applying all three systemic fixes). Results recorded from `python run_tests.py`.

---

## Test Results Summary

| Case | Question | Overall Result | Root Cause Category |
|------|----------|---------------|-------------------|
| Q1 | China market risks 2021 vs 2023 10-K | **Failure** | Wrong section retrieved — 10-K RISK FACTORS chunks not surfacing |
| Q2 | Total R&D expense 2022 vs FY2021 | **Success** | — |
| Q3 | Lowest automotive gross margin + MD&A | **Partial** | Section label mismatch (automotive GM table labeled as SERVICES AND OTHER SEGMENT) |
| Q4 | FCF quarter-by-quarter 2021–2023 | **Partial (Improved)** | YTD-only CF in 10-Q; FCF derivation instruction now added to LLM prompt |
| Q5 | Supply chain risk evolution Q1-2021→Q3-2023 | **Partial (Improved)** | EXHIBITS blocklist fix works; 2022–2023 RISK FACTORS still sparse in context |

---

## Case 1: China Market Risk Comparison (Q1) — FAILURE

**Question**: *Compare how 2021 10-K and 2023 10-K describe China market risks. What changed?*

**Observed behavior**: Query analysis correctly set `doc_type="annual_report"`, `years=["2021","2023"]`. The section-targeted retrieval pass for risk-factor queries (matching "market risk" keyword) fired but still failed to surface RISK FACTORS content. Top sources retrieved:
```
('FY2021', 'FINANCIAL STATEMENTS', 'table')
('FY2023', 'PART III', 'text')
('FY2021', 'OVERVIEW', 'text')
('FY2023', 'FINANCIAL STATEMENTS', 'text')
('FY2023', 'LEGAL PROCEEDINGS', 'text')
```
No chunks from RISK FACTORS section appeared. The LLM correctly reported missing context and could not compare the two years.

**What changed vs previous run**: The EXHIBITS blocklist fix removed certification/exhibit chunks from results — sources no longer include EXHIBITS or OTHER INFORMATION. However, FINANCIAL STATEMENTS and PART III still dominate. The section-targeted retrieval pass for RISK FACTORS ran (triggered by "market risk" keyword) but returned no results, meaning either the 10-K RISK FACTORS section chunks have a different section label in the index, or the BM25/vector match within that section didn't rank highly for "China market risk."

**Root cause tracing**:
- BM25 scores "China" most highly against financial statement footnotes (e.g., "Tesla (Shanghai) Co., Ltd." appears in asset tables with high term frequency in a structured list, boosting TF).
- Vector search maps "China market risks" closest to MARKET RISK DISCLOSURES section (interest rate / FX risk language), not to RISK FACTORS narrative.
- The 10-K RISK FACTORS section chunks discussing China are likely indexed under a different section label due to complex 10-K page structure, so the section-targeted retrieval pass finds an empty result set.

**Root cause**: **Section detection gap in 10-K parsing** — Some RISK FACTORS pages in the 10-K annual reports are mis-labeled (e.g., assigned to "FINANCIAL STATEMENTS" or "OVERVIEW") due to preceding title blocks. Additionally, section-blind RRF cannot distinguish RISK FACTORS narrative from financial table chunks containing the word "China."

**Concrete improvements**:
1. **Inspect 10-K RISK FACTORS chunks**: Query `chunks.json` to check how many 10-K chunks carry `section="RISK FACTORS"` — if zero, the 10-K RISK FACTORS section is being mis-labeled at parse time.
2. **Strengthen sub-query re-formulation**: For "compare risk disclosures" queries, auto-generate a sub-query like `"Risk Factors Item 1A China 2021 annual report"` with explicit Item 1A language to boost BM25 term matching against section-prefixed embeddings `[FY2021 | RISK FACTORS]`.
3. **Expand `_ITEM_MAP`** in `json_parser.py`: Verify 10-K Item 1A detection is firing correctly; add additional aliases for "Risk Factors" that appear in 10-K table-of-contents title blocks.

---

## Case 2: R&D Expense Annual Comparison (Q2) — SUCCESS

**Question**: *What was the total R&D expense across all four quarters of 2022? Compare to FY2021 from the 2021 annual report.*

**Observed behavior**: Query analysis set `doc_type="annual_report"`, `years=["2022","2021"]`, `quarters=[]`. Successfully retrieved and cited:
- FY2022 R&D: **$3,075 million**
- FY2021 R&D: **$2,593 million**
- YoY increase: **+$482M (+19%)**

The LLM correctly explained that "all four quarters of 2022" = FY2022 10-K figure, as there is no Q4 standalone 10-Q. The comparative income statement table with columns for 2022, 2021, 2020 was found as Source 9.

**Why it worked**: `doc_type="annual_report"` filter correctly restricted retrieval. R&D expense appears in a standardized income statement table with high keyword density (BM25 advantage). "Research and development" embeds semantically close to "R&D expense."

**No failure to analyze.**

---

## Case 3: Lowest Automotive Gross Margin + MD&A (Q3) — PARTIAL

**Question**: *Which quarter from 2021 to 2024 had the lowest automotive gross margin? What does the MD&A say about that quarter?*

**Observed behavior**: Query analysis returned `doc_type="quarterly_report"` (LLM decision), `years=[2021–2024]`, `quarters=["Q1","Q2","Q3"]`. Fix 3 (doc_type filter) correctly suppresses `quarterly_report` from being propagated to `hybrid_search`. All 8 top sources were from `SERVICES AND OTHER SEGMENT` section. Found Q1-2024 = 18.5% as the lowest reported margin.

**Failure symptoms**:
1. **Section label mismatch**: Automotive gross margin tables are parsed under section `SERVICES AND OTHER SEGMENT` (because the preceding title block on the page said "Automotive & Services and Other Segment") rather than `AUTOMOTIVE SEGMENT`. The table content is correct but section metadata is wrong.
2. **Q4 data still missing**: Even with the doc_type filter removed, 10-K automotive gross margin tables are also indexed under the wrong section label, so they don't surface reliably. Q4-2022, Q4-2023 remain unexamined.
3. **Section-targeted retrieval not triggered**: The `_MDA_KEYWORDS` regex matched but the subsequent MD&A section-targeted pass retrieved MD&A text chunks that don't contain the automotive gross margin table.

**Root cause**: **Section label assigned at page level** — When a 10-Q page header says "Automotive & Services and Other Segment" followed by tables for both sub-segments, the parser assigns `SERVICES AND OTHER SEGMENT` to all tables on that page. The automotive gross margin table gets the wrong section label.

**Concrete improvements**:
1. **Fix SECTION_MAP** in `json_parser.py`: Add mapping `"AUTOMOTIVESERVICESANDOTHERSEGMENT"` → `"AUTOMOTIVE SEGMENT"` (currently maps to `SERVICES AND OTHER SEGMENT`).
2. **Multi-pass table retrieval**: For gross margin queries, run a targeted pass with query `"automotive gross margin table percent"` restricted to `chunk_type="table"` without section constraint.
3. **Direct table search as fallback**: If fewer than 3 table chunks appear in top results for a metric-lookup query, run a secondary retrieval pass with `chunk_type="table"` filter to guarantee table coverage.

---

## Case 4: Free Cash Flow Quarter-by-Quarter (Q4_FCF) — PARTIAL (Improved)

**Question**: *Describe Tesla's free cash flow fluctuation quarter by quarter from 2021 to 2023.*

**Observed behavior** (improved vs v1): Sources now include LIQUIDITY AND CAPITAL RESOURCES and MD&A sections. The LLM correctly:
1. Explained that 10-Q filings only report YTD cumulative operating cash flows, not standalone quarterly FCF.
2. Demonstrated the subtraction methodology (Q2 FCF = 6-month YTD − Q1 YTD) using the FCF derivation instruction added to the LLM prompt.
3. Computed partial FCF figures for available periods (Q1-2021, Q2-2021, Q1-2023).

**Fix 5 impact**: The FCF derivation instruction explicitly teaches the model to subtract prior YTD from current YTD. The model now shows its arithmetic and correctly flags when intermediate YTD data is missing from context, rather than presenting YTD cumulative figures as if they were quarterly.

**Remaining failure symptoms**:
1. **Incomplete YTD coverage**: Not all 9 quarterly YTD periods (Q1–Q3 for 2021, 2022, 2023) were retrieved in a single pass. Missing: Q2-2022 YTD, Q3-2022 YTD, Q2-2023 YTD, Q3-2023 YTD.
2. **Number formatting artifacts remain**: Some parsed text still contains spacing anomalies in numbers. The normalization fixes address LaTeX-style artifacts but not all MinerU span-merge separators.

**Root cause**: (1) **SEC 10-Q inherent structure** — Quarterly FCF requires YTD subtraction across two retrieved chunks, requiring both to be present in top-12 context simultaneously. (2) **Retrieval breadth vs. depth trade-off** — Retrieving 9 sub-queries (one per quarter-period) is expensive and dilutes top-k slots.

**Concrete improvements**:
1. **Structured FCF sub-query generation**: When query mentions "free cash flow" + multiple years, auto-generate YTD pair sub-queries — for each quarter, retrieve both current and prior YTD to guarantee both endpoints are present.
2. **Increase context window**: Raise `max_chunks` from 12 to 16–20 for FCF queries to accommodate multiple YTD periods.
3. **Post-retrieval arithmetic step**: Dedicated FCF computation step that extracts YTD numbers from retrieved chunks and computes Q-over-Q differences before passing to the LLM.

---

## Case 5: Supply Chain Risk Evolution Q1-2021 to Q3-2023 (Q5) — PARTIAL (Significantly Improved)

**Question**: *How did Tesla's risk factor disclosures about supply chain evolve from 2021 Q1 to 2023 Q3? Cite specific quarters.*

**Observed behavior** (significantly improved vs v1): Top sources now include:
```
('Q1-2021', 'RISK FACTORS', 'text')
('Q2-2021', 'RISK FACTORS', 'text')
('Q3-2021', 'RISK FACTORS', 'text')
('Q2-2022', 'MANAGEMENT'S DISCUSSION AND ANALYSIS', 'text')
('Q3-2022', 'MANAGEMENT'S DISCUSSION AND ANALYSIS', 'text')
('Q1-2023', 'ENERGY SEGMENT', 'text')
```

The LLM delivered a substantive, well-cited answer:
- **Q1-2021**: Broad pandemic/event framing; semiconductor shortage first explicitly named as current active risk
- **Q2-2021**: Port congestion, labor shortages added
- **Q3-2021**: Geopolitical disruption framing deepened
- **Q3-2022**: Russia-Ukraine war impact; battery material sourcing risks added
- 2023: Limited coverage — retrieved ENERGY SEGMENT instead of RISK FACTORS for 2023 quarters

**Fix 1 impact**: EXHIBITS blocklist completely eliminated certification/exhibit pollution. RISK FACTORS sections for 2021 now dominate the top results — a major qualitative improvement from the previous run where all 6 top sources were from EXHIBITS or OTHER INFORMATION (returning zero useful content).

**Remaining failure symptoms**:
1. **2022–2023 RISK FACTORS absent**: Q2-2022, Q1-2022, Q1-2023 RISK FACTORS chunks not retrieved — MD&A or ENERGY SEGMENT chunks appear instead.
2. **Section-targeted pass partially effective**: The `_RISK_KEYWORDS` regex matched the query and the RISK FACTORS section pass retrieved 2021 data correctly but not 2022–2023 periods, suggesting inconsistent section labeling across years.

**Root cause**: **Inconsistent RISK FACTORS section labeling across years** — 2021 quarterly reports have RISK FACTORS sections cleanly labeled (Item 1A detection works). 2022–2023 quarterly reports may have different structural layouts or title block formats causing the section detector to assign different labels to the risk factor text.

**Concrete improvements**:
1. **Verify 2022–2023 RISK FACTORS chunks**: Check `data/chunks.json` for `year in ["2022","2023"]` and `section="RISK FACTORS"` to confirm they exist and are correctly labeled.
2. **Expand keyword patterns**: Check if 2022–2023 10-Q files use different Item numbering or phrasing (e.g., "ITEM 1A." vs "Item 1A.") that isn't matched by current `_ITEM_MAP` patterns.
3. **Fallback to broader retrieval**: When section-targeted pass returns < 3 results, fall back to unrestricted search with an explicit sub-query like `"supply chain risk Item 1A quarterly filing {year}"`.

---

## Systemic Issues — Updated Status

### Issue 1: Number Formatting Artifacts from MinerU Parsing

**Status**: Partially fixed.

**Fix applied**: Added `_normalize_financial_numbers()` in `json_parser.py` — handles `\$ X` LaTeX prefix, spaced decimals (`$12 30` → `$12.30`), space-separated thousands (`$1 237` → `$1,237`).

**Remaining**: Artifacts where MinerU outputs numbers as separate span tokens without the dollar prefix (e.g., plain `1237` in a number column) require table-level normalization during HTML-to-markdown conversion.

### Issue 2: EXHIBITS/OTHER INFORMATION Section Pollution

**Status**: Fixed. ✓

**Fix applied**: `SECTION_BLOCKLIST = {"EXHIBITS", "OTHER INFORMATION", "PART IV", "SECURITY OWNERSHIP", "CERTIFICATIONS"}` in `retriever.py`. BM25 skips blocklisted chunks during scoring; post-RRF filter removes any that survive vector search.

**Result**: Q5 improved from 0/8 relevant sources to 3/8 RISK FACTORS sources. Q1 no longer returns EXHIBITS/certification chunks.

### Issue 3: doc_type Auto-Filter Excluding Q4 Annual Data

**Status**: Fixed. ✓

**Fix applied**: In `qa_system.py`, only propagate `doc_type="annual_report"` from query analysis to `hybrid_search`. Never auto-restrict to `quarterly_report`.

**Result**: Q3 (automotive gross margin) no longer auto-filters to quarterly_report, allowing 10-K annual data to potentially appear. The remaining issue (section mislabeling) is a separate parser problem.

---

## Summary Table (v2 — Post-Fix)

| Case | Symptom | Root Cause | Fix Applied | Remaining Issue |
|------|---------|-----------|-------------|-----------------|
| Q1 | FINANCIAL STATEMENTS retrieved instead of RISK FACTORS | 10-K RISK FACTORS mislabeled; query-to-section embedding mismatch | Blocklist + section-targeted pass | Parser-level 10-K section labeling |
| Q2 | — (Success) | — | — | None |
| Q3 | Automotive GM table labeled as wrong section | Page-level section assignment; combined header mismapped | doc_type auto-filter removed | Parser SECTION_MAP fix needed |
| Q4 | YTD-only CF; incomplete quarterly coverage | SEC 10-Q structure limitation | FCF derivation prompt; number normalization | Incomplete YTD retrieval coverage |
| Q5 | EXHIBITS blocked 2022–2023 RISK FACTORS | EXHIBITS keyword pollution | Blocklist fix; section-targeted retrieval | 2022–2023 RISK FACTORS inconsistently labeled |
