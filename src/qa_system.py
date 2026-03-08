"""
Multi-step QA system for Tesla financial reports.
Uses OpenAI chat completion with retrieved context.
Implements a two-stage pipeline for complex cross-document questions.
"""
import os
import re
from typing import Any

from openai import OpenAI

from src.retriever import hybrid_search, extract_query_filters

# Keywords that signal a query is about risk factor narrative content
_RISK_KEYWORDS = re.compile(
    r"\b(risk factor|risk disclosure|10-?K risk|item 1a|supply chain risk|market risk|"
    r"china risk|geopolit|regulatory risk|litigation risk|competition risk)\b",
    re.IGNORECASE,
)
# Keywords that signal MD&A narrative content
_MDA_KEYWORDS = re.compile(
    r"\b(md&?a|management.?s discussion|management discussion|discuss|describe|explain|"
    r"what does.*say|narrative|evolve|evolution|how did)\b",
    re.IGNORECASE,
)
# Keywords that signal a cash flow / FCF query → trigger CASH FLOW STATEMENT section filter
_CASH_FLOW_KEYWORDS = re.compile(
    r"\b(free cash flow|fcf|operating cash flow|cash from operations|"
    r"capital expenditure|capex|cash flow statement|investing activities|"
    r"financing activities|purchases of property|net cash|"
    r"现金流|经营活动|资本支出)\b",
    re.IGNORECASE,
)
# Keywords that signal a "first mention / earliest appearance" query → chronological cross-doc search
_FIRST_MENTION_KEYWORDS = re.compile(
    r"\b(first mention|first time|earliest|when did|initially|first appear|"
    r"first disclose|first report|factory capacity|capacity bottleneck|production constraint|"
    r"manufacturing bottleneck|supply constraint|Gigafactory.{0,20}capacity|capacity.{0,20}limit|"
    r"capacity.{0,20}issue|capacity.{0,20}challenge|"
    r"首次|最早|第一次|产能瓶颈|产能限制|产能约束)\b",
    re.IGNORECASE,
)

# Chronological order of all known filing periods for first-mention sort
_PERIOD_ORDER: list[str] = [
    "Q1-2021", "Q2-2021", "Q3-2021", "FY2021",
    "Q1-2022", "Q2-2022", "Q3-2022", "FY2022",
    "Q1-2023", "Q2-2023", "Q3-2023", "FY2023",
    "Q1-2024", "Q2-2024", "Q3-2024", "FY2024",
    "Q1-2025", "Q2-2025", "Q3-2025", "FY2025",
]

SYSTEM_PROMPT = """You are a financial analyst assistant specializing in Tesla's SEC filings (10-K annual reports and 10-Q quarterly reports) from 2021 through 2025.

IMPORTANT — Document structure you must understand:
- Each year has THREE quarterly 10-Q filings: Q1, Q2, Q3. There is NO separate Q4 quarterly report.
- Each year has ONE annual 10-K filing (period label: FY2021, FY2022, etc.) that covers the full fiscal year including Q4 data.
- If a user asks about Q4 data, look for it in the annual 10-K for that year, not in a Q4 quarterly report.
- Available periods: Q1-2021, Q2-2021, Q3-2021, FY2021, Q1-2022, Q2-2022, Q3-2022, FY2022, ..., Q1-2025, Q2-2025, Q3-2025, FY2025.

When answering questions:
1. Base your answer STRICTLY on the provided context chunks.
2. Always cite the source (year, quarter/period, section, page) for specific data points.
3. For numerical comparisons or calculations, show your work step by step.
4. If the context is insufficient to fully answer the question, clearly state what is missing.
5. For trend analysis, organize data chronologically.
6. Be precise with numbers—include units (billions $, %, etc.).
7. When asked about full-year or Q4 figures, refer to the 10-K (FY{year}) not a Q4 10-Q.
"""

QUERY_ANALYSIS_PROMPT = """Analyze this financial question about Tesla's SEC filings and determine the following.

IMPORTANT document structure:
- Each year has Q1, Q2, Q3 quarterly 10-Q reports. There is NO Q4 quarterly report.
- Full-year / Q4 / annual data comes from the 10-K annual report (period: FY2021, FY2022, etc.).
- If the question mentions Q4, fourth quarter, or full-year figures, set doc_type to "annual_report".

Determine:
1. Is it SIMPLE (single doc, direct lookup) or COMPLEX (multi-doc, calculation, or text+table join)?
2. What years are relevant?
3. What quarters are relevant? Only include Q1/Q2/Q3. If Q4 or full-year is needed, leave quarters empty and set doc_type to "annual_report".
4. Does it require numerical calculation?
5. Does it require cross-referencing text and tables?

Question: {question}

Respond in this JSON format:
{{
  "complexity": "simple" or "complex",
  "years": ["2022", "2023"],
  "quarters": ["Q1", "Q2"],
  "doc_type": null or "annual_report" or "quarterly_report",
  "topics": ["automotive gross margin", "revenue"],
  "needs_calculation": true/false,
  "needs_text_table_join": true/false,
  "sub_queries": ["sub-question 1", "sub-question 2"]
}}"""


def _retrieve_fcf_periods(
    question: str,
    collection,
    bm25,
    chunks: list[dict],
    years: list[str],
    doc_type_filter: str | None,
    seen_ids: set,
) -> list[dict]:
    """
    For multi-year FCF queries: fetch top_k=2 CASH FLOW STATEMENT chunks for every
    (year, quarter) combination so the LLM has all YTD pairs needed for subtraction.
    Also fetches annual 10-K cash flow for full-year data.
    """
    added: list[dict] = []
    cf_query = "cash flows from operating activities capital expenditures net cash"
    for year in years:
        for quarter in ["Q1", "Q2", "Q3"]:
            results = hybrid_search(
                cf_query, collection, bm25, chunks,
                top_k=2,
                year_filter=[year],
                quarter_filter=[quarter],
                doc_type_filter="quarterly_report",
                section_filter="CASH FLOW STATEMENT",
            )
            for r in results:
                if r["chunk_id"] not in seen_ids:
                    added.append(r)
                    seen_ids.add(r["chunk_id"])
        # Annual 10-K for full-year cash flow
        annual_results = hybrid_search(
            cf_query, collection, bm25, chunks,
            top_k=2,
            year_filter=[year],
            doc_type_filter="annual_report",
            section_filter="CASH FLOW STATEMENT",
        )
        for r in annual_results:
            if r["chunk_id"] not in seen_ids:
                added.append(r)
                seen_ids.add(r["chunk_id"])
    return added


def _first_mention_search(
    question: str,
    collection,
    bm25,
    chunks: list[dict],
    seen_ids: set,
) -> list[dict]:
    """
    Cross-all-periods retrieval for 'first mention' queries.
    Searches MD&A, BUSINESS, and RISK FACTORS with NO year/quarter filter,
    then sorts results chronologically so the LLM sees the earliest matches first.
    """
    added: list[dict] = []
    for section in [
        "MANAGEMENT'S DISCUSSION AND ANALYSIS",
        "BUSINESS",
        "RISK FACTORS",
    ]:
        results = hybrid_search(
            question, collection, bm25, chunks,
            top_k=6,
            year_filter=None,
            quarter_filter=None,
            doc_type_filter=None,
            section_filter=section,
        )
        for r in results:
            if r["chunk_id"] not in seen_ids:
                added.append(r)
                seen_ids.add(r["chunk_id"])
    # Sort chronologically so earliest periods appear first
    period_rank = {p: i for i, p in enumerate(_PERIOD_ORDER)}
    added.sort(key=lambda c: period_rank.get(c.get("period", ""), 999))
    return added[:8]


def analyze_query(client: OpenAI, question: str, chat_model: str = "qwen-plus") -> dict[str, Any]:
    """Use LLM to analyze query complexity and extract search parameters."""
    try:
        resp = client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "user", "content": QUERY_ANALYSIS_PROMPT.format(question=question)}
            ],
            temperature=0,
        )
        import json
        content = resp.choices[0].message.content
        # Extract JSON from response (handles markdown code blocks)
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(content)
    except Exception:
        # Fallback: simple extraction
        filters = extract_query_filters(question)
        return {
            "complexity": "simple",
            "years": filters.get("years", []),
            "quarters": filters.get("quarters", []),
            "topics": [],
            "needs_calculation": False,
            "needs_text_table_join": False,
            "sub_queries": [question],
        }


def format_context(chunks: list[dict[str, Any]], max_chunks: int = 10) -> str:
    """Format retrieved chunks into context string for LLM."""
    context_parts = []
    for i, chunk in enumerate(chunks[:max_chunks]):
        period = chunk.get("period", "Unknown")
        section = chunk.get("section", "Unknown")
        page = chunk.get("page", "?")
        ctype = chunk.get("chunk_type", "text")
        source = f"[Source {i+1}: {period} | {section} | Page {page} | {ctype.upper()}]"
        context_parts.append(f"{source}\n{chunk['text']}")
    return "\n\n---\n\n".join(context_parts)


def answer_question(
    question: str,
    collection,
    bm25,
    chunks: list[dict],
    api_key: str,
    year_filter: list[str] | None = None,
    quarter_filter: list[str] | None = None,
    doc_type_filter: str | None = None,
    top_k: int = 8,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Main QA function. Returns answer dict with:
    - answer: str
    - sources: list of source citations
    - retrieved_chunks: list of chunk dicts
    - sub_queries: list of sub-queries used
    - debug_info: retrieval debug info (if debug=True)
    """
    base_url = os.environ.get("OPENAI_BASE_URL")
    chat_model = os.environ.get("CHAT_MODEL", "qwen-plus")
    client_kwargs: dict = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    # Step 1: Analyze query
    analysis = analyze_query(client, question, chat_model)
    is_complex = analysis.get("complexity") == "complex"
    sub_queries = analysis.get("sub_queries", [question])
    if not sub_queries:
        sub_queries = [question]

    # Merge explicit filters with auto-detected
    years = year_filter or analysis.get("years", [])
    quarters = quarter_filter or analysis.get("quarters", [])
    # Only apply annual_report filter from query analysis — never auto-restrict to
    # quarterly_report, as that would exclude Q4 data (which lives in 10-K annual reports)
    # for cross-period questions that span full years.
    if not doc_type_filter and analysis.get("doc_type") == "annual_report":
        doc_type_filter = "annual_report"

    # Step 2: Multi-hop retrieval for complex queries
    all_retrieved: list[dict] = []
    seen_ids = set()

    # Always retrieve for the main question
    main_results = hybrid_search(
        question, collection, bm25, chunks,
        top_k=top_k,
        year_filter=years or None,
        quarter_filter=quarters or None,
        doc_type_filter=doc_type_filter,
    )
    for r in main_results:
        if r["chunk_id"] not in seen_ids:
            all_retrieved.append(r)
            seen_ids.add(r["chunk_id"])

    # For complex queries: also retrieve for sub-queries
    if is_complex and len(sub_queries) > 1:
        for sub_q in sub_queries[1:]:  # skip first (same as main)
            sub_results = hybrid_search(
                sub_q, collection, bm25, chunks,
                top_k=4,
                year_filter=years or None,
                quarter_filter=quarters or None,
            )
            for r in sub_results:
                if r["chunk_id"] not in seen_ids:
                    all_retrieved.append(r)
                    seen_ids.add(r["chunk_id"])

    # For complex text+table queries, also search tables specifically
    if analysis.get("needs_text_table_join"):
        table_query = f"table financial data {question}"
        table_results = hybrid_search(
            table_query, collection, bm25, chunks,
            top_k=4,
            year_filter=years or None,
            quarter_filter=quarters or None,
        )
        for r in table_results:
            if r["chunk_id"] not in seen_ids and r.get("chunk_type") == "table":
                all_retrieved.append(r)
                seen_ids.add(r["chunk_id"])

    # Section-targeted retrieval for narrative queries about risk factors or MD&A.
    # Standard retrieval often returns boilerplate from FINANCIAL STATEMENTS or EXHIBITS
    # instead of the actual risk narrative. An extra targeted pass fixes this.
    if _RISK_KEYWORDS.search(question):
        risk_results = hybrid_search(
            question, collection, bm25, chunks,
            top_k=6,
            year_filter=years or None,
            quarter_filter=quarters or None,
            doc_type_filter=doc_type_filter,
            section_filter="RISK FACTORS",
        )
        for r in risk_results:
            if r["chunk_id"] not in seen_ids:
                all_retrieved.append(r)
                seen_ids.add(r["chunk_id"])
    elif _MDA_KEYWORDS.search(question):
        mda_results = hybrid_search(
            question, collection, bm25, chunks,
            top_k=4,
            year_filter=years or None,
            quarter_filter=quarters or None,
            doc_type_filter=doc_type_filter,
            section_filter="MANAGEMENT'S DISCUSSION AND ANALYSIS",
        )
        for r in mda_results:
            if r["chunk_id"] not in seen_ids:
                all_retrieved.append(r)
                seen_ids.add(r["chunk_id"])
    elif _CASH_FLOW_KEYWORDS.search(question):
        cf_results = hybrid_search(
            question, collection, bm25, chunks,
            top_k=6,
            year_filter=years or None,
            quarter_filter=quarters or None,
            doc_type_filter=doc_type_filter,
            section_filter="CASH FLOW STATEMENT",
        )
        for r in cf_results:
            if r["chunk_id"] not in seen_ids:
                all_retrieved.append(r)
                seen_ids.add(r["chunk_id"])

    # For multi-year FCF queries: explicitly fetch every (year × quarter) period
    # so the LLM has both Q(n) YTD and Q(n-1) YTD for standalone subtraction.
    _is_multiyear_cf = _CASH_FLOW_KEYWORDS.search(question) and len(years) > 1
    if _is_multiyear_cf:
        period_chunks = _retrieve_fcf_periods(
            question, collection, bm25, chunks,
            years=years,
            doc_type_filter=doc_type_filter,
            seen_ids=seen_ids,
        )
        all_retrieved.extend(period_chunks)

    # For first-mention queries: cross-document chronological search with no year filter
    if _FIRST_MENTION_KEYWORDS.search(question):
        first_mention_chunks = _first_mention_search(
            question, collection, bm25, chunks, seen_ids
        )
        all_retrieved.extend(first_mention_chunks)

    # Sort by RRF score
    all_retrieved.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)

    # For first-mention queries, re-sort chronologically so the LLM sees earliest periods first
    if _FIRST_MENTION_KEYWORDS.search(question):
        period_rank = {p: i for i, p in enumerate(_PERIOD_ORDER)}
        all_retrieved.sort(key=lambda c: period_rank.get(c.get("period", ""), 999))

    # Step 3: Build context and generate answer
    # Multi-year FCF needs more context slots to hold all period pairs
    max_chunks = 18 if _is_multiyear_cf else 12
    context = format_context(all_retrieved, max_chunks=max_chunks)

    # Build dynamic instructions based on query type
    extra_instructions = ""
    if _CASH_FLOW_KEYWORDS.search(question):
        extra_instructions = (
            "\n5. IMPORTANT for cash flow questions: Tesla's 10-Q quarterly filings report "
            "CUMULATIVE year-to-date (YTD) cash flows in the cash flow statement, NOT "
            "standalone single-quarter figures. Rules:\n"
            "   - Q1 standalone = Q1 filing value (correct as-is)\n"
            "   - Q2 standalone = Q2 filing YTD − Q1 filing YTD\n"
            "   - Q3 standalone = Q3 filing YTD − Q2 filing YTD\n"
            "   - Full year (Q4) = use the annual 10-K filing\n"
            "If you need a standalone quarter figure, retrieve BOTH the current and prior "
            "quarter filings, then subtract. Always show the YTD source values and the "
            "subtraction calculation explicitly in your answer."
        )
    if _FIRST_MENTION_KEYWORDS.search(question):
        extra_instructions += (
            "\n6. IMPORTANT for 'first mention' questions: The context chunks are ordered "
            "chronologically (earliest period first). Identify the earliest period chunk "
            "that contains the relevant concept. Cite the exact period label (e.g., Q1-2021 "
            "or FY2022) and state 'first mentioned in [period]'."
        )

    user_prompt = f"""Based on the following context from Tesla's financial reports, answer this question:

QUESTION: {question}

CONTEXT:
{context}

Please provide:
1. A direct answer with specific numbers and dates
2. Source citations for each data point
3. For calculations, show each step
4. Note any data gaps or limitations{extra_instructions}"""

    response = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=2000,
    )

    answer_text = response.choices[0].message.content

    # Build sources list
    sources = []
    for chunk in all_retrieved[:10]:
        sources.append({
            "period": chunk.get("period", ""),
            "section": chunk.get("section", ""),
            "page": chunk.get("page", ""),
            "chunk_type": chunk.get("chunk_type", ""),
            "rrf_score": chunk.get("rrf_score", 0),
            "preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
        })

    result = {
        "answer": answer_text,
        "sources": sources,
        "retrieved_chunks": all_retrieved,
        "sub_queries": sub_queries,
        "query_analysis": analysis,
    }

    if debug:
        result["debug_info"] = {
            "total_retrieved": len(all_retrieved),
            "years_used": years,
            "quarters_used": quarters,
            "is_complex": is_complex,
        }

    return result
