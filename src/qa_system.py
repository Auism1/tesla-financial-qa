"""
Tesla 财务报告的多步骤问答系统。
使用 OpenAI 聊天补全和检索到的上下文。
为复杂的跨文档问题实现两阶段流水线。
"""
import os
import re
from typing import Any

from openai import OpenAI

from src.retriever import hybrid_search, extract_query_filters

# 表示查询是关于风险因素叙述内容的关键词
_RISK_KEYWORDS = re.compile(
    r"\b(risk factor|risk disclosure|10-?K risk|item 1a|supply chain risk|market risk|"
    r"china risk|geopolit|regulatory risk|litigation risk|competition risk)\b",
    re.IGNORECASE,
)
# 表示 MD&A 叙述内容的关键词
_MDA_KEYWORDS = re.compile(
    r"\b(md&?a|management.?s discussion|management discussion|discuss|describe|explain|"
    r"what does.*say|narrative|evolve|evolution|how did)\b",
    re.IGNORECASE,
)
# 表示现金流 / FCF 查询的关键词 → 触发 CASH FLOW STATEMENT 章节过滤器
_CASH_FLOW_KEYWORDS = re.compile(
    r"\b(free cash flow|fcf|operating cash flow|cash from operations|"
    r"capital expenditure|capex|cash flow statement|investing activities|"
    r"financing activities|purchases of property|net cash|"
    r"现金流|经营活动|资本支出)\b",
    re.IGNORECASE,
)
# 表示"首次提及 / 最早出现"查询的关键词 → 按时间顺序跨文档搜索
_FIRST_MENTION_KEYWORDS = re.compile(
    r"\b(first mention|first time|earliest|when did|initially|first appear|"
    r"first disclose|first report|factory capacity|capacity bottleneck|production constraint|"
    r"manufacturing bottleneck|supply constraint|Gigafactory.{0,20}capacity|capacity.{0,20}limit|"
    r"capacity.{0,20}issue|capacity.{0,20}challenge|"
    r"首次|最早|第一次|产能瓶颈|产能限制|产能约束)\b",
    re.IGNORECASE,
)

# 所有已知申报期的时间顺序，用于首次提及排序
_PERIOD_ORDER: list[str] = [
    "Q1-2021", "Q2-2021", "Q3-2021", "FY2021",
    "Q1-2022", "Q2-2022", "Q3-2022", "FY2022",
    "Q1-2023", "Q2-2023", "Q3-2023", "FY2023",
    "Q1-2024", "Q2-2024", "Q3-2024", "FY2024",
    "Q1-2025", "Q2-2025", "Q3-2025", "FY2025",
]

SYSTEM_PROMPT = """你是一位专门研究 Tesla SEC 文件（2021 年至 2025 年的 10-K 年度报告和 10-Q 季度报告）的财务分析师助手。

重要 — 你必须理解的文档结构：
- 每年有三份季度 10-Q 文件：Q1、Q2、Q3。没有单独的 Q4 季度报告。
- 每年有一份年度 10-K 文件（期间标签：FY2021、FY2022 等），涵盖包括 Q4 数据在内的整个财年。
- 如果用户询问 Q4 数据，请在该年的年度 10-K 中查找，而不是在 Q4 季度报告中查找。
- 可用期间：Q1-2021、Q2-2021、Q3-2021、FY2021、Q1-2022、Q2-2022、Q3-2022、FY2022、...、Q1-2025、Q2-2025、Q3-2025、FY2025。

回答问题时：
1. 严格基于提供的上下文块回答。
2. 始终引用具体数据点的来源（年份、季度/期间、章节、页码）。
3. 对于数值比较或计算，逐步展示你的工作过程。
4. 如果上下文不足以完全回答问题，请明确说明缺少什么。
5. 对于趋势分析，按时间顺序组织数据。
6. 对数字要精确 — 包括单位（十亿美元、% 等）。
7. 当被问及全年或 Q4 数据时，请参考 10-K（FY{year}）而不是 Q4 10-Q。
"""

QUERY_ANALYSIS_PROMPT = """分析这个关于 Tesla SEC 文件的财务问题，并确定以下内容。

重要的文档结构：
- 每年有 Q1、Q2、Q3 季度 10-Q 报告。没有 Q4 季度报告。
- 全年 / Q4 / 年度数据来自 10-K 年度报告（期间：FY2021、FY2022 等）。
- 如果问题提到 Q4、第四季度或全年数据，请将 doc_type 设置为 "annual_report"。

确定：
1. 是简单（单个文档，直接查找）还是复杂（多文档、计算或文本+表格连接）？
2. 哪些年份相关？
3. 哪些季度相关？仅包括 Q1/Q2/Q3。如果需要 Q4 或全年数据，请将 quarters 留空并将 doc_type 设置为 "annual_report"。
4. 是否需要数值计算？
5. 是否需要交叉引用文本和表格？

问题：{question}

以此 JSON 格式响应：
{{
  "complexity": "simple" 或 "complex",
  "years": ["2022", "2023"],
  "quarters": ["Q1", "Q2"],
  "doc_type": null 或 "annual_report" 或 "quarterly_report",
  "topics": ["automotive gross margin", "revenue"],
  "needs_calculation": true/false,
  "needs_text_table_join": true/false,
  "sub_queries": ["子问题 1", "子问题 2"]
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
    对于多年 FCF 查询：为每个（年份、季度）组合获取 top_k=2 个 CASH FLOW STATEMENT 块，
    以便 LLM 拥有减法所需的所有 YTD 对。
    还获取年度 10-K 现金流以获取全年数据。
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
        # 年度 10-K 的全年现金流
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
    用于"首次提及"查询的跨所有期间检索。
    在 MD&A、BUSINESS 和 RISK FACTORS 中搜索，不使用年份/季度过滤器，
    然后按时间顺序对结果排序，以便 LLM 首先看到最早的匹配项。
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
    # 按时间顺序排序，以便最早的期间首先出现
    period_rank = {p: i for i, p in enumerate(_PERIOD_ORDER)}
    added.sort(key=lambda c: period_rank.get(c.get("period", ""), 999))
    return added[:8]


def analyze_query(client: OpenAI, question: str, chat_model: str = "qwen-plus") -> dict[str, Any]:
    """使用 LLM 分析查询复杂度并提取搜索参数。"""
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
        # 从响应中提取 JSON（处理 markdown 代码块）
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(content)
    except Exception:
        # 回退：简单提取
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
    """将检索到的块格式化为 LLM 的上下文字符串。"""
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
    主要的问答函数。返回包含以下内容的答案字典：
    - answer: str
    - sources: 来源引用列表
    - retrieved_chunks: 块字典列表
    - sub_queries: 使用的子查询列表
    - debug_info: 检索调试信息（如果 debug=True）
    """
    base_url = os.environ.get("OPENAI_BASE_URL")
    chat_model = os.environ.get("CHAT_MODEL", "qwen-plus")
    client_kwargs: dict = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    # 步骤 1：分析查询
    analysis = analyze_query(client, question, chat_model)
    is_complex = analysis.get("complexity") == "complex"
    sub_queries = analysis.get("sub_queries", [question])
    if not sub_queries:
        sub_queries = [question]

    # 将显式过滤器与自动检测的过滤器合并
    years = year_filter or analysis.get("years", [])
    quarters = quarter_filter or analysis.get("quarters", [])
    # 仅应用来自查询分析的 annual_report 过滤器 — 永远不要自动限制为
    # quarterly_report，因为这会排除 Q4 数据（位于 10-K 年度报告中）
    # 对于跨越整年的跨期间问题。
    if not doc_type_filter and analysis.get("doc_type") == "annual_report":
        doc_type_filter = "annual_report"

    # 步骤 2：复杂查询的多跳检索
    all_retrieved: list[dict] = []
    seen_ids = set()

    # 始终为主问题检索
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

    # 对于复杂查询：也为子查询检索
    if is_complex and len(sub_queries) > 1:
        for sub_q in sub_queries[1:]:  # 跳过第一个（与主查询相同）
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

    # 对于复杂的文本+表格查询，也专门搜索表格
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

    # 针对关于风险因素或 MD&A 的叙述性查询的章节定向检索。
    # 标准检索通常返回来自 FINANCIAL STATEMENTS 或 EXHIBITS 的样板文本
    # 而不是实际的风险叙述。额外的定向检索可以解决这个问题。
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

    # 对于多年 FCF 查询：显式获取每个（年份 × 季度）期间
    # 以便 LLM 拥有 Q(n) YTD 和 Q(n-1) YTD 进行独立减法。
    _is_multiyear_cf = _CASH_FLOW_KEYWORDS.search(question) and len(years) > 1
    if _is_multiyear_cf:
        period_chunks = _retrieve_fcf_periods(
            question, collection, bm25, chunks,
            years=years,
            doc_type_filter=doc_type_filter,
            seen_ids=seen_ids,
        )
        all_retrieved.extend(period_chunks)

    # 对于首次提及查询：无年份过滤器的跨文档时间顺序搜索
    if _FIRST_MENTION_KEYWORDS.search(question):
        first_mention_chunks = _first_mention_search(
            question, collection, bm25, chunks, seen_ids
        )
        all_retrieved.extend(first_mention_chunks)

    # 按 RRF 分数排序
    all_retrieved.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)

    # 对于首次提及查询，按时间顺序重新排序，以便 LLM 首先看到最早的期间
    if _FIRST_MENTION_KEYWORDS.search(question):
        period_rank = {p: i for i, p in enumerate(_PERIOD_ORDER)}
        all_retrieved.sort(key=lambda c: period_rank.get(c.get("period", ""), 999))

    # 步骤 3：构建上下文并生成答案
    # 多年 FCF 需要更多上下文槽来容纳所有期间对
    max_chunks = 18 if _is_multiyear_cf else 12
    context = format_context(all_retrieved, max_chunks=max_chunks)

    # 根据查询类型构建动态指令
    extra_instructions = ""
    if _CASH_FLOW_KEYWORDS.search(question):
        extra_instructions = (
            "\n5. 重要提示（现金流问题）：Tesla 的 10-Q 季度文件在现金流量表中报告"
            "累计年初至今（YTD）现金流，而不是独立的单季度数据。规则：\n"
            "   - Q1 独立 = Q1 文件值（正确）\n"
            "   - Q2 独立 = Q2 文件 YTD − Q1 文件 YTD\n"
            "   - Q3 独立 = Q3 文件 YTD − Q2 文件 YTD\n"
            "   - 全年（Q4）= 使用年度 10-K 文件\n"
            "如果你需要独立季度数据，请检索当前和前一季度的文件，然后相减。始终在答案中"
            "明确显示 YTD 源值和减法计算。"
        )
    if _FIRST_MENTION_KEYWORDS.search(question):
        extra_instructions += (
            "\n6. 重要提示（'首次提及'问题）：上下文块按时间顺序排列"
            "（最早期间在前）。识别包含相关概念的最早期间块。引用确切的期间标签（例如 Q1-2021 "
            "或 FY2022）并说明'首次提及于 [期间]'。"
        )

    user_prompt = f"""根据以下来自 Tesla 财务报告的上下文，回答这个问题：

问题：{question}

上下文：
{context}

请提供：
1. 包含具体数字和日期的直接答案
2. 每个数据点的来源引用
3. 对于计算，展示每个步骤
4. 注明任何数据缺口或限制{extra_instructions}"""

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

    # 构建来源列表
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
