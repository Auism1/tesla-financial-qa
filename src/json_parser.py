"""
Tesla SEC 文件（10-K 和 10-Q）的 MinerU JSON 解析器。
使用文本缓冲区 + 标题边界方法进行精确分块，
并通过上下文感知的章节检测来防止误分类。
"""
import json
import re
from pathlib import Path
from typing import Any

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# ── 章节检测 ──────────────────────────────────────────────────────────

SECTION_MAP: dict[str, str] = {
    # 部分标题
    "PARTI": "PART I - FINANCIAL INFORMATION",
    "PARTII": "PART II - OTHER INFORMATION",
    "PARTIII": "PART III",
    "PARTIV": "PART IV",
    "PARTIFINANCIALINFORMATION": "PART I - FINANCIAL INFORMATION",
    "PARTIIOTHERINFORMATION": "PART II - OTHER INFORMATION",
    # 财务报表
    "FINANCIALSTATEMENTS": "FINANCIAL STATEMENTS",
    "CONSOLIDATEDBALANCESHEETS": "BALANCE SHEET",
    "CONSOLIDATEDBALANCESHEET": "BALANCE SHEET",
    "CONSOLIDATEDSTATEMENTSOFOPERATIONS": "INCOME STATEMENT",
    "CONSOLIDATEDSTATEMENTSOFCOMPREHENSIVEINCOME": "INCOME STATEMENT",
    "CONSOLIDATEDSTATEMENTSOFCASHFLOWS": "CASH FLOW STATEMENT",
    "CONSOLIDATEDSTATEMENTOFCASHFLOWS": "CASH FLOW STATEMENT",
    "BALANCESHEETS": "BALANCE SHEET",
    "BALANCESHEET": "BALANCE SHEET",
    "STATEMENTSOFOPERATIONS": "INCOME STATEMENT",
    "STATEMENTOFOPERATIONS": "INCOME STATEMENT",
    "STATEMENTSOFCASHFLOWS": "CASH FLOW STATEMENT",
    "STATEMENTOFCASHFLOWS": "CASH FLOW STATEMENT",
    "INCOMESTATEMENT": "INCOME STATEMENT",
    "CONSOLIDATEDSTATEMENTSOFSTOCKHOLDERSEQUITY": "EQUITY STATEMENT",
    "CONSOLIDATEDSTATEMENTSOFEQUITY": "EQUITY STATEMENT",
    # 附注
    "NOTESTOCONSOLIDATEDFINANCIALSTATEMENTS": "NOTES TO FINANCIAL STATEMENTS",
    "NOTESTOCONDENSEDCONSOLIDATEDFINANCIALSTATEMENTS": "NOTES TO FINANCIAL STATEMENTS",
    # MD&A 和运营结果
    "MANAGEMENTSDISCUSSIONANDANALYSISOFFINANCIALCONDITIONANDRESULTSOFOPERATIONS": "MANAGEMENT'S DISCUSSION AND ANALYSIS",
    "MANAGEMENTSDISCUSSIONANDANALYSIS": "MANAGEMENT'S DISCUSSION AND ANALYSIS",
    "RESULTSOFOPERATIONS": "RESULTS OF OPERATIONS",
    "LIQUIDITYANDCAPITALRESOURCES": "LIQUIDITY AND CAPITAL RESOURCES",
    "CRITICALACCOUNTINGPOLICIESANDESTIMATES": "CRITICAL ACCOUNTING ESTIMATES",
    "CRITICALACCOUNTINGESTIMATES": "CRITICAL ACCOUNTING ESTIMATES",
    "QUANTITATIVEANDQUALITATIVEDISCLOSURESABOUTMARKETRISK": "MARKET RISK DISCLOSURES",
    "QUANTITATIVEANDQUALITATIVEDISCLOSURES": "MARKET RISK DISCLOSURES",
    # 风险和法律
    "RISKFACTORS": "RISK FACTORS",
    "LEGALPROCEEDINGS": "LEGAL PROCEEDINGS",
    # 10-K 特定章节
    "BUSINESS": "BUSINESS",
    "PROPERTIES": "PROPERTIES",
    "EXECUTIVECOMPENSATION": "EXECUTIVE COMPENSATION",
    "SELECTEDFINANCIALDATA": "SELECTED FINANCIAL DATA",
    "PRINCIPALACCOUNTANTFEESANDSERVICES": "PRINCIPAL ACCOUNTANT FEES",
    "CHANGESINANDISAGREEMENTSWITHACCOUNTANTS": "ACCOUNTING CHANGES",
    "SECURITYOWNERSHIPOFCERTAINBENEFICIALOWNERSANDMANAGEMENT": "SECURITY OWNERSHIP",
    # 业务分部（仅在附注外作为顶级章节有效）
    "AUTOMOTIVESEGMENT": "AUTOMOTIVE SEGMENT",
    "AUTOMOTIVESEGMENTREVENUES": "AUTOMOTIVE SEGMENT",
    "ENERGYGENERATIONANDSTORAGE": "ENERGY SEGMENT",
    "ENERGYGENERATIONANDSTORAGESSEGMENT": "ENERGY SEGMENT",
    "SERVICESANDOTHER": "SERVICES AND OTHER SEGMENT",
    "AUTOMOTIVESERVICESANDOTHERSEGMENT": "SERVICES AND OTHER SEGMENT",
    # 旧版季度更新章节
    "HIGHLIGHTS": "HIGHLIGHTS",
    "SUMMARY": "SUMMARY",
    "FINANCIALSUMMARY": "FINANCIAL SUMMARY",
    "OPERATIONALSUMMARY": "OPERATIONAL SUMMARY",
    "KEYMETRICS": "KEY METRICS",
    "ADDITIONALINFORMATION": "ADDITIONAL INFORMATION",
    "OUTLOOK": "OUTLOOK",
}

_SECTION_KEYWORDS: list[tuple[str, str]] = [
    ("MANAGEMENTSDISCUSSION", "MANAGEMENT'S DISCUSSION AND ANALYSIS"),
    ("RESULTOFOPERATION", "RESULTS OF OPERATIONS"),
    ("LIQUIDITYANDCAPITAL", "LIQUIDITY AND CAPITAL RESOURCES"),
    ("CRITICALACCOUNTING", "CRITICAL ACCOUNTING ESTIMATES"),
    ("QUANTITATIVEANDQUALITATIVE", "MARKET RISK DISCLOSURES"),
    ("RISKFACTOR", "RISK FACTORS"),
    ("LEGALPROCEEDING", "LEGAL PROCEEDINGS"),
    ("NOTESTOCONSOLIDATED", "NOTES TO FINANCIAL STATEMENTS"),
    ("NOTESTOCONDENSED", "NOTES TO FINANCIAL STATEMENTS"),
    ("CONSOLIDATEDBALANCE", "BALANCE SHEET"),
    ("CONSOLIDATEDSTATEMENT", "INCOME STATEMENT"),
    ("STATEMENTOFCASHFLOW", "CASH FLOW STATEMENT"),
    ("STATEMENTSOFCASHFLOW", "CASH FLOW STATEMENT"),
    ("AUTOMOTIVESEGMENT", "AUTOMOTIVE SEGMENT"),
    ("ENERGYSEGMENT", "ENERGY SEGMENT"),
    ("ENERGYGENERATION", "ENERGY SEGMENT"),
    ("SERVICESANDOTHER", "SERVICES AND OTHER SEGMENT"),
    ("FINANCIALSTATEMENT", "FINANCIAL STATEMENTS"),
]

# Item 编号 → Part I 章节（10-Q / 10-K Part I）
_ITEM_MAP_PART_I: dict[str, str] = {
    "1": "FINANCIAL STATEMENTS",
    "1A": "RISK FACTORS",
    "1B": "LEGAL PROCEEDINGS",
    "2": "MANAGEMENT'S DISCUSSION AND ANALYSIS",
    "3": "MARKET RISK DISCLOSURES",
    "4": "CONTROLS AND PROCEDURES",
    "5": "OTHER INFORMATION",
    "6": "EXHIBITS",
}

# Item 编号 → Part II 章节（10-Q Part II 有不同的 Item 含义）
_ITEM_MAP_PART_II: dict[str, str] = {
    "1": "LEGAL PROCEEDINGS",
    "1A": "RISK FACTORS",
    "2": "OTHER INFORMATION",
    "3": "OTHER INFORMATION",
    "4": "OTHER INFORMATION",
    "5": "OTHER INFORMATION",
    "6": "EXHIBITS",
}

# 在附注内部作为子主题出现的分部章节。
# 当 in_notes=True 时，这些不应触发章节变更。
_NOTES_INTERNAL_SECTIONS = {
    "AUTOMOTIVE SEGMENT",
    "ENERGY SEGMENT",
    "SERVICES AND OTHER SEGMENT",
}


def _normalise(text: str) -> str:
    return re.sub(r"[\s\-_./,;:\'\"&()]", "", text.upper())


def detect_section(text: str, current_part: str = "PART_I") -> str | None:
    """
    从标题字符串检测章节。
    使用 current_part 选择正确的 Item 编号映射。
    返回规范的章节名称或 None。
    """
    if not text or len(text.strip()) < 2:
        return None

    norm = _normalise(text)

    if norm in SECTION_MAP:
        return SECTION_MAP[norm]

    # Item 编号匹配 — 使用 Part 感知的映射
    item_match = re.match(r"^ITEM\s*([0-9]+[A-Z]?)[.\s]", text.strip(), re.IGNORECASE)
    if item_match:
        item_num = item_match.group(1).upper()
        item_map = _ITEM_MAP_PART_II if current_part == "PART_II" else _ITEM_MAP_PART_I
        if item_num in item_map:
            return item_map[item_num]

    for keyword, section in _SECTION_KEYWORDS:
        if keyword in norm:
            return section

    return None


# ── 文件名解析 ───────────────────────────────────────────────────────────

def parse_filename(path: Path) -> dict[str, str]:
    """从 Tesla-Q1-2021.json 或 Tesla-10K-2022.json 提取年份、季度、文档类型。"""
    name = path.stem
    m = re.match(r"Tesla-(Q[1-4]|10K)-(\d{4})", name, re.IGNORECASE)
    if not m:
        return {"filename": path.name, "year": "", "quarter": "", "doc_type": "unknown", "period": "UNKNOWN"}
    qtype = m.group(1).upper()
    year = m.group(2)
    if qtype == "10K":
        return {"filename": path.name, "year": year, "quarter": "10K", "doc_type": "annual_report", "period": f"FY{year}"}
    return {"filename": path.name, "year": year, "quarter": qtype, "doc_type": "quarterly_report", "period": f"{qtype}-{year}"}


# ── 文本 / 表格提取 ────────────────────────────────────────────────────

def _normalize_financial_numbers(text: str) -> str:
    # 修复 "$ 464" 或 "$   1,641" — 美元符号与数字之间被 PDF 对齐空格分隔
    text = re.sub(r"\$\s{1,20}(\d)", r"$\1", text)
    text = re.sub(r"\\\$\s*", "$", text)
    text = re.sub(r"\$(\d+)\s+(\d{2,3})\b", lambda m: f"${m.group(1)}.{m.group(2)}", text)
    text = re.sub(r"\$(\d{1,3})\s+(\d{3})\b", lambda m: f"${m.group(1)},{m.group(2)}", text)
    return text


def extract_text_from_block(block: dict) -> str:
    """递归地从任何块中提取纯文本（lines→spans + 嵌套块）。"""
    parts: list[str] = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            content = span.get("content", "").strip()
            if content:
                parts.append(content)
    for sub in block.get("blocks", []):
        t = extract_text_from_block(sub)
        if t:
            parts.append(t)
    return " ".join(parts)


def html_table_to_markdown(html: str) -> str:
    if HAS_BS4:
        return _bs4_table_to_md(html)
    return _regex_table_to_md(html)


def _bs4_table_to_md(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    rows_data: list[list[str]] = []
    for tr in soup.find_all("tr"):
        cells = []
        for td in tr.find_all(["td", "th"]):
            text = td.get_text(separator=" ").strip().replace("|", "/")
            span = int(td.get("colspan", 1))
            cells.append(text)
            for _ in range(span - 1):
                cells.append("")
        if any(c for c in cells):
            rows_data.append(cells)
    if not rows_data:
        return ""
    max_cols = max(len(r) for r in rows_data)
    for r in rows_data:
        while len(r) < max_cols:
            r.append("")
    header = rows_data[0]
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * max_cols) + " |\n"
    for row in rows_data[1:]:
        if any(c.strip() for c in row):
            md += "| " + " | ".join(row) + " |\n"
    return md


def _regex_table_to_md(html: str) -> str:
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE)
    rows_data = []
    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL | re.IGNORECASE)
        cleaned = [re.sub(r"<[^>]+>", "", c).strip().replace("|", "/") for c in cells]
        if any(cleaned):
            rows_data.append(cleaned)
    if not rows_data:
        return ""
    max_cols = max(len(r) for r in rows_data)
    for r in rows_data:
        while len(r) < max_cols:
            r.append("")
    header = rows_data[0]
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * max_cols) + " |\n"
    for row in rows_data[1:]:
        if any(c.strip() for c in row):
            md += "| " + " | ".join(row) + " |\n"
    return md


def _infer_section_from_table(content: str) -> str | None:
    """
    当表格前没有标题时，从表格内容推断财务报表类型。
    检查标识报表类型的特征行标签。
    """
    upper = content.upper()
    if "CASH FLOWS FROM OPERATING ACTIVITIES" in upper or "NET CASH PROVIDED BY" in upper:
        return "CASH FLOW STATEMENT"
    if "TOTAL ASSETS" in upper and ("TOTAL LIABILITIES" in upper or "STOCKHOLDERS" in upper):
        return "BALANCE SHEET"
    if "NET INCOME" in upper and "EARNINGS PER SHARE" in upper:
        return "INCOME STATEMENT"
    return None


def extract_table_content(block: dict) -> str:
    """
    提取表格内容。优先级：
    1. 顶层 markdown 字段（未来的 MinerU 版本）
    2. 顶层 html 字段
    3. 嵌套的 span html（当前 MinerU 结构）
    4. 回退到纯文本提取
    """
    if block.get("markdown"):
        return block["markdown"]
    if block.get("html"):
        return html_table_to_markdown(block["html"])
    # 当前 MinerU：html 位于 blocks → lines → spans 内部
    for inner in block.get("blocks", []):
        for line in inner.get("lines", []):
            for span in line.get("spans", []):
                if span.get("html"):
                    return html_table_to_markdown(span["html"])
    return extract_text_from_block(block)


# ── 主解析器 ────────────────────────────────────────────────────────────────

_MIN_TEXT_LEN = 20


def parse_json(json_path: str | Path) -> list[dict[str, Any]]:
    """
    使用文本缓冲区 + 标题边界方法将 MinerU JSON 文件解析为原始块。

    关键行为：
    - 文本在当前标题/章节下累积到缓冲区中
    - 遇到新标题或表格时刷新缓冲区
    - 章节检测具有上下文感知：附注内的分部标题不会
      逃离附注章节上下文
    - Part I / Part II 状态控制正确的 Item 编号 → 章节映射
    - Note N 子章节在 'subsection' 字段中跟踪
    """
    json_path = Path(json_path)
    meta = parse_filename(json_path)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    pages: list[dict] = data.get("pdf_info", [])
    chunks: list[dict[str, Any]] = []

    # 解析器状态
    current_section = "OVERVIEW"
    current_subsection = ""   # 例如 "Note 12: Commitments and Contingencies"
    current_part = "PART_I"   # "PART_I" 或 "PART_II"
    in_notes = False           # 在财务报表附注内时为 True
    text_buffer: list[str] = []
    last_page = 1

    def flush_buffer(page_num: int) -> None:
        """将累积的文本缓冲区刷新为单个块。"""
        if not text_buffer:
            return
        combined = _normalize_financial_numbers(" ".join(text_buffer).strip())
        if len(combined) >= _MIN_TEXT_LEN:
            chunks.append({
                "text": combined,
                "chunk_type": "text",
                "section": current_section,
                "subsection": current_subsection,
                "page": page_num,
                **meta,
            })
        text_buffer.clear()

    for page in pages:
        page_num: int = page.get("page_idx", 0) + 1
        last_page = page_num

        for block in page.get("para_blocks", []):
            btype = block.get("type", "")

            # ── 标题：章节边界 ───────────────────────────────────────
            if btype == "title":
                flush_buffer(page_num)
                title_text = extract_text_from_block(block).strip()
                if not title_text:
                    continue

                # 超过 200 字符的标题可能是误分类的正文文本
                if len(title_text) >= 200:
                    text_buffer.append(title_text)
                    continue

                section = detect_section(title_text, current_part)

                if section:
                    # 跟踪 Part 转换
                    if "PART I" in section and "PART II" not in section:
                        current_part = "PART_I"
                        in_notes = False
                    elif "PART II" in section:
                        current_part = "PART_II"
                        in_notes = False

                    if section == "NOTES TO FINANCIAL STATEMENTS":
                        in_notes = True
                        current_section = section
                        current_subsection = ""
                    elif in_notes and section in _NOTES_INTERNAL_SECTIONS:
                        # 此分部标题是附注内的子主题 — 不要
                        # 更改主章节，只更新子章节标签
                        current_subsection = section
                    else:
                        in_notes = False
                        current_section = section
                        current_subsection = ""
                else:
                    # 检查 "Note N – Title" 模式
                    note_m = re.match(r"^Note\s+(\d+)\s*[–\-]\s*(.+)", title_text, re.IGNORECASE)
                    if note_m and in_notes:
                        current_subsection = f"Note {note_m.group(1)}: {note_m.group(2)[:60].strip()}"
                    elif len(title_text) >= _MIN_TEXT_LEN:
                        # 非章节标题 → 视为文本内容
                        text_buffer.append(title_text)

            # ── 文本、列表、方程 → 缓冲区 ───────────────────────────────
            elif btype in ("text", "list", "inline_equation", "interline_equation"):
                t = extract_text_from_block(block).strip()
                if t:
                    text_buffer.append(t)

            # ── ref_text（10-K 中的法律引用）→ 缓冲区 ───────────────────────
            elif btype == "ref_text":
                t = extract_text_from_block(block).strip()
                if t:
                    text_buffer.append(t)

            # ── 表格标题 / 脚注 → 缓冲区（表格的上下文）─────────
            elif btype in ("table_caption", "table_footnote"):
                t = extract_text_from_block(block).strip()
                if t:
                    text_buffer.append(t)

            # ── 表格 → 先刷新缓冲区，然后独立块 ─────────────
            elif btype in ("table", "table_body"):
                flush_buffer(page_num)
                content = extract_table_content(block)
                content = _normalize_financial_numbers(content)
                if content and len(content.strip()) >= _MIN_TEXT_LEN:
                    # 如果当前章节对于此表格的内容似乎不正确，
                    # 从表格自己的行标签推断正确的章节。
                    table_section = current_section
                    inferred = _infer_section_from_table(content)
                    if inferred and inferred != current_section:
                        # 仅在我们处于相邻的财务报表章节时覆盖
                        # （防止覆盖有意义的上下文，如附注或 MD&A）
                        _financial_stmt_sections = {
                            "INCOME STATEMENT", "BALANCE SHEET", "CASH FLOW STATEMENT",
                            "EQUITY STATEMENT", "FINANCIAL STATEMENTS", "OVERVIEW",
                            "PART I - FINANCIAL INFORMATION",
                        }
                        if current_section in _financial_stmt_sections:
                            table_section = inferred
                    label = current_subsection or table_section
                    chunks.append({
                        "text": f"[TABLE: {label[:80]}]\n{content}",
                        "chunk_type": "table",
                        "section": table_section,
                        "subsection": current_subsection,
                        "page": page_num,
                        **meta,
                    })

            # header, page_number, image, aside_text → 静默跳过

    # 在文档末尾刷新任何剩余的缓冲区
    flush_buffer(last_page)

    return chunks


def parse_all_jsons(dataset_dir: str | Path) -> list[dict[str, Any]]:
    """递归解析数据集目录中的所有 .json 文件。"""
    dataset_dir = Path(dataset_dir)
    all_chunks: list[dict[str, Any]] = []
    json_files = sorted(dataset_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    for jf in json_files:
        print(f"  Parsing {jf.name[:50]}...", end=" ", flush=True)
        try:
            chunks = parse_json(jf)
            print(f"→ {len(chunks)} chunks")
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"ERROR: {e}")
    return all_chunks
