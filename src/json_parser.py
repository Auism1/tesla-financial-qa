"""
MinerU JSON parser for Tesla SEC filings (10-K and 10-Q).
Uses a text-buffer + title-boundary approach for accurate chunking,
with context-aware section detection to prevent misclassification.
"""
import html as html_lib
import json
import re
from pathlib import Path
from typing import Any

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# ── Section detection ──────────────────────────────────────────────────────────

SECTION_MAP: dict[str, str] = {
    # Part headers
    "PARTI": "PART I - FINANCIAL INFORMATION",
    "PARTII": "PART II - OTHER INFORMATION",
    "PARTIII": "PART III",
    "PARTIV": "PART IV",
    "PARTIFINANCIALINFORMATION": "PART I - FINANCIAL INFORMATION",
    "PARTIIOTHERINFORMATION": "PART II - OTHER INFORMATION",
    # Financial statements
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
    # Notes
    "NOTESTOCONSOLIDATEDFINANCIALSTATEMENTS": "NOTES TO FINANCIAL STATEMENTS",
    "NOTESTOCONDENSEDCONSOLIDATEDFINANCIALSTATEMENTS": "NOTES TO FINANCIAL STATEMENTS",
    # MD&A and results
    "MANAGEMENTSDISCUSSIONANDANALYSISOFFINANCIALCONDITIONANDRESULTSOFOPERATIONS": "MANAGEMENT'S DISCUSSION AND ANALYSIS",
    "MANAGEMENTSDISCUSSIONANDANALYSIS": "MANAGEMENT'S DISCUSSION AND ANALYSIS",
    "RESULTSOFOPERATIONS": "RESULTS OF OPERATIONS",
    "LIQUIDITYANDCAPITALRESOURCES": "LIQUIDITY AND CAPITAL RESOURCES",
    "CRITICALACCOUNTINGPOLICIESANDESTIMATES": "CRITICAL ACCOUNTING ESTIMATES",
    "CRITICALACCOUNTINGESTIMATES": "CRITICAL ACCOUNTING ESTIMATES",
    "QUANTITATIVEANDQUALITATIVEDISCLOSURESABOUTMARKETRISK": "MARKET RISK DISCLOSURES",
    "QUANTITATIVEANDQUALITATIVEDISCLOSURES": "MARKET RISK DISCLOSURES",
    # Risk & legal
    "RISKFACTORS": "RISK FACTORS",
    "LEGALPROCEEDINGS": "LEGAL PROCEEDINGS",
    # 10-K specific
    "BUSINESS": "BUSINESS",
    "PROPERTIES": "PROPERTIES",
    "EXECUTIVECOMPENSATION": "EXECUTIVE COMPENSATION",
    "SELECTEDFINANCIALDATA": "SELECTED FINANCIAL DATA",
    "PRINCIPALACCOUNTANTFEESANDSERVICES": "PRINCIPAL ACCOUNTANT FEES",
    "CHANGESINANDISAGREEMENTSWITHACCOUNTANTS": "ACCOUNTING CHANGES",
    "SECURITYOWNERSHIPOFCERTAINBENEFICIALOWNERSANDMANAGEMENT": "SECURITY OWNERSHIP",
    # Business segments (only valid as top-level outside Notes)
    "AUTOMOTIVESEGMENT": "AUTOMOTIVE SEGMENT",
    "AUTOMOTIVESEGMENTREVENUES": "AUTOMOTIVE SEGMENT",
    "ENERGYGENERATIONANDSTORAGE": "ENERGY SEGMENT",
    "ENERGYGENERATIONANDSTORAGESSEGMENT": "ENERGY SEGMENT",
    "SERVICESANDOTHER": "SERVICES AND OTHER SEGMENT",
    "AUTOMOTIVESERVICESANDOTHERSEGMENT": "SERVICES AND OTHER SEGMENT",
    # Legacy quarterly update sections
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

# Item number → section for Part I (10-Q / 10-K Part I)
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

# Item number → section for Part II (10-Q Part II has different Item meanings)
_ITEM_MAP_PART_II: dict[str, str] = {
    "1": "LEGAL PROCEEDINGS",
    "1A": "RISK FACTORS",
    "2": "OTHER INFORMATION",
    "3": "OTHER INFORMATION",
    "4": "OTHER INFORMATION",
    "5": "OTHER INFORMATION",
    "6": "EXHIBITS",
}

# Segment sections that appear INSIDE Notes as sub-topics.
# When in_notes=True, these should NOT trigger a section change.
_NOTES_INTERNAL_SECTIONS = {
    "AUTOMOTIVE SEGMENT",
    "ENERGY SEGMENT",
    "SERVICES AND OTHER SEGMENT",
}


def _normalise(text: str) -> str:
    return re.sub(r"[\s\-_./,;:\'\"&()]", "", text.upper())


def detect_section(text: str, current_part: str = "PART_I") -> str | None:
    """
    Detect section from a title string.
    Uses current_part to pick the correct Item number map.
    Returns canonical section name or None.
    """
    if not text or len(text.strip()) < 2:
        return None

    norm = _normalise(text)

    if norm in SECTION_MAP:
        return SECTION_MAP[norm]

    # Item number match — use Part-aware map
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


# ── Filename parsing ───────────────────────────────────────────────────────────

def parse_filename(path: Path) -> dict[str, str]:
    """Extract year, quarter, doc_type from Tesla-Q1-2021.json or Tesla-10K-2022.json."""
    name = path.stem
    m = re.match(r"Tesla-(Q[1-4]|10K)-(\d{4})", name, re.IGNORECASE)
    if not m:
        return {"filename": path.name, "year": "", "quarter": "", "doc_type": "unknown", "period": "UNKNOWN"}
    qtype = m.group(1).upper()
    year = m.group(2)
    if qtype == "10K":
        return {"filename": path.name, "year": year, "quarter": "10K", "doc_type": "annual_report", "period": f"FY{year}"}
    return {"filename": path.name, "year": year, "quarter": qtype, "doc_type": "quarterly_report", "period": f"{qtype}-{year}"}


# ── Text / table extraction ────────────────────────────────────────────────────

def _normalize_financial_numbers(text: str) -> str:
    # Fix "$ 464" or "$   1,641" — dollar sign separated from number by PDF alignment spaces
    text = re.sub(r"\$\s{1,20}(\d)", r"$\1", text)
    text = re.sub(r"\\\$\s*", "$", text)
    text = re.sub(r"\$(\d+)\s+(\d{2,3})\b", lambda m: f"${m.group(1)}.{m.group(2)}", text)
    text = re.sub(r"\$(\d{1,3})\s+(\d{3})\b", lambda m: f"${m.group(1)},{m.group(2)}", text)
    return text


def extract_text_from_block(block: dict) -> str:
    """Recursively extract plain text from any block (lines→spans + nested blocks)."""
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
    Infer the financial statement type from table content when no title precedes the table.
    Checks for characteristic row labels that identify the statement type.
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
    Extract table content. Priority:
    1. Top-level markdown field (future MinerU versions)
    2. Top-level html field
    3. Nested span html (current MinerU structure)
    4. Fallback to plain text extraction
    """
    if block.get("markdown"):
        return block["markdown"]
    if block.get("html"):
        return html_table_to_markdown(block["html"])
    # Current MinerU: html lives inside blocks → lines → spans
    for inner in block.get("blocks", []):
        for line in inner.get("lines", []):
            for span in line.get("spans", []):
                if span.get("html"):
                    return html_table_to_markdown(span["html"])
    return extract_text_from_block(block)


# ── Main parser ────────────────────────────────────────────────────────────────

_MIN_TEXT_LEN = 20


def parse_json(json_path: str | Path) -> list[dict[str, Any]]:
    """
    Parse a MinerU JSON file into raw chunks using text-buffer + title-boundary approach.

    Key behaviors:
    - Text accumulates in a buffer under the current title/section
    - Buffer is flushed when a new title or table is encountered
    - Section detection is context-aware: segment titles inside Notes do not
      escape the Notes section context
    - Part I / Part II state controls correct Item number → section mapping
    - Note N subsections are tracked in the 'subsection' field
    """
    json_path = Path(json_path)
    meta = parse_filename(json_path)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    pages: list[dict] = data.get("pdf_info", [])
    chunks: list[dict[str, Any]] = []

    # Parser state
    current_section = "OVERVIEW"
    current_subsection = ""   # e.g. "Note 12: Commitments and Contingencies"
    current_part = "PART_I"   # "PART_I" or "PART_II"
    in_notes = False           # True when inside Notes to Financial Statements
    text_buffer: list[str] = []
    last_page = 1

    def flush_buffer(page_num: int) -> None:
        """Flush accumulated text buffer as a single chunk."""
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

            # ── Title: section boundary ───────────────────────────────────────
            if btype == "title":
                flush_buffer(page_num)
                title_text = extract_text_from_block(block).strip()
                if not title_text:
                    continue

                # Titles longer than 200 chars are likely mis-classified body text
                if len(title_text) >= 200:
                    text_buffer.append(title_text)
                    continue

                section = detect_section(title_text, current_part)

                if section:
                    # Track Part transitions
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
                        # This segment title is a sub-topic within Notes — do not
                        # change the main section, just update subsection label
                        current_subsection = section
                    else:
                        in_notes = False
                        current_section = section
                        current_subsection = ""
                else:
                    # Check for "Note N – Title" pattern
                    note_m = re.match(r"^Note\s+(\d+)\s*[–\-]\s*(.+)", title_text, re.IGNORECASE)
                    if note_m and in_notes:
                        current_subsection = f"Note {note_m.group(1)}: {note_m.group(2)[:60].strip()}"
                    elif len(title_text) >= _MIN_TEXT_LEN:
                        # Non-section title → treat as text content
                        text_buffer.append(title_text)

            # ── Text, list, equations → buffer ───────────────────────────────
            elif btype in ("text", "list", "inline_equation", "interline_equation"):
                t = extract_text_from_block(block).strip()
                if t:
                    text_buffer.append(t)

            # ── ref_text (legal refs in 10-K) → buffer ───────────────────────
            elif btype == "ref_text":
                t = extract_text_from_block(block).strip()
                if t:
                    text_buffer.append(t)

            # ── Table caption / footnote → buffer (context for table) ─────────
            elif btype in ("table_caption", "table_footnote"):
                t = extract_text_from_block(block).strip()
                if t:
                    text_buffer.append(t)

            # ── Table → flush buffer first, then standalone chunk ─────────────
            elif btype in ("table", "table_body"):
                flush_buffer(page_num)
                content = extract_table_content(block)
                content = _normalize_financial_numbers(content)
                if content and len(content.strip()) >= _MIN_TEXT_LEN:
                    # If the current section seems wrong for this table's content,
                    # infer the correct section from the table's own row labels.
                    table_section = current_section
                    inferred = _infer_section_from_table(content)
                    if inferred and inferred != current_section:
                        # Only override if we're in an adjacent financial-statement section
                        # (prevents overriding meaningful context like Notes or MD&A)
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

            # header, page_number, image, aside_text → skip silently

    # Flush any remaining buffer at end of document
    flush_buffer(last_page)

    return chunks


def parse_all_jsons(dataset_dir: str | Path) -> list[dict[str, Any]]:
    """Recursively parse all .json files in the dataset directory."""
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


# ── Markdown parser ────────────────────────────────────────────────────────────

def parse_md(md_path: str | Path) -> list[dict[str, Any]]:
    """
    Parse a Markdown file (MinerU-exported) into raw chunks.

    Markdown format specifics:
    - All section headers use H1 (#) exclusively
    - Tables are inline HTML (<table>...</table>), possibly multi-line
    - Financial numbers may contain PDF spacing: "$     17,141"
    - HTML entities (&#x27; etc.) appear in text
    """
    md_path = Path(md_path)
    meta = parse_filename(md_path)

    lines = md_path.read_text(encoding="utf-8").splitlines()
    chunks: list[dict[str, Any]] = []

    # Parser state — identical logic to parse_json()
    current_section = "OVERVIEW"
    current_subsection = ""
    current_part = "PART_I"
    in_notes = False
    text_buffer: list[str] = []
    in_table = False
    table_lines: list[str] = []

    def flush_buffer() -> None:
        if not text_buffer:
            return
        combined = _normalize_financial_numbers(" ".join(text_buffer).strip())
        if len(combined) >= _MIN_TEXT_LEN:
            chunks.append({
                "text": combined,
                "chunk_type": "text",
                "section": current_section,
                "subsection": current_subsection,
                "page": 0,
                **meta,
            })
        text_buffer.clear()

    def flush_table() -> None:
        html_content = "\n".join(table_lines)
        content = html_table_to_markdown(html_content)
        content = _normalize_financial_numbers(content)
        if not content or len(content.strip()) < _MIN_TEXT_LEN:
            return
        table_section = current_section
        inferred = _infer_section_from_table(content)
        if inferred and inferred != current_section:
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
            "page": 0,
            **meta,
        })

    for line in lines:
        stripped = line.strip()

        # ── Inside a multi-line HTML table ───────────────────────────────────
        if in_table:
            table_lines.append(stripped)
            if "</table>" in stripped.lower():
                flush_table()
                table_lines.clear()
                in_table = False
            continue

        # ── Start of HTML table ──────────────────────────────────────────────
        if stripped.lower().startswith("<table"):
            flush_buffer()
            table_lines = [stripped]
            if "</table>" in stripped.lower():
                flush_table()
                table_lines.clear()
            else:
                in_table = True
            continue

        # ── H1 (or any #-level) header → section boundary ───────────────────
        if stripped.startswith("#"):
            m = re.match(r"^#+\s+(.*)", stripped)
            if not m:
                continue
            title_text = html_lib.unescape(m.group(1).strip())
            if not title_text:
                continue
            flush_buffer()

            if len(title_text) >= 200:
                text_buffer.append(title_text)
                continue

            section = detect_section(title_text, current_part)
            if section:
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
                    current_subsection = section
                else:
                    in_notes = False
                    current_section = section
                    current_subsection = ""
            else:
                note_m = re.match(r"^Note\s+(\d+)\s*[–\-]\s*(.+)", title_text, re.IGNORECASE)
                if note_m and in_notes:
                    current_subsection = f"Note {note_m.group(1)}: {note_m.group(2)[:60].strip()}"
                elif len(title_text) >= _MIN_TEXT_LEN:
                    text_buffer.append(title_text)
            continue

        # ── Regular text line ────────────────────────────────────────────────
        if stripped:
            text_buffer.append(html_lib.unescape(stripped))

    flush_buffer()
    return chunks


def parse_all_markdowns(dataset_dir: str | Path) -> list[dict[str, Any]]:
    """Recursively parse all .md files in the dataset directory."""
    dataset_dir = Path(dataset_dir)
    all_chunks: list[dict[str, Any]] = []
    md_files = sorted(dataset_dir.rglob("*.md"))
    print(f"Found {len(md_files)} Markdown files")
    for mf in md_files:
        print(f"  Parsing {mf.name[:50]}...", end=" ", flush=True)
        try:
            chunks = parse_md(mf)
            print(f"→ {len(chunks)} chunks")
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"ERROR: {e}")
    return all_chunks
