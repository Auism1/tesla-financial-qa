# 失败分析 — 特斯拉 SEC 财报问答系统

针对实际系统（JSON 数据集，6,153 个块，应用所有三个系统性修复后）运行的 5 个测试案例的深度分析。结果记录自 `python run_tests.py`。

---

## 测试结果摘要

| 案例 | 问题 | 总体结果 | 根本原因类别 |
|------|----------|---------------|-------------------|
| Q1 | 中国市场风险 2021 vs 2023 10-K | **失败** | 检索到错误章节 — 10-K RISK FACTORS 块未浮现 |
| Q2 | 2022 年总研发费用 vs FY2021 | **成功** | — |
| Q3 | 最低汽车毛利率 + MD&A | **部分成功** | 章节标签不匹配（汽车 GM 表格标记为 SERVICES AND OTHER SEGMENT） |
| Q4 | 2021-2023 年每季度 FCF | **部分成功（改进）** | 10-Q 中仅有 YTD 现金流；FCF 推导指令现已添加到 LLM 提示 |
| Q5 | 供应链风险演变 Q1-2021→Q3-2023 | **部分成功（改进）** | EXHIBITS 黑名单修复有效；2022-2023 RISK FACTORS 在上下文中仍稀疏 |

---

## 案例 1：中国市场风险比较（Q1）— 失败

**问题**：*比较 2021 年 10-K 和 2023 年 10-K 如何描述中国市场风险。有什么变化？*

**观察到的行为**：查询分析正确设置了 `doc_type="annual_report"`，`years=["2021","2023"]`。针对风险因素查询的章节定向检索通道（匹配"market risk"关键词）触发但仍未能浮现 RISK FACTORS 内容。检索到的顶部来源：
```
('FY2021', 'FINANCIAL STATEMENTS', 'table')
('FY2023', 'PART III', 'text')
('FY2021', 'OVERVIEW', 'text')
('FY2023', 'FINANCIAL STATEMENTS', 'text')
('FY2023', 'LEGAL PROCEEDINGS', 'text')
```
没有来自 RISK FACTORS 章节的块出现。LLM 正确报告了缺失的上下文，无法比较两年。

**与之前运行相比的变化**：EXHIBITS 黑名单修复从结果中删除了认证/附件块 — 来源不再包括 EXHIBITS 或 OTHER INFORMATION。然而，FINANCIAL STATEMENTS 和 PART III 仍占主导地位。RISK FACTORS 的章节定向检索通道运行了（由"market risk"关键词触发）但返回了空结果，这意味着要么 10-K RISK FACTORS 章节块在索引中有不同的章节标签，要么该章节内的 BM25/向量匹配对"China market risk"的排名不高。

**根本原因追踪**：
- BM25 对"China"的评分最高是针对财务报表脚注（例如，"Tesla (Shanghai) Co., Ltd."出现在资产表中，在结构化列表中具有高词频，提升了 TF）。
- 向量搜索将"China market risks"映射到最接近 MARKET RISK DISCLOSURES 章节（利率/外汇风险语言），而不是 RISK FACTORS 叙述。
- 讨论中国的 10-K RISK FACTORS 章节块可能由于复杂的 10-K 页面结构而在不同的章节标签下索引，因此章节定向检索通道找到了空结果集。

**根本原因**：**10-K 解析中的章节检测缺口** — 10-K 年报中的某些 RISK FACTORS 页面被错误标记（例如，分配给"FINANCIAL STATEMENTS"或"OVERVIEW"），这是由于前面的标题块。此外，章节盲 RRF 无法区分 RISK FACTORS 叙述和包含"China"一词的财务表格块。

**具体改进**：
1. **检查 10-K RISK FACTORS 块**：查询 `chunks.json` 以检查有多少 10-K 块带有 `section="RISK FACTORS"` — 如果为零，则 10-K RISK FACTORS 章节在解析时被错误标记。
2. **加强子查询重新表述**：对于"比较风险披露"查询，自动生成子查询，如 `"Risk Factors Item 1A China 2021 annual report"`，使用明确的 Item 1A 语言来提升 BM25 术语匹配，针对章节前缀嵌入 `[FY2021 | RISK FACTORS]`。
3. **扩展 `_ITEM_MAP`**：在 `json_parser.py` 中验证 10-K Item 1A 检测是否正确触发；为 10-K 目录标题块中出现的"Risk Factors"添加额外别名。

---

## 案例 2：研发费用年度比较（Q2）— 成功

**问题**：*2022 年所有四个季度的研发费用总额是多少？与 2021 年年报中的 FY2021 比较。*

**观察到的行为**：查询分析设置了 `doc_type="annual_report"`，`years=["2022","2021"]`，`quarters=[]`。成功检索并引用：
- FY2022 研发：**$3,075 百万**
- FY2021 研发：**$2,593 百万**
- 同比增长：**+$482M (+19%)**

LLM 正确解释了"2022 年所有四个季度"= FY2022 10-K 数字，因为没有单独的 Q4 10-Q。找到了包含 2022、2021、2020 列的比较损益表表格作为来源 9。

**为什么有效**：`doc_type="annual_report"` 过滤器正确限制了检索。研发费用出现在标准化的损益表表格中，具有高关键词密度（BM25 优势）。"Research and development"在语义上嵌入接近"R&D expense"。

**无失败需要分析。**

---

## 案例 3：最低汽车毛利率 + MD&A（Q3）— 部分成功

**问题**：*2021 年至 2024 年哪个季度的汽车毛利率最低？MD&A 对该季度有何评论？*

**观察到的行为**：查询分析返回 `doc_type="quarterly_report"`（LLM 决策），`years=[2021–2024]`，`quarters=["Q1","Q2","Q3"]`。修复 3（doc_type 过滤器）正确抑制了 `quarterly_report` 传播到 `hybrid_search`。所有 8 个顶部来源都来自 `SERVICES AND OTHER SEGMENT` 章节。找到 Q1-2024 = 18.5% 作为报告的最低利润率。

**失败症状**：
1. **章节标签不匹配**：汽车毛利率表格在 `SERVICES AND OTHER SEGMENT` 章节下解析（因为页面上的前面标题块说"Automotive & Services and Other Segment"），而不是 `AUTOMOTIVE SEGMENT`。表格内容正确，但章节元数据错误。
2. **Q4 数据仍缺失**：即使删除了 doc_type 过滤器，10-K 汽车毛利率表格也在错误的章节标签下索引，因此它们不能可靠地浮现。Q4-2022、Q4-2023 仍未检查。
3. **章节定向检索未触发**：`_MDA_KEYWORDS` 正则表达式匹配，但随后的 MD&A 章节定向通道检索的 MD&A 文本块不包含汽车毛利率表格。

**根本原因**：**在页面级别分配的章节标签** — 当 10-Q 页面标题说"Automotive & Services and Other Segment"后跟两个子细分的表格时，解析器将 `SERVICES AND OTHER SEGMENT` 分配给该页面上的所有表格。汽车毛利率表格获得了错误的章节标签。

**具体改进**：
1. **修复 SECTION_MAP**：在 `json_parser.py` 中添加映射 `"AUTOMOTIVESERVICESANDOTHERSEGMENT"` → `"AUTOMOTIVE SEGMENT"`（当前映射到 `SERVICES AND OTHER SEGMENT`）。
2. **多通道表格检索**：对于毛利率查询，运行带有查询 `"automotive gross margin table percent"` 的定向通道，限制为 `chunk_type="table"`，无章节约束。
3. **直接表格搜索作为回退**：如果顶部结果中出现少于 3 个表格块用于指标查找查询，则运行带有 `chunk_type="table"` 过滤器的辅助检索通道以保证表格覆盖。

---

## 案例 4：每季度自由现金流（Q4_FCF）— 部分成功（改进）

**问题**：*描述特斯拉 2021 年至 2023 年每季度自由现金流的波动情况。*

**观察到的行为**（与 v1 相比改进）：来源现在包括 LIQUIDITY AND CAPITAL RESOURCES 和 MD&A 章节。LLM 正确地：
1. 解释了 10-Q 财报仅报告 YTD 累计经营现金流，而不是独立的季度 FCF。
2. 演示了减法方法（Q2 FCF = 6 个月 YTD − Q1 YTD），使用添加到 LLM 提示的 FCF 推导指令。
3. 计算了可用期间的部分 FCF 数字（Q1-2021、Q2-2021、Q1-2023）。

**修复 5 影响**：FCF 推导指令明确教导模型从当前 YTD 减去先前 YTD。模型现在显示其算术，并在上下文中缺少中间 YTD 数据时正确标记，而不是将 YTD 累计数字呈现为季度数字。

**剩余失败症状**：
1. **YTD 覆盖不完整**：并非所有 9 个季度 YTD 期间（2021、2022、2023 的 Q1-Q3）都在一次通道中检索到。缺失：Q2-2022 YTD、Q3-2022 YTD、Q2-2023 YTD、Q3-2023 YTD。
2. **数字格式伪影仍然存在**：一些解析的文本仍包含数字中的间距异常。规范化修复解决了 LaTeX 样式的伪影，但不是所有 MinerU span-merge 分隔符。

**根本原因**：(1) **SEC 10-Q 固有结构** — 季度 FCF 需要跨两个检索块的 YTD 减法，要求两者同时出现在前 12 个上下文中。(2) **检索广度 vs. 深度权衡** — 检索 9 个子查询（每个季度期间一个）成本高昂且稀释了 top-k 槽位。

**具体改进**：
1. **结构化 FCF 子查询生成**：当查询提到"自由现金流"+ 多年时，自动生成 YTD 对子查询 — 对于每个季度，检索当前和先前 YTD 以保证两个端点都存在。
2. **增加上下文窗口**：对于 FCF 查询，将 `max_chunks` 从 12 提高到 16-20，以容纳多个 YTD 期间。
3. **检索后算术步骤**：专用 FCF 计算步骤，从检索的块中提取 YTD 数字并在传递给 LLM 之前计算 Q-over-Q 差异。

---

## 案例 5：供应链风险演变 Q1-2021 至 Q3-2023（Q5）— 部分成功（显著改进）

**问题**：*特斯拉关于供应链的风险因素披露从 2021 年 Q1 到 2023 年 Q3 如何演变？引用具体季度。*

**观察到的行为**（与 v1 相比显著改进）：顶部来源现在包括：
```
('Q1-2021', 'RISK FACTORS', 'text')
('Q2-2021', 'RISK FACTORS', 'text')
('Q3-2021', 'RISK FACTORS', 'text')
('Q2-2022', 'MANAGEMENT'S DISCUSSION AND ANALYSIS', 'text')
('Q3-2022', 'MANAGEMENT'S DISCUSSION AND ANALYSIS', 'text')
('Q1-2023', 'ENERGY SEGMENT', 'text')
```

LLM 提供了实质性的、引用良好的答案：
- **Q1-2021**：广泛的疫情/事件框架；半导体短缺首次明确命名为当前活跃风险
- **Q2-2021**：增加了港口拥堵、劳动力短缺
- **Q3-2021**：地缘政治破坏框架加深
- **Q3-2022**：俄乌战争影响；增加了电池材料采购风险
- 2023：覆盖有限 — 检索到 ENERGY SEGMENT 而不是 2023 季度的 RISK FACTORS

**修复 1 影响**：EXHIBITS 黑名单完全消除了认证/附件污染。2021 年的 RISK FACTORS 章节现在主导顶部结果 — 与之前运行相比质量有重大改进，之前所有 6 个顶部来源都来自 EXHIBITS 或 OTHER INFORMATION（返回零有用内容）。

**剩余失败症状**：
1. **2022-2023 RISK FACTORS 缺失**：Q2-2022、Q1-2022、Q1-2023 RISK FACTORS 块未检索到 — 出现 MD&A 或 ENERGY SEGMENT 块。
2. **章节定向通道部分有效**：`_RISK_KEYWORDS` 正则表达式匹配了查询，RISK FACTORS 章节通道正确检索了 2021 年数据，但没有检索 2022-2023 期间，表明跨年份的章节标记不一致。

**根本原因**：**跨年份的 RISK FACTORS 章节标记不一致** — 2021 年季度报告具有清晰标记的 RISK FACTORS 章节（Item 1A 检测有效）。2022-2023 年季度报告可能具有不同的结构布局或标题块格式，导致章节检测器为风险因素文本分配不同的标签。

**具体改进**：
1. **验证 2022-2023 RISK FACTORS 块**：检查 `data/chunks.json` 中 `year in ["2022","2023"]` 和 `section="RISK FACTORS"` 以确认它们存在并正确标记。
2. **扩展关键词模式**：检查 2022-2023 10-Q 文件是否使用不同的 Item 编号或措辞（例如，"ITEM 1A." vs "Item 1A."），当前 `_ITEM_MAP` 模式未匹配。
3. **回退到更广泛的检索**：当章节定向通道返回 < 3 个结果时，回退到无限制搜索，使用明确的子查询，如 `"supply chain risk Item 1A quarterly filing {year}"`。

---

## 系统性问题 — 更新状态

### 问题 1：MinerU 解析的数字格式伪影

**状态**：部分修复。

**应用的修复**：在 `json_parser.py` 中添加了 `_normalize_financial_numbers()` — 处理 `\$ X` LaTeX 前缀、间隔小数（`$12 30` → `$12.30`）、空格分隔的千位（`$1 237` → `$1,237`）。

**剩余**：MinerU 将数字输出为单独的 span 令牌而没有美元前缀的伪影（例如，数字列中的纯 `1237`）需要在 HTML 到 markdown 转换期间进行表级规范化。

### 问题 2：EXHIBITS/OTHER INFORMATION 章节污染

**状态**：已修复。✓

**应用的修复**：`retriever.py` 中的 `SECTION_BLOCKLIST = {"EXHIBITS", "OTHER INFORMATION", "PART IV", "SECURITY OWNERSHIP", "CERTIFICATIONS"}`。BM25 在评分期间跳过黑名单块；RRF 后过滤器删除任何通过向量搜索存活的块。

**结果**：Q5 从 0/8 相关来源改进到 3/8 RISK FACTORS 来源。Q1 不再返回 EXHIBITS/认证块。

### 问题 3：doc_type 自动过滤排除 Q4 年度数据

**状态**：已修复。✓

**应用的修复**：在 `qa_system.py` 中，仅从查询分析传播 `doc_type="annual_report"` 到 `hybrid_search`。从不自动限制为 `quarterly_report`。

**结果**：Q3（汽车毛利率）不再自动过滤为 quarterly_report，允许 10-K 年度数据可能出现。剩余问题（章节错误标记）是一个单独的解析器问题。

---

## 摘要表（v2 — 修复后）

| 案例 | 症状 | 根本原因 | 应用的修复 | 剩余问题 |
|------|---------|-----------|-------------|-----------------|
| Q1 | 检索到 FINANCIAL STATEMENTS 而不是 RISK FACTORS | 10-K RISK FACTORS 错误标记；查询到章节嵌入不匹配 | 黑名单 + 章节定向通道 | 解析器级 10-K 章节标记 |
| Q2 | —（成功） | — | — | 无 |
| Q3 | 汽车 GM 表格标记为错误章节 | 页面级章节分配；组合标题映射错误 | doc_type 自动过滤移除 | 需要解析器 SECTION_MAP 修复 |
| Q4 | 仅 YTD 现金流；季度覆盖不完整 | SEC 10-Q 结构限制 | FCF 推导提示；数字规范化 | YTD 检索覆盖不完整 |
| Q5 | EXHIBITS 阻止了 2022-2023 RISK FACTORS | EXHIBITS 关键词污染 | 黑名单修复；章节定向检索 | 2022-2023 RISK FACTORS 标记不一致 |
