# 特斯拉 SEC 财报问答系统

一个端到端的检索增强生成（RAG）系统，用于查询特斯拉 2021-2025 年的 SEC 财报文件（10-K 年报和 10-Q 季报）。效果图.png

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 在 .env 中设置 API 密钥
echo "OPENAI_API_KEY=sk-your-key" > .env
echo "OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1" >> .env
echo "CHAT_MODEL=qwen-plus" >> .env
echo "EMBEDDING_MODEL=text-embedding-v3" >> .env

# 3. 构建索引（嵌入计算需要约 5-10 分钟）
python ingest.py

# 4. 启动 UI
python app.py
# → http://localhost:7860
```

---

## 数据概览

### 已成功处理的文档

| 年份 | 10-K（年报） | 10-Q Q1 | 10-Q Q2 | 10-Q Q3 |
|------|--------------|---------|---------|---------|
| 2021 | Tesla-10K-2021.json | Tesla-Q1-2021.json | Tesla-Q2-2021.json | Tesla-Q3-2021.json |
| 2022 | Tesla-10K-2022.json | Tesla-Q1-2022.json | Tesla-Q2-2022.json | Tesla-Q3-2022.json |
| 2023 | Tesla-10K-2023.json | Tesla-Q1-2023.json | Tesla-Q2-2023.json | Tesla-Q3-2023.json |
| 2024 | Tesla-10K-2024.json | Tesla-Q1-2024.json | Tesla-Q2-2024.json | Tesla-Q3-2024.json |
| 2025 | Tesla-10K-2025.json | Tesla-Q1-2025.json | Tesla-Q2-2025.json | Tesla-Q3-2025.json |

**总计：20 个文件**（5 个年度 10-K + 15 个季度 10-Q）。注意：Q4 报告不单独提交；10-K 涵盖包括 Q4 在内的整个财年。

### 索引统计

| 指标 | 数值 |
|--------|-------|
| 索引的总块数 | 6,153 |
| 表格块 | 1,010 |
| 文本块 | 5,143 |
| 季度报告块 | 3,530 |
| 年度报告块 | 2,623 |
| 覆盖期间 | 20（FY2021–FY2025，每年 Q1–Q3） |

### 数据源格式

所有文档均为 PDF 解析后的 **MinerU JSON 输出**（`_version_name: 2.7.5`）。每个 JSON 文件包含：
- `pdf_info`：包含 `para_blocks` 的页面对象数组
- 块类型：`title`、`text`、`list`、`table`、`table_caption`、`table_footnote`、`header`、`page_number`
- 表格以 HTML 形式嵌入在 `blocks[].lines[].spans[].html` 中

---

## 系统设计决策

### 1. 文档解析（`src/json_parser.py`）

**功能**：直接读取 MinerU JSON 输出，按页提取结构化块。

**实现方式**：
- `title` 块 → 通过 `SECTION_MAP`（精确匹配）+ 子串关键词回退 + SEC Item 编号模式（如 "Item 1A" → RISK FACTORS）进行章节标题检测
- `text` 和 `list` 块 → 带有当前章节元数据的纯文本块
- `table` 块 → 从嵌套 span 中提取 HTML → BeautifulSoup 解析 → 管道分隔的 markdown 表格（保留所有列标题，处理 `colspan`）
- `table_caption` → 缓冲并添加到下一个表格的标签前
- `table_footnote` → 附加到最近发出的表格块
- `header`、`page_number`、`ref_text` → 完全跳过

**为什么选择 JSON 而非 Markdown/OCR**：MinerU JSON 提供结构化的块级访问，使表格提取可靠且保留列标题。基于 OCR 的方法（pdf2image + Tesseract）会丢失列关系。纯 markdown 转换会丢失章节检测所需的块类型元数据。

### 2. 分块策略（`src/chunker.py`）

**基于章节的语义分块**：
- 同一 `(filename, section)` 对内的文本块在拆分前合并，确保每个块具有连贯的含义（例如，给定季度的所有 MD&A 文本保持在一起）
- 表格是**原子的** — 每个表格是一个不可分割的块。拆分财务表格会破坏列标题关系
- 最大文本块大小：**1,200 字符**，连续块之间有 **150 字符重叠**

**每个块的元数据**：`chunk_id`、`filename`、`year`、`quarter`、`period`、`doc_type`、`section`、`chunk_type`、`page`

**嵌入字符串**：`[{period} | {section}] {text}` — 前缀将时间和主题上下文编码到每个嵌入中，提高时间特定查询的检索精度。

**为什么采用这种策略**：SEC 财报具有明确定义的章节边界（PART I Item 1、MD&A、风险因素等）。尊重这些边界意味着对"汽车毛利率"的查询会针对正确的章节，而不是从不相关的章节中提取片段。

### 3. 嵌入（`src/indexer.py`）

- 提供商：**阿里云百炼 DashScope**（北京），OpenAI 兼容 API
- 模型：`text-embedding-v3`（1536 维）
- DashScope 批处理限制 = 每次 API 调用 10 个输入；内部分块
- 向量距离：ChromaDB 中的余弦相似度

### 4. 混合检索（`src/retriever.py`）

```
查询 → [BM25 top-20] + [向量 top-20] → RRF 融合 (k=60) → top-k 结果
```

- **BM25**（BM25Okapi）确保精确的财务术语匹配（"自由现金流"、"Q3 2022"、特定金额）不会因语义漂移而遗漏
- **向量搜索**（ChromaDB 余弦）处理改写和概念级查询
- **倒数排名融合**（k=60）结合两种排名，无需分数归一化

**自动过滤**：年份、季度和 doc_type（`annual_report`/`quarterly_report`）通过正则表达式从查询中自动提取，并作为 ChromaDB `where` 子句和 BM25 后过滤应用。

### 5. 多步骤问答流程（`src/qa_system.py`）

1. **查询分析**：LLM 分类复杂度（简单/复杂），提取年份/季度/主题，为多跳问题生成子查询
2. **多跳检索**：为主问题 + 每个子查询检索；通过 `chunk_id` 去重
3. **表格特定检索**：如果查询需要文本+表格连接，显式重新搜索表格块
4. **上下文组装**：按 RRF 分数排名的前 12 个块，格式化为 `period | section | page` 引用
5. **LLM 生成**：`qwen-plus` 配合财务分析师系统提示，要求引用答案

---

## 测试集与结果

### 复杂测试问题（v2 — 修复后结果）

| # | 问题 | 类型 | 结果 | 关键发现 |
|---|----------|------|--------|-------------|
| Q1 | 比较 2021 年 10-K 和 2023 年 10-K 如何描述中国市场风险。有什么变化？ | 跨文档比较 | **失败** | 仍检索到 FINANCIAL STATEMENTS；解析器中 10-K RISK FACTORS 章节标注缺口 |
| Q2 | 2022 年所有四个季度的研发费用总额？与 FY2021 比较。 | 数值计算 | **成功** | 找到 $3,075M（FY2022）vs $2,593M（FY2021）；doc_type 过滤正常工作 |
| Q3 | 2021-2024 年哪个季度的汽车毛利率最低？MD&A 怎么说？ | 文本 + 表格连接 | **部分成功** | 找到 Q1-2024 = 18.5%；汽车 GM 表格被错误标记为 SERVICES AND OTHER SEGMENT |
| Q4 | 特斯拉 2021 年至 2023 年每季度自由现金流波动情况。 | 时间多文档 | **部分成功（改进）** | 添加了 FCF 推导提示；检索 LIQUIDITY 章节；部分 YTD 中点仍缺失 |
| Q5 | 供应链风险披露从 Q1-2021 到 Q3-2023 如何演变？引用季度。 | 多文档综合 | **部分成功（改进）** | EXHIBITS 黑名单恢复了 2021 RISK FACTORS 检索；2022-2023 仍稀疏 |

**总体成功率**：1 成功 / 3 部分成功 / 1 失败（计数相同；Q4 和 Q5 答案质量显著提高）。

### v1 和 v2 之间应用的修复

| 修复 | 文件 | 更改 | 影响 |
|-----|------|--------|--------|
| EXHIBITS 黑名单 | `src/retriever.py` | 从 BM25 和 RRF 后阻止 EXHIBITS/OTHER INFORMATION/PART IV | Q5：0 → 3 个 RISK FACTORS 来源被检索 |
| 数字规范化 | `src/json_parser.py` | 规范化 LaTeX `\$`、间隔小数、空格千位分隔符 | 减少 "$1237 billion" 类型的伪影 |
| doc_type 自动过滤移除 | `src/qa_system.py` | 仅从分析中传播 `annual_report`；从不自动限制为 `quarterly_report` | Q3：10-K 年度数据不再被排除 |
| 章节定向检索 | `src/retriever.py` + `src/qa_system.py` | 对叙述性查询额外检索 RISK FACTORS / MD&A | Q5：改进；Q1 仍因解析器标注缺口而失败 |
| FCF 推导提示 | `src/qa_system.py` | 指示 LLM 从当前 YTD 减去先前 YTD 以获得季度 FCF | Q4：模型现在显示正确的算术 |

### 剩余系统性问题

- **10-K RISK FACTORS 章节标注**：年报 RISK FACTORS 页面在解析期间被错误标记（为 FINANCIAL STATEMENTS 或 OVERVIEW），使章节定向检索对 10-K 风险查询无效
- **汽车细分章节不匹配**：标题为"Automotive & Services and Other Segment"的页面上的表格被标记为 `SERVICES AND OTHER SEGMENT` 而非 `AUTOMOTIVE SEGMENT`
- **FCF 的 YTD 覆盖不完整**：多年 FCF 查询需要同时 9 个 YTD 中点；并非所有都适合前 12 个上下文窗口

详细的每个案例根本原因分析和具体改进建议请参见 [FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md)。

---

## 文件结构

```
tesla/
├── src/
│   ├── json_parser.py   # MinerU JSON 解析器（主要）
│   ├── chunker.py       # 语义分块
│   ├── indexer.py       # ChromaDB + BM25 索引构建器/加载器
│   ├── retriever.py     # 混合 BM25+向量与 RRF 融合
│   └── qa_system.py     # 多步骤问答流程
├── dataset/             # 20 个 MinerU JSON 文件（2021–2025）
│   ├── 2021/
│   ├── 2022/
│   ├── 2023/
│   ├── 2024/
│   └── 2025/
├── data/
│   ├── chunks.json      # 6,153 个最终块
│   └── bm25_index.pkl   # BM25Okapi 序列化
├── chroma_db/           # ChromaDB 持久向量存储
├── app.py               # Gradio Web UI（端口 7860）
├── ingest.py            # 摄取流程
├── requirements.txt
└── .env                 # API 密钥（未提交）
```

## CLI 参考

```bash
python ingest.py              # 完整解析 + 索引构建
python ingest.py --index-only # 仅从现有 chunks.json 重建索引
python app.py                 # 在 http://localhost:7860 启动 Gradio UI
```

## 依赖项

| 包 | 版本 | 用途 |
|---------|---------|---------|
| chromadb | 0.4.22 | 向量存储 |
| openai | 1.12.0 | API 客户端（OpenAI 兼容） |
| rank-bm25 | 0.2.2 | BM25 关键词搜索 |
| gradio | 4.19.2 | Web UI |
| beautifulsoup4 | 4.12.3 | HTML 表格解析 |
| python-dotenv | 1.0.1 | 环境配置 |
| numpy | 1.26.4 | 数值运算 |
| tqdm | 4.66.1 | 进度条 |
