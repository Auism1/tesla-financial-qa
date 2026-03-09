"""
特斯拉财报问答系统的 Gradio 5 UI。
运行：python app.py
"""
import os
import json
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from src.indexer import load_index, index_exists
from src.qa_system import answer_question

# ── 全局状态 ───────────────────────────────────────────────────────────────
_state: dict = {"collection": None, "bm25": None, "chunks": None, "api_key": None}


def load_system(api_key: str) -> str:
    """加载索引系统"""
    if not api_key.strip():
        return "❌ 请输入您的 OpenAI / DashScope API 密钥。"
    if not index_exists():
        return "❌ 未找到索引。请先运行 `python ingest.py`。"
    try:
        col, bm25, chunks = load_index(api_key.strip())
        _state["collection"] = col
        _state["bm25"] = bm25
        _state["chunks"] = chunks
        _state["api_key"] = api_key.strip()
        periods = sorted(set(c.get("period") for c in chunks))
        return (
            f"✅ 已加载 {len(chunks)} 个块 | "
            f"{len(periods)} 个期间：{periods[0]} → {periods[-1]}"
        )
    except Exception as e:
        return f"❌ 加载失败：{e}"


def ask(
    question: str,
    year_filter: list[str],
    doc_type_filter: str,
    show_debug: bool,
    show_sources: bool,
) -> tuple[str, str, str]:
    """处理问答请求"""
    if _state["collection"] is None:
        return "⚠️ 系统未加载。请输入 API 密钥并点击加载。", "", ""
    if not question.strip():
        return "⚠️ 请输入问题。", "", ""

    years = [y for y in (year_filter or []) if y != "全部"]
    dtype = None if doc_type_filter == "全部" else doc_type_filter

    try:
        result = answer_question(
            question=question,
            collection=_state["collection"],
            bm25=_state["bm25"],
            chunks=_state["chunks"],
            api_key=_state["api_key"],
            year_filter=years or None,
            doc_type_filter=dtype,
            top_k=10,
            debug=show_debug,
        )
    except Exception as e:
        return f"❌ 错误：{e}", "", ""

    answer = result["answer"]

    sources_md = ""
    if show_sources and result.get("sources"):
        lines = ["### 📚 检索来源\n"]
        for i, src in enumerate(result["sources"], 1):
            lines.append(
                f"**{i}. {src['period']} | {src['section']} | "
                f"第 {src['page']} 页 | {src['chunk_type'].upper()}**\n"
                f"> {src['preview']}\n"
                f"*(RRF 分数：{src['rrf_score']:.4f})*\n"
            )
        sources_md = "\n".join(lines)

    debug_str = ""
    if show_debug and result.get("debug_info"):
        debug_data = {
            "query_analysis": result.get("query_analysis", {}),
            "sub_queries": result.get("sub_queries", []),
            "debug_info": result.get("debug_info", {}),
        }
        debug_str = f"```json\n{json.dumps(debug_data, indent=2, ensure_ascii=False)}\n```"

    return answer, sources_md, debug_str


def set_question(q: str) -> str:
    return q


EXAMPLES = [
    "2021 年 10-K 和 2023 年 10-K 如何描述中国市场风险？有什么变化？",
    "2022 年所有四个季度的研发费用总额是多少？与 2021 年年报中的 FY2021 比较。",
    "2021 年至 2024 年哪个季度的汽车毛利率最低？MD&A 对该季度有何评论？",
    "描述特斯拉 2021 年至 2023 年每季度自由现金流的波动情况。",
    "特斯拉关于供应链的风险因素披露从 2021 年 Q1 到 2023 年 Q3 如何演变？",
    "根据 2023 年 10-K，特斯拉在 FY2023 的总收入和净利润是多少？",
    "比较特斯拉在 2022 年 Q1-Q3 的流动性状况（现金及现金等价物）。",
    "特斯拉在 2024 年 10-K 中披露的与自动驾驶和 AI 相关的主要风险是什么？",
]

# ── 构建 UI ───────────────────────────────────────────────────────────────────
with gr.Blocks(title="特斯拉财报问答", theme=gr.themes.Base()) as demo:
    gr.HTML("""
    <div style="text-align:center;padding:16px 0 8px">
      <h1 style="margin:0">🚗 特斯拉 SEC 财报问答</h1>
      <p style="color:#888;margin:4px 0 0">
        覆盖 2021–2025 · 10-K 年报 + 10-Q 季报 · BM25 + 向量混合检索
      </p>
    </div>
    """)

    with gr.Row():
        # ── 左侧面板 ──────────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=260):
            gr.Markdown("### ⚙️ 设置")
            api_key_box = gr.Textbox(
                label="API 密钥（OpenAI / DashScope）",
                type="password",
                value=os.environ.get("OPENAI_API_KEY", ""),
                placeholder="sk-…",
            )
            load_btn = gr.Button("🔄 加载系统", variant="primary")
            status_box = gr.Textbox(label="状态", interactive=False, lines=3)

            gr.Markdown("### 🔍 过滤器")
            year_cb = gr.CheckboxGroup(
                choices=["全部", "2021", "2022", "2023", "2024", "2025"],
                value=["全部"],
                label="年份",
            )
            dtype_dd = gr.Dropdown(
                choices=["全部", "annual_report", "quarterly_report"],
                value="全部",
                label="文档类型",
            )
            src_cb = gr.Checkbox(value=True, label="显示来源")
            dbg_cb = gr.Checkbox(value=False, label="显示调试信息")

        # ── 右侧面板 ─────────────────────────────────────────────────────────
        with gr.Column(scale=3):
            gr.Markdown("### 💬 问题")
            q_box = gr.Textbox(
                label="",
                placeholder="例如：特斯拉 2022 年 Q3 的自由现金流是多少？",
                lines=3,
            )
            with gr.Row():
                ask_btn = gr.Button("🔎 提问", variant="primary", scale=3)
                clr_btn = gr.Button("🗑 清除", scale=1)

            gr.Markdown("#### 📋 示例问题")
            for eq in EXAMPLES:
                gr.Button(eq, size="sm").click(fn=lambda q=eq: q, outputs=q_box)

    gr.Markdown("---")
    answer_out = gr.Markdown(label="答案")
    sources_out = gr.Markdown(label="来源")
    debug_out   = gr.Markdown(label="调试")

    # ── 事件 ──────────────────────────────────────────────────────────────────
    load_btn.click(fn=load_system, inputs=[api_key_box], outputs=[status_box])

    ask_btn.click(
        fn=ask,
        inputs=[q_box, year_cb, dtype_dd, dbg_cb, src_cb],
        outputs=[answer_out, sources_out, debug_out],
    )

    clr_btn.click(
        fn=lambda: ("", "", "", ""),
        outputs=[q_box, answer_out, sources_out, debug_out],
    )

    # 如果环境变量中存在 API 密钥，启动时自动加载
    demo.load(
        fn=lambda: load_system(os.environ.get("OPENAI_API_KEY", "")),
        outputs=[status_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
