"""
Gradio 5 UI for Tesla Financial Reports Q&A System.
Run: python app.py
"""
import os
import json
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from src.indexer import load_index, index_exists
from src.qa_system import answer_question

# ── Global state ───────────────────────────────────────────────────────────────
_state: dict = {"collection": None, "bm25": None, "chunks": None, "api_key": None}


def load_system(api_key: str) -> str:
    if not api_key.strip():
        return "❌ Please enter your OpenAI / DashScope API key."
    if not index_exists():
        return "❌ Index not found. Run `python ingest.py` first."
    try:
        col, bm25, chunks = load_index(api_key.strip())
        _state["collection"] = col
        _state["bm25"] = bm25
        _state["chunks"] = chunks
        _state["api_key"] = api_key.strip()
        periods = sorted(set(c.get("period") for c in chunks))
        return (
            f"✅ Loaded {len(chunks)} chunks | "
            f"{len(periods)} periods: {periods[0]} → {periods[-1]}"
        )
    except Exception as e:
        return f"❌ Load failed: {e}"


def ask(
    question: str,
    year_filter: list[str],
    doc_type_filter: str,
    show_debug: bool,
    show_sources: bool,
) -> tuple[str, str, str]:
    if _state["collection"] is None:
        return "⚠️ System not loaded. Enter API key and click Load.", "", ""
    if not question.strip():
        return "⚠️ Please enter a question.", "", ""

    years = [y for y in (year_filter or []) if y != "All"]
    dtype = None if doc_type_filter == "All" else doc_type_filter

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
        return f"❌ Error: {e}", "", ""

    answer = result["answer"]

    sources_md = ""
    if show_sources and result.get("sources"):
        lines = ["### 📚 Retrieved Sources\n"]
        for i, src in enumerate(result["sources"], 1):
            lines.append(
                f"**{i}. {src['period']} | {src['section']} | "
                f"Page {src['page']} | {src['chunk_type'].upper()}**\n"
                f"> {src['preview']}\n"
                f"*(RRF score: {src['rrf_score']:.4f})*\n"
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
    "How did Tesla describe China market risks in the 2021 10-K vs 2023 10-K? What changed?",
    "What was the total R&D expense across all four quarters of 2022? Compare to FY2021 from the 2021 annual report.",
    "Which quarter from 2021 to 2024 had the lowest automotive gross margin? What does the MD&A say about that quarter?",
    "Describe Tesla's free cash flow fluctuation quarter by quarter from 2021 to 2023.",
    "How did Tesla's risk factor disclosures about supply chain evolve from 2021 Q1 to 2023 Q3?",
    "What was Tesla's total revenue and net income in FY2023 according to the 2023 10-K?",
    "Compare Tesla's liquidity position (cash and cash equivalents) across Q1-Q3 2022.",
    "What were the key risks Tesla disclosed in the 2024 10-K related to autonomous driving and AI?",
]

# ── Build UI ───────────────────────────────────────────────────────────────────
with gr.Blocks(title="Tesla Financial Q&A", theme=gr.themes.Base()) as demo:
    gr.HTML("""
    <div style="text-align:center;padding:16px 0 8px">
      <h1 style="margin:0">🚗 Tesla SEC Filings Q&amp;A</h1>
      <p style="color:#888;margin:4px 0 0">
        Covering 2021–2025 · 10-K Annual Reports + 10-Q Quarterly Reports · BM25 + Vector hybrid retrieval
      </p>
    </div>
    """)

    with gr.Row():
        # ── Left panel ──────────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=260):
            gr.Markdown("### ⚙️ Setup")
            api_key_box = gr.Textbox(
                label="API Key (OpenAI / DashScope)",
                type="password",
                value=os.environ.get("OPENAI_API_KEY", ""),
                placeholder="sk-…",
            )
            load_btn = gr.Button("🔄 Load System", variant="primary")
            status_box = gr.Textbox(label="Status", interactive=False, lines=3)

            gr.Markdown("### 🔍 Filters")
            year_cb = gr.CheckboxGroup(
                choices=["All", "2021", "2022", "2023", "2024", "2025"],
                value=["All"],
                label="Year",
            )
            dtype_dd = gr.Dropdown(
                choices=["All", "annual_report", "quarterly_report"],
                value="All",
                label="Document type",
            )
            src_cb = gr.Checkbox(value=True, label="Show sources")
            dbg_cb = gr.Checkbox(value=False, label="Show debug info")

        # ── Right panel ─────────────────────────────────────────────────────────
        with gr.Column(scale=3):
            gr.Markdown("### 💬 Question")
            q_box = gr.Textbox(
                label="",
                placeholder="e.g. What was Tesla's free cash flow in Q3 2022?",
                lines=3,
            )
            with gr.Row():
                ask_btn = gr.Button("🔎 Ask", variant="primary", scale=3)
                clr_btn = gr.Button("🗑 Clear", scale=1)

            gr.Markdown("#### 📋 Example questions")
            for eq in EXAMPLES:
                gr.Button(eq, size="sm").click(fn=lambda q=eq: q, outputs=q_box)

    gr.Markdown("---")
    answer_out = gr.Markdown(label="Answer")
    sources_out = gr.Markdown(label="Sources")
    debug_out   = gr.Markdown(label="Debug")

    # ── Events ──────────────────────────────────────────────────────────────────
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

    # Auto-load on startup if API key present in env
    demo.load(
        fn=lambda: load_system(os.environ.get("OPENAI_API_KEY", "")),
        outputs=[status_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
