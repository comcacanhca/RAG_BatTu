"""
Gradio Web UI — AlignRAG
Agentic RAG for LLM Alignment Research
"""

import logging
import gradio as gr
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

from pipeline import RAGPipeline, load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

config = load_config()
pipeline = RAGPipeline(config)
pipeline.init_models(skip_llm=False)
try:
    pipeline.load_existing_index()
except Exception:
    pass


def fetch_and_index(query: str, max_papers: int, progress=gr.Progress()):
    if not query.strip():
        return "❌ Please enter a search query."
    progress(0, desc="Searching arXiv...")
    try:
        chunk_dicts = pipeline.fetch_and_parse(query=query, max_papers=int(max_papers))
        if not chunk_dicts:
            return "❌ No papers found or downloaded."
        progress(0.6, desc="Building index...")
        pipeline.build_index(chunk_dicts)
        num_papers = len(set(c["arxiv_id"] for c in chunk_dicts))
        return (f"✅ Done! Indexed **{num_papers} papers** → "
                f"**{len(chunk_dicts)} chunks**.\n\nYou can now ask questions.")
    except Exception as e:
        logger.exception("Error in fetch_and_index")
        return f"❌ Error: {str(e)}"


def ask_question(question: str):
    if not question.strip():
        return "Please enter a question.", "", ""
    try:
        result = pipeline.query(question)

        # ── Answer ──────────────────────────────────────────────────────────
        answer = result["answer"]

        # ── References ──────────────────────────────────────────────────────
        refs_md = ""
        for c in result.get("citations", []):
            authors = ", ".join(c.get("authors", [])[:2])
            if len(c.get("authors", [])) > 2:
                authors += " et al."
            refs_md += (
                f"**[{c['index']}]** {c['title']}  \n"
                f"&nbsp;&nbsp;&nbsp;&nbsp;{authors}  \n"
                f"&nbsp;&nbsp;&nbsp;&nbsp;"
                f"[📄 Abstract]({c.get('abs_url', '')}) "
                f"| [📥 PDF]({c.get('pdf_url', '')})\n\n"
            )

        # ── Agent Trace ──────────────────────────────────────────────────────
        mode = result.get("mode", "general")
        sub_queries = result.get("sub_queries", [question])
        gap_sup = result.get("gap_supplemented", False)
        revised = result.get("answer_revised", False)
        confidence = result.get("confidence", "?")
        gen_stats = result.get("generation_stats", {})

        mode_icons = {"comparison": "⚖️", "taxonomy": "🗂️", "general": "💬", "table": "📋"}
        conf_icons = {"high": "🟢", "medium": "🟡", "low": "🔴"}

        trace = f"### Agent Pipeline Trace\n\n"
        trace += f"**Mode detected:** {mode_icons.get(mode, '❓')} `{mode}`  \n"
        trace += (f"**Retrieval confidence:** {conf_icons.get(confidence, '⚪')} `{confidence}` "
                  f"(top={result.get('retrieval_top_score', 0):.2f}, "
                  f"margin={result.get('retrieval_margin', 0):.2f}, "
                  f"relevant={result.get('retrieval_num_relevant', 0)})\n\n")

        if len(sub_queries) > 1:
            trace += "**① Query Planner** — sub-queries generated:\n"
            for i, sq in enumerate(sub_queries, 1):
                trace += f"  - `{sq}`\n"
        else:
            trace += f"**① Query Planner** — single query ({mode} mode)\n"

        if mode in ("comparison", "taxonomy"):
            gap_label = "✅ coverage OK" if not gap_sup else "⚡ gap found → supplemental retrieval ran"
            trace += f"\n**② Gap Detector** — {gap_label}  \n"
        else:
            trace += "\n**② Gap Detector** — skipped (general mode)  \n"

        if mode in ("comparison", "taxonomy"):
            rev_label = "✏️ answer revised" if revised else "✅ PASS"
            trace += f"\n**③ Self-Critique** — {rev_label}  \n"
        else:
            trace += "\n**③ Self-Critique** — skipped (general mode)  \n"

        if gen_stats:
            cov = gen_stats.get("citation_coverage", 0)
            trace += (f"\n**Generation stats:** "
                      f"{gen_stats.get('answer_tokens', 0)} tokens · "
                      f"{gen_stats.get('num_citations_used', 0)} papers cited · "
                      f"coverage {cov:.0%} · "
                      f"uncited sentences: {gen_stats.get('uncited_sentences', 0)}")

        return answer, refs_md, trace

    except Exception as e:
        logger.exception("Error in ask_question")
        return f"Error: {str(e)}", "", ""


# ─── UI Layout ────────────────────────────────────────────────────────────────

with gr.Blocks(title="AlignRAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🔬 AlignRAG\n"
        "**Agentic RAG for LLM Alignment Research** — "
        "Query Planning · Gap Detection · Self-Critique · Section-aware Retrieval"
    )

    with gr.Tab("🔍 Step 1: Fetch & Index Papers"):
        gr.Markdown(
            "Search arXiv for alignment papers (DPO, PPO, RLHF, KTO, ...) "
            "and build the vector index."
        )
        with gr.Row():
            query_input = gr.Textbox(
                label="arXiv Search Query",
                placeholder="e.g. LLM alignment preference optimization DPO PPO",
                scale=3)
            max_papers_input = gr.Slider(
                label="Max Papers", minimum=10, maximum=500, value=100, step=10, scale=1)
        fetch_btn = gr.Button("🚀 Fetch & Index", variant="primary")
        fetch_status = gr.Markdown()
        fetch_btn.click(fn=fetch_and_index,
                        inputs=[query_input, max_papers_input],
                        outputs=fetch_status)

    with gr.Tab("💬 Step 2: Ask Questions"):
        gr.Markdown(
            "Ask comparison, taxonomy, or general questions. "
            "The agentic pipeline will automatically plan sub-queries, "
            "detect evidence gaps, and self-critique the answer."
        )
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Research Question", lines=3,
                    placeholder=(
                        "Try:\n"
                        "• Compare DPO vs PPO training stability and failure modes\n"
                        "• What are the main families of preference optimization methods?\n"
                        "• How does the DPO objective differ from RLHF?"
                    ))
                ask_btn = gr.Button("🧠 Ask", variant="primary")
                answer_output = gr.Markdown(label="Answer")
                refs_output = gr.Markdown(label="References")

            with gr.Column(scale=1):
                trace_output = gr.Markdown(label="Agent Pipeline Trace")

        ask_btn.click(fn=ask_question,
                      inputs=question_input,
                      outputs=[answer_output, refs_output, trace_output])

    gr.Markdown(
        "---\n"
        "*Embedding: bge-large-en-v1.5 · Reranker: bge-reranker-large · "
        "LLM: Qwen2.5-7B (local HF) · FAISS + BM25*"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
