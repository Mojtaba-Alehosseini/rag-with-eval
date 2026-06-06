"""Gradio UI for the RAG demo.

    python app.py            # local
    # On Hugging Face Spaces (Gradio SDK), this file is the entry point.

Question box → cited answer + expandable sources + latency, with live toggles for the
retrieval upgrades (reranking / hybrid) so a visitor can feel the difference base vs
upgraded. Provider selection is automatic: Gemini if GEMINI_API_KEY is set (the hosted
demo), else local Ollama, else the offline extractive provider.
"""

from __future__ import annotations

from rag.config import Config
from rag.pipeline import QueryResult, RAGPipeline
from rag.retrieve import Retriever

CONFIG = Config.from_env()


def _load_or_build(config: Config) -> RAGPipeline:
    pipe = RAGPipeline.from_config(config)
    try:
        pipe.load()
    except FileNotFoundError:
        pipe.build()
    return pipe


PIPE = _load_or_build(CONFIG)


def _format_citations(result: QueryResult) -> str:
    if not result.citations:
        return "_No sources (the answer was not found in the corpus)._"
    lines = ["### Sources"]
    for c in result.citations:
        used = " ✅ cited" if c.cited else ""
        lines.append(
            f"**[{c.index}] {c.source} · p.{c.page}**  _(score {c.score}){used}_\n\n"
            f"> {c.snippet}\n"
        )
    return "\n".join(lines)


def answer_fn(question: str, use_rerank: bool, use_hybrid: bool):
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", "", ""
    cfg = CONFIG.with_overrides(
        reranker="cross-encoder" if use_rerank else "none",
        hybrid=bool(use_hybrid),
    )
    pipe = RAGPipeline(
        cfg,
        PIPE.embedder,
        PIPE.llm,
        store=PIPE.store,
        retriever=Retriever.from_config(cfg, PIPE.embedder, PIPE.store),
    )
    result = pipe.query(question)
    meta = (
        f"llm=`{result.providers['llm']}` · embed=`{result.providers['embed']}` · "
        f"rerank=`{result.providers['rerank']}` · hybrid=`{result.providers['hybrid']}` · "
        f"{result.latency_ms:.0f} ms · {'answered' if result.answered else 'no answer'}"
    )
    return result.answer, _format_citations(result), meta


EXAMPLES = [
    ["What does the BM25 ranking function score a document by?", False, False],
    ["Which evaluation metric does reranking most directly improve?", True, False],
    ["How many dimensions does the all-MiniLM-L6-v2 model produce?", False, False],
    ["Che cos'è il RAG?", False, True],
    ["What is the capital of France?", False, False],
]


def build_demo():
    import gradio as gr

    with gr.Blocks(title="RAG with Evaluation") as demo:
        gr.Markdown(
            "# RAG over documents — with citations\n"
            "Ask a question about the corpus. Every answer cites its sources, and "
            "out-of-corpus questions are refused instead of hallucinated. Toggle the "
            "retrieval upgrades to compare base vs. reranked/hybrid."
        )
        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(label="Question", placeholder="Ask about the corpus…", lines=2)
            with gr.Column(scale=1):
                use_rerank = gr.Checkbox(label="Cross-encoder reranker", value=False)
                use_hybrid = gr.Checkbox(label="Hybrid (BM25 + dense)", value=False)
        ask = gr.Button("Ask", variant="primary")
        answer = gr.Markdown(label="Answer")
        meta = gr.Markdown()
        citations = gr.Markdown()

        ask.click(answer_fn, [question, use_rerank, use_hybrid], [answer, citations, meta])
        question.submit(answer_fn, [question, use_rerank, use_hybrid], [answer, citations, meta])
        gr.Examples(EXAMPLES, inputs=[question, use_rerank, use_hybrid])
    return demo


if __name__ == "__main__":
    build_demo().launch()
