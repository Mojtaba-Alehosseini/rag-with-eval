# Project: RAG over documents with evaluation

## What this is
A Gradio RAG app over a real document corpus that answers WITH citations, plus a RAGAS/offline
harness reporting faithfulness, answer-relevancy, and context-precision, and showing a measurable
improvement from one retrieval upgrade (cross-encoder reranking or BM25+dense hybrid search).

## Stack
Python 3.11+ · LlamaIndex `SentenceSplitter` (chunking) · ChromaDB (store) ·
sentence-transformers (embeddings + cross-encoder reranker) · Ollama (`llama3.1:8b`) /
Gemini (demo) · RAGAS + a dependency-free offline metric fallback · Gradio · pytest · ruff · mypy.

The heavy stack is OPTIONAL and lazy-imported. With only the core deps installed the whole
pipeline + eval run on deterministic offline providers (hashing embeddings, extractive LLM,
numpy vector store), so tests are hermetic and CI needs no GPU, no API key, no model download.

## Commands
- App:    `python app.py`
- Build index: `python -m rag.pipeline build`     (loads `data/corpus/` -> chunks -> embeds -> store)
- Query:  `python -m rag.pipeline query "your question"`
- Tests:  `pytest -q`     Lint: `ruff check .`     Types: `mypy src`
- Eval:   `python eval/run_eval.py`                (prints base-vs-upgraded metric table)

## Conventions
- Every answer MUST return citations (source file + page/section). No citation -> it's a bug.
- Retrieval components are swappable (base vs reranked/hybrid) via config (env vars / `Config`)
  so we can A/B them in eval.
- NEVER import heavy deps at module top level — import lazily inside functions and fall back to
  the offline provider so the core stays installable and tests stay hermetic.
- Pin deps. Secrets in `.env`. Don't commit copyrighted PDFs — use the sample corpus or a fetch script.

## Gotchas (learned the hard way)
- RAGAS defaults to an OpenAI judge AND OpenAI embeddings and raises with no `OPENAI_API_KEY`.
  `eval/run_eval.py` reconfigures the judge to local Ollama / Gemini before calling `evaluate()`.
- Use LlamaIndex `SentenceSplitter`, NOT LangChain's `RecursiveCharacterTextSplitter`. Don't mix frameworks.
- The package is importable as `rag` (src layout) after `pip install -e .` — it is `rag.*`, not `src.rag.*`.

## Done per step
Step's verify command passes -> conventional commit.
