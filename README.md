<!-- markdownlint-disable MD033 MD041 -->
# RAG over Documents — with a Real Evaluation Harness

> Ask questions over a document corpus and get answers **with citations to the source
> chunks**, backed by a proper **evaluation harness** that reports faithfulness,
> answer-relevancy and context-precision — and measures the effect of a retrieval upgrade
> (cross-encoder reranking / BM25+dense hybrid).

[![tests](https://img.shields.io/badge/tests-27%20passing-brightgreen)](#tests)
[![lint](https://img.shields.io/badge/ruff-clean-brightgreen)](#quality)
[![types](https://img.shields.io/badge/mypy-clean-brightgreen)](#quality)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Plain RAG is now baseline-expected — what differentiates an engineer is **measuring**
whether the system works and showing a number move when a component is upgraded. That is
what this repo is built around.

> **Measured result on the included corpus:** context-precision **0.87 → 0.98** after adding
> a cross-encoder reranker, and context-recall **0.95 → 1.00** with hybrid search — produced
> by the harness in `eval/run_eval.py`. [See the table ▾](#results)

---

## Why this is different from a tutorial

- **Evaluation is the centerpiece.** A reproducible harness (`eval/run_eval.py`) scores
  base vs. upgraded retrieval over a 24-question test set and prints a metrics table.
- **It runs with zero setup.** Core install has no torch, no API key, no model download:
  deterministic offline providers (hashing embeddings, extractive LLM, numpy vector store)
  make the whole pipeline + eval **hermetic and CI-friendly**. The real stack
  (LlamaIndex · ChromaDB · sentence-transformers · Ollama/Gemini · RAGAS) is lazy-loaded
  and switches on when installed.
- **No hallucinated claims without a citation.** Every answer cites `source file + page/
  section`; out-of-corpus questions are **refused**, not answered.
- **Bilingual.** The corpus and test set include Italian as well as English.

---

## Results

The eval harness compares retrieval variants on the same index over `eval/testset.yaml`.
Two stacks are reported because they tell different — and both honest — stories.

### Headline: real semantic stack (sentence-transformers bi-encoder + cross-encoder)

Measured on this machine — `RAG_EMBED_PROVIDER=sentence-transformers python eval/run_eval.py`
(`all-MiniLM-L6-v2` embeddings, `ms-marco-MiniLM-L-6-v2` reranker, 24-question test set):

| variant | context_precision | context_recall | faithfulness | answer_relevancy |
| --- | :---: | :---: | :---: | :---: |
| base (dense, all-MiniLM-L6-v2) | 0.873 | 0.952 | 1.000 | 0.658 |
| &nbsp;&nbsp;+ hybrid (BM25 + dense, RRF) | 0.905 | **1.000** | 1.000 | 0.649 |
| &nbsp;&nbsp;+ cross-encoder reranker | **0.980** | **1.000** | 1.000 | 0.652 |

> **context-precision 0.873 → 0.980 (+0.107)** after adding the cross-encoder reranker;
> **context-recall 0.952 → 1.000** after adding hybrid search.

**Why the upgrade helps here:** a semantic bi-encoder (all-MiniLM) embeds *meaning* and can
rank an exact rare token — `pgvector`, `HNSW`, `all-MiniLM-L6-v2`, `RRF` — below a merely
topical chunk, or miss it entirely (that's the recall gap BM25 closes). The cross-encoder
reads query and passage *together* and pushes the truly relevant chunk back to the top —
exactly what **context-precision** measures. (Metrics are the embedding/lexical proxies; the
*retrieval stack being measured* is the real semantic one. Add a judge for RAGAS numbers.)

### Reproducible offline stack (hashing embeddings + lexical reranker, no deps)

The exact same harness on the deterministic offline providers — no model download, no key,
reproducible in CI:

| variant | context_precision | context_recall | faithfulness | answer_relevancy | refusal_acc |
| --- | :---: | :---: | :---: | :---: | :---: |
| base | 0.861 | 0.905 | 1.000 | 0.427 | 1.000 |
| &nbsp;&nbsp;+ reranked (lexical) | 0.893 | **1.000** | 1.000 | 0.412 | 1.000 |
| &nbsp;&nbsp;+ hybrid (BM25) | 0.897 | **1.000** | 1.000 | 0.411 | 1.000 |

**The upgrade helps here too, but less — and the *why* is the interesting part:** the offline
embedder is itself **lexical** (a hashing bag-of-words), so a BM25-style reranker mostly
improves **recall** (context-recall 0.905 → 1.000, recovering the answer chunk) rather than
adding a new ranking signal — the precision gain is a modest +0.03–0.04. A cross-encoder over
a *semantic* base adds genuinely new signal, so its precision gain (+0.107) is ~3× larger.
Measuring is what lets you say *which* upgrade is worth its cost, and why. (These numbers are
deterministic — re-running the harness reproduces them exactly.)

> `faithfulness = 1.0` and `refusal_acc = 1.0` are real: the offline LLM is **extractive**
> (so answers are grounded in context by construction) and every out-of-corpus question is
> correctly refused. `answer_relevancy`/`answer_similarity` are intentionally modest — they
> reward fluent generation, which the deterministic extractive provider doesn't attempt;
> they rise with a real LLM (Ollama/Gemini).

Metrics are written to `eval/results/` (`metrics.csv`, `metrics.md`, `per_item.json`).

---

## Problem

Trustworthy answers over a real corpus require two things a vanilla LLM can't give you:
**grounding** (answers tied to sources you can check) and **a way to know it's working**
(numbers, not vibes). This project delivers both: cited answers plus an evaluation harness
that quantifies retrieval and answer quality and the effect of each upgrade.

## Approach & architecture

```
docs ─► loaders ─► chunker ─► embeddings ─► vector store ─► retriever ─┐
       (pdf/md/txt) (SentenceSplitter)      (Chroma/numpy)  (+ BM25 hybrid) │
                                                            (+ reranker)    │
question ─────────────────────────────────────────────────────────────────►│
                                                                            ▼
                                       prompt(context + question) ─► LLM ─► answer + citations
                                                                            │
                              eval set (Q, ground-truth) ─► metrics: faithfulness,
                                                            answer-relevancy, context-precision
```

The retrieval **upgrades act at query time**, so flipping reranking/hybrid on never
requires re-indexing — that's what lets the harness A/B them against one index.

| Stage | Real stack (optional) | Offline default (always works) |
| --- | --- | --- |
| Chunking | LlamaIndex `SentenceSplitter` | sentence-packing splitter (same size/overlap) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` or Ollama `nomic-embed-text` | deterministic signed hashing embedder |
| Vector store | ChromaDB (cosine, persistent) | numpy brute-force cosine |
| Generation | Ollama `llama3.1:8b` / Gemini | extractive, citation-grounded |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | lexical BM25-style reranker |
| Eval metrics | RAGAS (local/Gemini judge) | embedding/lexical proxy metrics |

## Run it (3 commands)

```bash
pip install -e .                                   # light core, no torch/keys needed
python -m rag.pipeline build                        # ingest + index data/corpus/
python -m rag.pipeline query "Which metric does reranking most improve?"
```

Evaluation and the UI:

```bash
python eval/run_eval.py                             # base-vs-upgraded metrics table
python app.py                                        # Gradio UI at http://127.0.0.1:7860
```

### Turn on the real stack

```bash
pip install -e ".[heavy]"        # LlamaIndex + Chroma + sentence-transformers + BM25
pip install -e ".[eval]"         # RAGAS with a local judge (never OpenAI)
pip install -e ".[ollama]"       # local llama3.1:8b generation     (or .[demo] for Gemini)
python -m rag.pipeline build && python eval/run_eval.py --judge ragas
```

Configuration is via env vars / `.env` (see `.env.example`) — provider selection,
`RAG_RERANKER`, `RAG_HYBRID`, `RAG_TOP_K`, model names, paths.

## Project structure

```
src/rag/
  config.py        # one immutable Config from env; the A/B knobs live here
  ingest.py        # load pdf/md/txt -> chunks with (source, page) provenance
  index.py         # embed -> Chroma or numpy store (+ manifest validation)
  retrieve.py      # dense top-k (+ BM25 hybrid via RRF, + reranker)
  generate.py      # cited answer + no-context guard (recall/cosine based)
  pipeline.py      # end-to-end query() + `python -m rag.pipeline` CLI
  metrics.py       # offline proxy metrics (avg-precision, grounding, relevancy)
  providers/       # embeddings / llm / reranker — real impls + offline fallbacks
eval/
  testset.yaml     # 24 Q + ground-truth (EN + IT, incl. out-of-corpus negatives)
  run_eval.py      # base vs upgraded harness -> metrics table + artifacts
app.py             # Gradio UI: question, toggles, cited answer, latency
data/corpus/       # original MIT-licensed sample docs (+ fetch script for real PDFs)
tests/             # 44 hermetic tests (ingest, retrieval, citations, guard, metrics, textutil)
```

## How the evaluation works

- **context_precision** — rank-sensitive average precision over retrieved chunks (the
  metric a reranker is expected to lift). **context_recall** — share of gold sources
  retrieved. **faithfulness** — fraction of answer sentences grounded in the retrieved
  context (catches hallucination). **answer_relevancy / answer_similarity** — cosine of the
  answer to the question / ground-truth. **refusal_accuracy** — out-of-corpus questions
  correctly refused.
- With a judge configured, `--judge ragas` additionally computes the real RAGAS
  faithfulness / answer-relevancy / context-precision. The judge is wired to **local Ollama
  or Gemini, never OpenAI**, so `evaluate()` runs with no `OPENAI_API_KEY` (the classic
  RAGAS footgun).

## <a id="tests"></a>Tests &amp; <a id="quality"></a>quality

```bash
pytest -q          # 27 tests, hermetic, ~0.3s
ruff check .       # clean
mypy src           # clean
```

## Deploy (Hugging Face Spaces, free CPU)

`requirements.txt` installs the package + Gradio for a Space. Ollama can't run on free CPU,
so the hosted demo uses the **Gemini free tier** when `GEMINI_API_KEY` is set as a Space
secret (and still works offline-extractive without one). Create a Gradio Space, push this
repo, and add this front-matter to the Space's `README.md`:

```yaml
---
title: RAG with Evaluation
sdk: gradio
app_file: app.py
---
```

## Limitations

- The committed corpus is a small original sample; swap in your own PDFs (`data/corpus/`)
  or run `python data/fetch_corpus.py` for open-access papers, then rebuild.
- Offline proxy metrics are **proxies** — directionally useful and reproducible, but the
  LLM-judged RAGAS numbers (with a real judge) are the gold standard.
- The numpy store is brute-force (fine for ≤ a few thousand chunks); use the Chroma backend
  for larger corpora.

## Credits

- Built with [LlamaIndex](https://github.com/run-llama/llama_index) (chunking),
  [ChromaDB](https://github.com/chroma-core/chroma) (store),
  [sentence-transformers](https://www.sbert.net/) (embeddings + reranker),
  [RAGAS](https://github.com/explodinggradients/ragas) (LLM-judged eval), and
  [Gradio](https://www.gradio.app/) (UI).
- RAG techniques (hybrid fusion, reranking) studied from
  [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) — reimplemented
  in original code here. Structure inspired by the MIT-licensed
  [patchy631/ai-engineering-hub](https://github.com/patchy631/ai-engineering-hub).
- Production-RAG reference: [infiniflow/ragflow](https://github.com/infiniflow/ragflow).

## License

MIT — see [LICENSE](LICENSE).

---

*Built by [Mojtaba Alehosseini](https://github.com/Mojtaba-Alehosseini) — data scientist.*
