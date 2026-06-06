# Project 02 — RAG over Documents, with a Real Evaluation Harness
**Track:** LLM / RAG · **Flagship:** pin-worthy · **Difficulty:** Medium · **Effort:** ~5 days · **Demo:** Hugging Face Spaces (Gradio)

> Execute with Claude Code. Read `claude-code-playbook.md` first. One step at a time, verify, commit.

---

## 0. Build setup, dependencies & gotchas (read before Step 1)
- **Package layout:** `src/` layout with `__init__.py`; `pip install -e .`; give `ingest.py`/`index.py` a `if __name__ == "__main__":` guard (or expose one `python -m rag.pipeline build` CLI) so the run commands work.
- **Dependencies** (pin after install): `llama-index`, `chromadb`, `sentence-transformers`, `ragas`, `gradio`, `pandas`, `pytest`, `ruff`, `mypy` (add `langchain` ONLY if you choose the LangChain path).
- **Chunker:** the stack is LlamaIndex → use `SentenceSplitter(chunk_size=1024, chunk_overlap=200)`, NOT LangChain's `RecursiveCharacterTextSplitter`. Don't mix frameworks.
- **RAGAS/DeepEval judge gotcha (THE most likely failure in this whole package):** both default to an OpenAI judge AND OpenAI embeddings, and `evaluate()` raises without `OPENAI_API_KEY`. In Step 4, BEFORE computing metrics, configure the evaluator LLM + embeddings to your local Ollama (wrap `ChatOllama` / `OllamaEmbeddings` via `LangchainLLMWrapper` / `LangchainEmbeddingsWrapper`) or the free Gemini tier; verify metrics compute with NO OpenAI key set. Pick ONE library (Ragas) to keep setup small.
- **Demo model:** Ollama can't run on a free HF CPU Space — deploy the demo with the free Gemini tier; keep local models for dev/eval only.

---

## 1. Objective
Ask questions over a real document corpus and get answers **with citations** to the source chunks — plus a proper **evaluation harness** (RAGAS / DeepEval) that reports faithfulness, answer-relevancy, and context-precision, and shows a measurable improvement from one retrieval upgrade (e.g., reranking or hybrid search).

## 2. Why this project
RAG is now *baseline-expected* for AI/ML roles, so doing plain RAG doesn't differentiate you — **adding evaluation does.** Reporting "context precision 0.61 → 0.84 after adding a cross-encoder reranker" is exactly the rigor hiring managers say separates an engineer from a tutorial-follower. It also pairs with your BI instinct for measuring things.

## 3. Make it uniquely yours (customization)
- **Pick a corpus you can speak to:** University of Genova course material / lecture notes, Italian public-administration docs (e.g., Comune di Genova or a ministry's public PDFs), or a set of AI papers you're reading for your thesis. The local/Italian angle matches your EU target.
- **Bilingual retrieval** (English + Italian) — a small, memorable edge.
- **Cite like an analyst:** every answer lists the exact source (file + page/section) so it's verifiable — frame it as "no hallucinated claims without a citation."

## 4. Tech stack & prerequisites (free)
- Python 3.11. **LlamaIndex** (or LangChain) for ingestion/retrieval. **ChromaDB** or FAISS.
- **Ollama** (`llama3.1:8b`) for generation + `nomic-embed-text` or `sentence-transformers` for embeddings (all local/free).
- **Eval:** `ragas` and/or `deepeval` (both Apache-2.0). A small judge model via the free Gemini tier *or* local model.
- **UI:** Gradio (integrates natively with HF Spaces). pytest, ruff, mypy.

## 5. Reference repos & originality approach
**Approach: STUDY + safe FORK of one folder.**
- `patchy631/ai-engineering-hub` — MIT, ~35.5k★ — **OK to fork one project folder and heavily extend** (attribute it). Good RAG starting structure.
- `NirDiamant/RAG_Techniques` — ⚠️ non-commercial license — **study only**; reimplement techniques (HyDE, fusion, reranking) in your own code.
- `run-llama/llama_index` (MIT) and `confident-ai/deepeval` (Apache), `explodinggradients/ragas` (Apache) — use as **dependencies** (configure their judge LLM + embeddings locally — see section 0).
- `infiniflow/ragflow` (Apache) — study as the "what production RAG looks like" reference; cite as your benchmark target.

## 6. Corpus
Your chosen PDFs/docs in `data/corpus/` (keep it modest, ~30–100 docs). Don't commit copyrighted PDFs; commit a small sample or a download script + a note on licensing.

## 7. Architecture
```
docs ─► loaders ─► chunker ─► embeddings ─► vector store ─► retriever ─┐
                                                          (+ reranker) │
question ───────────────────────────────────────────────────────────►│
                                                                       ▼
                                            prompt(context+question) ─► LLM ─► answer + citations
                                                                       │
                               eval set (Q, ground-truth) ─► RAGAS/DeepEval metrics
```

## 8. Repository structure
```
rag-with-eval/
├── README.md  CLAUDE.md  pyproject.toml  .env.example  LICENSE
├── app.py                      # Gradio UI
├── src/rag/
│   ├── ingest.py               # load + chunk
│   ├── index.py                # embed + vector store
│   ├── retrieve.py             # base retriever + optional reranker/hybrid
│   ├── generate.py             # answer with citations
│   └── pipeline.py             # end-to-end query()
├── eval/
│   ├── testset.yaml            # questions + ground-truth answers/contexts
│   └── run_eval.py             # RAGAS/DeepEval → metrics table
├── tests/
└── data/corpus/                # your docs (or a fetch script)
```

## 9. CLAUDE.md (paste as `./CLAUDE.md`)
```markdown
# Project: RAG over documents with evaluation

## What this is
A Gradio RAG app over a real document corpus that answers WITH citations, plus a RAGAS/DeepEval
harness reporting faithfulness, answer-relevancy, context-precision, and showing improvement from
one retrieval upgrade (reranking or hybrid search).

## Stack
Python 3.11 · LlamaIndex · ChromaDB/FAISS · Ollama (llama3.1:8b) + nomic-embed-text ·
ragas + deepeval (Apache) · Gradio · pytest · ruff · mypy.

## Commands
- App: `python app.py`        - Ingest/index: `python -m src.rag.ingest && python -m src.rag.index`
- Tests: `pytest -q`  Lint: `ruff check .`  Types: `mypy src`  - Eval: `python eval/run_eval.py`

## Conventions
- Every answer MUST return citations (source file + page/section). No citation → it's a bug.
- Retrieval components are swappable (base vs reranked) via config so we can A/B them in eval.
- Pin deps. Secrets in .env. Don't commit copyrighted PDFs — use a sample or fetch script.

## Done per step
Step's verify command passes → conventional commit.
```

## 10. Step-by-step build plan
### Step 1 — Scaffold + ingestion  ·  Day 1
**Context:** Need clean ingestion before anything else.
- [ ] Repo skeleton (section 8), pinned deps, `.gitignore`, MIT `LICENSE`.
- [ ] `ingest.py`: load PDFs/text, chunk (`RecursiveCharacterTextSplitter`, size≈1024, overlap≈200), keep metadata (source, page).
**Verify:** `pytest tests/test_ingest.py` — chunks carry source+page; a 10-page sample yields the expected chunk count.
**Commit:** `feat: document loaders + chunking with source metadata`

### Step 2 — Index + base retrieval  ·  Day 2
- [ ] `index.py`: embed chunks → Chroma. `retrieve.py`: top-k cosine retrieval.
**Verify:** a known question retrieves the chunk that contains the answer in top-k (assert in a test).
**Commit:** `feat: embeddings, vector store, base top-k retrieval`

### Step 3 — Answer with citations  ·  Day 3
- [ ] `generate.py` + `pipeline.py`: prompt the LLM with retrieved context; force it to answer and **cite** the sources it used; if no relevant context, say so (no hallucination).
**Verify:** run 5 questions; each answer includes citations resolvable to real chunks; an out-of-corpus question returns "not found."
**Commit:** `feat: cited answer generation pipeline`

### Step 4 — Evaluation harness (the centerpiece)  ·  Day 4
**Context:** This is what makes the repo stand out.
- [ ] `eval/testset.yaml`: ~20 Q + ground-truth. `run_eval.py`: compute RAGAS/DeepEval metrics (faithfulness, answer-relevancy, context-precision) for the base pipeline.
- [ ] Implement ONE upgrade — a cross-encoder **reranker** or **hybrid (BM25+dense)** search — behind a config flag. Re-run eval.
**Verify:** `python eval/run_eval.py` prints a metrics table for base vs upgraded; the upgrade improves at least one metric. Record the numbers.
**Commit:** `feat: RAGAS/DeepEval harness + reranker showing measurable gain`

### Step 5 — UI, README, deploy  ·  Day 5
- [ ] `app.py` Gradio: question box, answer + expandable citations, a toggle for base vs reranked, latency display.
- [ ] README from section 12 with the before/after metric table + demo GIF. ruff/mypy clean.
- [ ] Deploy to HF Spaces (Gradio).
**Verify:** public Space answers a question with citations; metric table is in the README.
**Commit:** `feat: Gradio UI + docs + HF Spaces demo`; tag `v1.0`.

## 11. Evaluation & testing
- Headline metrics: faithfulness / answer-relevancy / context-precision, base vs upgraded, in a table.
- Tests: retrieval relevance, citation presence, "no-context → no-answer" behavior.

## 12. README outline
One-liner + demo link + the **metric improvement table** up top → Problem (trustworthy answers over a real corpus) → Approach (pipeline + the upgrade) + architecture diagram → Results (metrics, screenshots) → Run it (3 commands) → Limitations → Credits (LlamaIndex, RAGAS/DeepEval; "RAG techniques studied from NirDiamant — reimplemented").

## 13. Demo deployment
HF Spaces, Gradio SDK, free CPU. If using a local model is too heavy for the free tier, switch the hosted demo to a free-tier hosted model and note it in the README.

## 14. Git & commit cadence
5 commits across the week; tag `v1.0`.

## 15. Run-in-Claude-Code prompts
- `Read @PLAN.md and @CLAUDE.md. Do Step 4 only. First write eval/testset.yaml with 20 Q+ground-truth, then implement run_eval.py with RAGAS, run it on the base pipeline, and show the metrics before we add the reranker.`
- After base metrics: `Now add a cross-encoder reranker behind a config flag, re-run eval, and print a base-vs-reranked table.`

## 16. Interview talking points
- "How do you know your RAG is good?" → your metrics + the A/B improvement.
- "How do you prevent hallucination?" → citations + no-context guard + faithfulness metric.
- "What would you do at scale?" → ragflow-style parsing, hybrid search, eval-in-CI.

## 17. Stretch goals
- Eval-in-CI (GitHub Action runs the harness on every push). · Hybrid + rerank combined. · Conversational memory. · Italian/English. · Swap Chroma → pgvector.

## 18. Definition of done
Public Space answers with citations; README shows a real metric improvement table; tests/lint/types pass; clean commit history; MIT license + attributions.

## 19. References
ai-engineering-hub (MIT) · RAG_Techniques (non-commercial — study only) · LlamaIndex (MIT) · RAGAS / DeepEval (Apache) · ragflow (Apache) · HF Spaces Gradio docs.
