"""Evaluation harness: base vs upgraded retrieval, with a metrics table.

    python eval/run_eval.py                      # offline proxy metrics (no key needed)
    python eval/run_eval.py --judge ragas        # add real RAGAS metrics (needs a judge)

It builds the index once (the index is upgrade-agnostic), then runs each retrieval
variant over eval/testset.yaml and prints a base-vs-upgraded table. Results are saved
to eval/results/ (CSV + a Markdown table to paste into the README + per-item JSON).

The RAGAS path is configured to a LOCAL judge (Ollama) or Gemini — never OpenAI — so it
runs with no OPENAI_API_KEY, per the gotcha in section 0 of the project spec.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

# Make `rag` importable whether or not the package was pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import Config  # noqa: E402
from rag.metrics import aggregate, evaluate_item  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.providers.llm import _ollama_reachable  # noqa: E402
from rag.retrieve import Retriever  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def _force_utf8_stdout() -> None:
    """Windows consoles default to cp1252; print UTF-8 so tables/symbols don't crash."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass

# The retrieval variants we A/B. The index is identical for all of them.
VARIANTS = {
    "base": {"reranker": "none", "hybrid": False},
    "reranked": {"reranker": "cross-encoder", "hybrid": False},
    "hybrid": {"reranker": "none", "hybrid": True},
    "hybrid+rerank": {"reranker": "cross-encoder", "hybrid": True},
}

METRIC_ORDER = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_similarity",
    "answer_rate",
    "refusal_accuracy",
    "guard_accuracy",
]


def load_testset(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text("utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Test set at {path} is empty or malformed.")
    return data


def run_variant(name: str, cfg: Config, shared: RAGPipeline, testset: list[dict]) -> tuple[dict, list]:
    """Run one retrieval variant, reusing the shared embedder/store/LLM."""
    pipe = RAGPipeline(
        cfg,
        shared.embedder,
        shared.llm,
        store=shared.store,
        retriever=Retriever.from_config(cfg, shared.embedder, shared.store),
    )
    item_metrics = []
    per_item_rows = []
    for item in testset:
        result = pipe.query(item["question"])
        m = evaluate_item(item, result, shared.embedder)
        item_metrics.append(m)
        per_item_rows.append(
            {
                "variant": name,
                "question": item["question"],
                "type": item.get("type"),
                "answered": result.answered,
                "answer": result.answer,
                "retrieved_sources": [r.chunk.source for r in result.retrieved],
                "gold_sources": item.get("ground_truth_sources"),
                "latency_ms": round(result.latency_ms, 1),
                # Underscore-prefixed: consumed by RAGAS, stripped before JSON dump.
                "_contexts": [r.chunk.text for r in result.retrieved],
                "_reference": item["ground_truth"],
                **{k: v for k, v in asdict(m).items() if k not in {"type", "answered"}},
            }
        )
    return aggregate(item_metrics), per_item_rows


def build_table(results: dict[str, dict]) -> pd.DataFrame:
    rounded = {
        name: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in agg.items()}
        for name, agg in results.items()
    }
    df = pd.DataFrame(rounded).T
    cols = [c for c in METRIC_ORDER if c in df.columns]
    return df[cols]


def df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub Markdown table (no tabulate dependency)."""
    header = "| variant | " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| --- | " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + str(idx) + " | " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for idx, row in df.iterrows()
    ]
    return "\n".join([header, sep, *rows]) + "\n"


def ragas_scores(cfg: Config, per_item_rows: list[dict]) -> dict | None:
    """Best-effort real RAGAS metrics with a LOCAL judge (never OpenAI). None on failure."""
    try:
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        try:
            from ragas import EvaluationDataset
        except Exception:
            from ragas.dataset_schema import EvaluationDataset
    except Exception as exc:
        print(f"  [ragas] not available ({exc}); install with: pip install -e \".[eval]\"")
        return None

    # Configure the judge to a local/free model so evaluate() needs no OpenAI key.
    try:
        if cfg.gemini_api_key:
            from langchain_google_genai import (
                ChatGoogleGenerativeAI,
                GoogleGenerativeAIEmbeddings,
            )

            judge_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=cfg.gemini_model))
            judge_emb = LangchainEmbeddingsWrapper(
                GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            )
        elif _ollama_reachable(cfg.ollama_host):
            from langchain_ollama import ChatOllama, OllamaEmbeddings

            judge_llm = LangchainLLMWrapper(
                ChatOllama(model=cfg.ollama_llm_model, base_url=cfg.ollama_host)
            )
            judge_emb = LangchainEmbeddingsWrapper(
                OllamaEmbeddings(model=cfg.ollama_embed_model, base_url=cfg.ollama_host)
            )
        else:
            print("  [ragas] no local judge (set GEMINI_API_KEY or start Ollama); skipping.")
            return None
    except Exception as exc:
        print(f"  [ragas] could not build a local judge ({exc}); skipping.")
        return None

    samples = [
        {
            "user_input": r["question"],
            "retrieved_contexts": r.get("_contexts", []),
            "response": r["answer"],
            "reference": r.get("_reference", ""),
        }
        for r in per_item_rows
        if r.get("type") == "factual"
    ]
    try:
        dataset = EvaluationDataset.from_list(samples)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=judge_llm,
            embeddings=judge_emb,
        )
        df = result.to_pandas()
        out: dict[str, float] = {}
        for name in ("faithfulness", "answer_relevancy", "context_precision"):
            if name in df.columns:
                out[name] = round(float(df[name].mean()), 3)
        return out
    except Exception as exc:
        print(f"  [ragas] evaluate() failed ({exc}); reporting offline proxy only.")
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAG evaluation harness.")
    parser.add_argument("--testset", default=str(HERE / "testset.yaml"))
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["base", "reranked", "hybrid"],
        choices=list(VARIANTS),
    )
    parser.add_argument("--judge", choices=["offline", "ragas"], default="offline")
    args = parser.parse_args(argv)
    _force_utf8_stdout()

    testset = load_testset(Path(args.testset))
    base_cfg = Config.from_env()

    print(f"Building index ({base_cfg.vector_store} store)…")
    shared = RAGPipeline.from_config(base_cfg).build()
    assert shared.store is not None
    print(
        f"  {shared.store.count()} chunks | embedder={shared.embedder.name} "
        f"| llm={shared.llm.name}\n"
    )

    results: dict[str, dict] = {}
    all_rows: list[dict] = []
    for name in args.variants:
        cfg = base_cfg.with_overrides(**VARIANTS[name])
        print(f"Running variant: {name}  ({VARIANTS[name]})")
        agg, rows = run_variant(name, cfg, shared, testset)
        results[name] = agg
        all_rows.extend(rows)

    table = build_table(results)
    print("\n=== Offline proxy metrics (base vs upgraded) ===")
    print(table.to_string())

    # Headline improvement.
    if "base" in results and "reranked" in results:
        b = results["base"]["context_precision"]
        r = results["reranked"]["context_precision"]
        if b is not None and r is not None:
            print(f"\ncontext_precision: base {b:.3f} -> reranked {r:.3f}  (delta {r - b:+.3f})")

    if args.judge == "ragas":
        print("\n=== RAGAS metrics (LLM judge) ===")
        for name in args.variants:
            scored = ragas_scores(base_cfg, [r for r in all_rows if r["variant"] == name])
            if scored:
                print(f"  {name}: {scored}")

    # Persist artifacts (strip underscore-prefixed RAGAS-only fields from the JSON).
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "metrics.csv")
    (RESULTS / "metrics.md").write_text(df_to_markdown(table), "utf-8")
    public_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_rows]
    (RESULTS / "per_item.json").write_text(
        json.dumps(public_rows, ensure_ascii=False, indent=2), "utf-8"
    )
    print(f"\nSaved: {RESULTS / 'metrics.csv'}, metrics.md, per_item.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
