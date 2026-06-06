"""Step 4 verify: the harness runs base vs upgraded over the test set and the guard holds.

We assert the harness produces sane, in-range metrics and that out-of-corpus questions
are refused. The actual base-vs-upgraded improvement is recorded in eval/results/ by
running `python eval/run_eval.py` (data-dependent, so not asserted as a brittle inequality).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from rag.config import Config
from rag.metrics import aggregate, evaluate_item
from rag.pipeline import RAGPipeline
from rag.retrieve import Retriever

ROOT = Path(__file__).resolve().parents[1]


def _run_variant(cfg: Config, shared: RAGPipeline, testset: list[dict]):
    pipe = RAGPipeline(
        cfg, shared.embedder, shared.llm,
        store=shared.store,
        retriever=Retriever.from_config(cfg, shared.embedder, shared.store),
    )
    metrics = [evaluate_item(item, pipe.query(item["question"]), shared.embedder) for item in testset]
    return aggregate(metrics)


def test_harness_base_vs_reranked(built_pipeline, offline_config):
    testset = yaml.safe_load((ROOT / "eval" / "testset.yaml").read_text("utf-8"))
    assert len(testset) >= 20

    base = _run_variant(offline_config.with_overrides(reranker="none"), built_pipeline, testset)
    reranked = _run_variant(
        offline_config.with_overrides(reranker="lexical"), built_pipeline, testset
    )

    for report in (base, reranked):
        assert 0.0 <= report["context_precision"] <= 1.0
        assert 0.0 <= report["faithfulness"] <= 1.0
        assert report["refusal_accuracy"] == 1.0        # every out-of-corpus question refused
        assert report["answer_rate"] >= 0.8             # most factual questions answered

    # Both variants should retrieve the gold source for most questions.
    assert base["context_recall"] >= 0.8
    assert reranked["context_recall"] >= 0.8
