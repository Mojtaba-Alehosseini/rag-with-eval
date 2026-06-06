"""Offline proxy metrics: average precision, grounding/faithfulness, guard handling."""

from __future__ import annotations

import pytest

from rag.metrics import _faithfulness, average_precision, evaluate_item
from rag.pipeline import QueryResult
from rag.prompts import NO_ANSWER
from rag.providers.embeddings import HashingEmbedder


@pytest.mark.parametrize(
    "rels,expected",
    [
        ([True, False, True], (1.0 + 2 / 3) / 2),
        ([False, True], 0.5),
        ([True, True], 1.0),
        ([False, False], 0.0),
        ([], 0.0),
    ],
)
def test_average_precision(rels, expected):
    assert average_precision(rels) == pytest.approx(expected)


def test_faithfulness_grounded_vs_hallucinated():
    context = {"bm25", "ranking", "function", "terms", "corpus", "rare"}
    grounded = "BM25 is a ranking function that weights rare terms across the corpus. [1]"
    hallucinated = "The Eiffel Tower was completed in Paris in 1889. [1]"
    assert _faithfulness(grounded, context) > _faithfulness(hallucinated, context)
    assert _faithfulness(hallucinated, context) < 0.5


def test_negative_item_scored_on_guard_only():
    embedder = HashingEmbedder(dim=128)
    item = {"question": "What is the capital of France?", "ground_truth": NO_ANSWER,
            "ground_truth_sources": [], "type": "negative"}
    refused = QueryResult(question=item["question"], answer=NO_ANSWER, answered=False)
    m = evaluate_item(item, refused, embedder)
    assert m.type == "negative"
    assert m.correct_guard is True
    assert m.context_precision is None


def test_factual_item_metrics_in_range(built_pipeline):
    item = {
        "question": "What does the BM25 ranking function score a document by?",
        "ground_truth": "How often the query terms appear, weighted by how rare each term is.",
        "ground_truth_sources": ["04_retrieval.md"],
        "type": "factual",
    }
    result = built_pipeline.query(item["question"])
    m = evaluate_item(item, result, built_pipeline.embedder)
    assert m.answered
    assert 0.0 <= m.context_precision <= 1.0
    assert m.context_recall == 1.0                     # gold source retrieved
    assert m.faithfulness > 0.5                         # extractive answers are grounded
    assert 0.0 <= m.answer_relevancy <= 1.0
