"""Offline, dependency-free proxy metrics for RAG evaluation.

These let ``eval/run_eval.py`` always produce a metrics table — even with no judge
LLM and no API key — so the harness is reproducible in CI. When a judge model is
configured, ``run_eval.py`` additionally computes the real RAGAS metrics.

Metric definitions (clearly labelled "proxy" in the output):
  * context_precision  — rank-sensitive average precision over retrieved chunks, using
                         source-file relevance labels from the test set. This is the
                         metric a reranker is expected to lift.
  * context_recall     — fraction of ground-truth source files present in the retrieval.
  * faithfulness       — fraction of answer sentences whose content tokens are grounded
                         in (recalled by) the retrieved context. Catches hallucination.
  * answer_relevancy   — cosine similarity between the answer and the question.
  * answer_similarity  — cosine similarity between the answer and the ground truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from rag.textutil import content_tokens, split_sentences

if TYPE_CHECKING:
    from rag.pipeline import QueryResult
    from rag.providers.embeddings import Embedder

_CITATION_RE = re.compile(r"\[\d+\]")


def _content_tokens(text: str) -> set[str]:
    return set(content_tokens(text))


def _cos(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def is_relevant(ground_truth_tokens: set[str], chunk_text: str, threshold: float = 0.5) -> bool:
    """A chunk is relevant if it contains most of the reference answer's content words.

    This is a lexical stand-in for RAGAS's reference-based context precision (does this
    passage support the gold answer?) — far more discriminating than "same source file",
    so a chunk that is merely on-topic but doesn't contain the answer counts as a miss.
    """
    if not ground_truth_tokens:
        return False
    chunk = set(content_tokens(chunk_text))
    return len(ground_truth_tokens & chunk) / len(ground_truth_tokens) >= threshold


def average_precision(relevances: list[bool]) -> float:
    """RAGAS-style context precision: mean of precision@k at each relevant rank."""
    total_relevant = sum(relevances)
    if total_relevant == 0:
        return 0.0
    hits = 0
    score = 0.0
    for k, rel in enumerate(relevances, start=1):
        if rel:
            hits += 1
            score += hits / k
    return score / total_relevant


def _faithfulness(answer: str, context_tokens: set[str], threshold: float = 0.6) -> float:
    clean = _CITATION_RE.sub("", answer)
    sentences = split_sentences(clean)
    if not sentences:
        return 0.0
    grounded = 0
    counted = 0
    for sent in sentences:
        toks = _content_tokens(sent)
        if not toks:
            continue
        counted += 1
        recall = len(toks & context_tokens) / len(toks)
        if recall >= threshold:
            grounded += 1
    return grounded / counted if counted else 0.0


@dataclass
class ItemMetrics:
    type: str
    answered: bool
    correct_guard: bool                 # negative refused, or factual answered
    context_precision: float | None = None
    context_recall: float | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    answer_similarity: float | None = None


def evaluate_item(item: dict, result: QueryResult, embedder: Embedder) -> ItemMetrics:
    """Compute proxy metrics for one (question, answer, retrieval) triple."""
    is_negative = item.get("type") == "negative"
    answered = result.answered

    if is_negative:
        # The only thing that matters for an out-of-corpus question: did we refuse?
        return ItemMetrics(type="negative", answered=answered, correct_guard=not answered)

    gt_tokens = set(content_tokens(item["ground_truth"]))
    relevances = [is_relevant(gt_tokens, r.chunk.text) for r in result.retrieved]
    context_precision = average_precision(relevances)
    context_recall = 1.0 if any(relevances) else 0.0

    faithfulness = answer_relevancy = answer_similarity = None
    if answered:
        context_tokens: set[str] = set()
        for r in result.retrieved:
            context_tokens |= _content_tokens(r.chunk.text)
        faithfulness = _faithfulness(result.answer, context_tokens)

        a_vec = embedder.embed_query(result.answer)
        q_vec = embedder.embed_query(item["question"])
        g_vec = embedder.embed_query(item["ground_truth"])
        answer_relevancy = max(0.0, _cos(a_vec, q_vec))
        answer_similarity = max(0.0, _cos(a_vec, g_vec))

    return ItemMetrics(
        type="factual",
        answered=answered,
        correct_guard=answered,
        context_precision=context_precision,
        context_recall=context_recall,
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        answer_similarity=answer_similarity,
    )


def _mean(values: list[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    return float(np.mean(nums)) if nums else None


def aggregate(items: list[ItemMetrics]) -> dict[str, float | None]:
    """Average per-item metrics into a single report row."""
    factual = [m for m in items if m.type == "factual"]
    negative = [m for m in items if m.type == "negative"]
    answered_factual = [m for m in factual if m.answered]

    guard_hits = sum(1 for m in items if m.correct_guard)
    return {
        "context_precision": _mean([m.context_precision for m in factual]),
        "context_recall": _mean([m.context_recall for m in factual]),
        "faithfulness": _mean([m.faithfulness for m in answered_factual]),
        "answer_relevancy": _mean([m.answer_relevancy for m in answered_factual]),
        "answer_similarity": _mean([m.answer_similarity for m in answered_factual]),
        "answer_rate": (len(answered_factual) / len(factual)) if factual else None,
        "refusal_accuracy": (
            sum(1 for m in negative if m.correct_guard) / len(negative) if negative else None
        ),
        "guard_accuracy": guard_hits / len(items) if items else None,
    }
