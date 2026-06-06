"""Reranker providers (the retrieval *upgrade* the eval harness A/B-tests).

``CrossEncoderReranker`` is the real cross-encoder (sentence-transformers).
``LexicalReranker`` is the deterministic offline fallback: it re-scores the
candidate set with an exact token-overlap signal that is complementary to the
(hashing/dense) first-stage retriever, so reranking produces a real, measurable
re-ordering even with no heavy deps installed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from rag.textutil import tokenize

if TYPE_CHECKING:
    from rag.config import Config
    from rag.models import Retrieved


class Reranker:
    """Base reranker. Subclasses implement :meth:`rerank`."""

    name: str = "base"

    def rerank(self, query: str, candidates: list[Retrieved], top_n: int) -> list[Retrieved]:
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    """Real cross-encoder reranking (query, passage) -> relevance score."""

    name = "cross-encoder"

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers not installed. Run: pip install -e \".[heavy]\"."
            ) from exc
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[Retrieved], top_n: int) -> list[Retrieved]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs)
        order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)
        return [
            _with(candidates[i], score=float(scores[i]), method="rerank") for i in order[:top_n]
        ]


class LexicalReranker(Reranker):
    """Deterministic offline reranker.

    Re-scores candidates with a BM25-style exact-overlap signal computed over the
    candidate set itself (idf from the candidates, sublinear tf, length norm).
    Because it uses exact tokens rather than hashed dense vectors, it routinely
    corrects the ordering of the first-stage retriever.
    """

    name = "lexical"

    def rerank(self, query: str, candidates: list[Retrieved], top_n: int) -> list[Retrieved]:
        if not candidates:
            return []
        q_terms = set(tokenize(query))
        docs_tokens = [tokenize(c.chunk.text) for c in candidates]
        n = len(candidates)
        avg_len = sum(len(d) for d in docs_tokens) / n if n else 1.0
        # Document frequency of each query term within the candidate set.
        df = {t: sum(1 for d in docs_tokens if t in d) for t in q_terms}
        k1, b = 1.5, 0.75

        scores = []
        for d in docs_tokens:
            dl = len(d) or 1
            tf: dict[str, int] = {}
            for tok in d:
                if tok in q_terms:
                    tf[tok] = tf.get(tok, 0) + 1
            s = 0.0
            for t, f in tf.items():
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avg_len))
            scores.append(s)

        order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        return [_with(candidates[i], score=scores[i], method="rerank") for i in order[:top_n]]


def _with(retrieved: Retrieved, *, score: float, method: str) -> Retrieved:
    from rag.models import Retrieved as R

    return R(chunk=retrieved.chunk, score=score, method=method)


def get_reranker(config: Config) -> Reranker | None:
    """Return a reranker, or None when reranking is disabled.

    For ``cross-encoder`` we use the real model if sentence-transformers is
    importable, otherwise the deterministic lexical reranker.
    """
    setting = config.reranker
    if setting == "none":
        return None
    if setting == "cross-encoder":
        try:
            import sentence_transformers  # noqa: F401
        except Exception:
            return LexicalReranker()
        return CrossEncoderReranker(config.rerank_model)
    if setting == "lexical":
        return LexicalReranker()
    raise ValueError(f"Unknown reranker: {setting!r}")
