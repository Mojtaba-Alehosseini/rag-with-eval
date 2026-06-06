"""Retrieval: base dense top-k, plus two swappable upgrades.

  * Hybrid search  — fuse dense + BM25 (lexical) rankings with Reciprocal Rank Fusion.
  * Reranking      — re-score the candidate pool with a (cross-encoder or lexical) reranker.

Both are config-driven so the eval harness can A/B them against the base retriever.
A small deterministic BM25 ships here so hybrid works with no extra dependencies.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from rag.models import Retrieved
from rag.providers.reranker import get_reranker
from rag.textutil import tokenize

if TYPE_CHECKING:
    from rag.config import Config
    from rag.index import VectorStore
    from rag.providers.embeddings import Embedder
    from rag.providers.reranker import Reranker


class BM25:
    """Minimal deterministic BM25 Okapi over an in-memory corpus."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(d) for d in documents]
        self.n = len(self.docs)
        self.avg_len = (sum(len(d) for d in self.docs) / self.n) if self.n else 0.0
        self.df: dict[str, int] = {}
        for doc in self.docs:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1
        self.tf: list[dict[str, int]] = []
        for doc in self.docs:
            counts: dict[str, int] = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            self.tf.append(counts)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        q_terms = tokenize(query)
        scores = []
        for i, counts in enumerate(self.tf):
            dl = len(self.docs[i]) or 1
            s = 0.0
            for term in q_terms:
                f = counts.get(term, 0)
                if f == 0:
                    continue
                s += self._idf(term) * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
                )
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


def reciprocal_rank_fusion(
    rankings: list[list[Retrieved]], c: int = 60
) -> list[Retrieved]:
    """Fuse multiple rankings by RRF (rank-based, no score normalisation needed)."""
    fused: dict[str, float] = {}
    by_id: dict[str, Retrieved] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            cid = item.chunk.chunk_id
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (c + rank)
            by_id.setdefault(cid, item)
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [Retrieved(chunk=by_id[cid].chunk, score=score, method="hybrid") for cid, score in ordered]


class Retriever:
    """Config-driven retriever combining dense, optional hybrid, optional rerank."""

    def __init__(
        self,
        config: Config,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self._chunks = None
        self._bm25 = None
        # Top dense cosine similarity from the most recent retrieve() call. The
        # no-context guard reads this because it is a stable, calibrated scale
        # (unlike post-rerank scores).
        self.last_dense_top_score: float = 0.0
        if config.hybrid:
            self._chunks = store.all_chunks()
            self._bm25 = BM25([c.text for c in self._chunks])

    @classmethod
    def from_config(cls, config: Config, embedder: Embedder, store: VectorStore) -> Retriever:
        return cls(config, embedder, store, reranker=get_reranker(config))

    def _pool_size(self) -> int:
        if self.reranker is not None or self.config.hybrid:
            pool = max(self.config.top_k, self.config.rerank_top_n) * 4
            return min(pool, self.store.count()) or self.config.top_k
        return self.config.top_k

    def retrieve(self, query: str) -> list[Retrieved]:
        pool = self._pool_size()
        q_emb = self.embedder.embed_query(query)
        dense = self.store.query(q_emb, pool)
        self.last_dense_top_score = dense[0].score if dense else 0.0

        candidates = dense
        if self.config.hybrid and self._bm25 is not None and self._chunks is not None:
            bm = self._bm25.search(query, pool)
            bm_ret = [
                Retrieved(chunk=self._chunks[i], score=score, method="bm25") for i, score in bm
            ]
            candidates = reciprocal_rank_fusion([dense, bm_ret])[:pool]

        if self.reranker is not None:
            return self.reranker.rerank(query, candidates, self.config.rerank_top_n)
        return candidates[: self.config.top_k]
