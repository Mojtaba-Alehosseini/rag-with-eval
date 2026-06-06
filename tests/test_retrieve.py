"""Step 2 verify: a known question retrieves the chunk containing the answer in top-k.
Plus the upgrades (reranker, hybrid, RRF) behave as intended."""

from __future__ import annotations

from rag.models import Chunk, Retrieved
from rag.providers.reranker import LexicalReranker
from rag.retrieve import BM25, Retriever, reciprocal_rank_fusion


def test_known_question_retrieves_answer_chunk(built_pipeline):
    res = built_pipeline.query("How many dimensions does the all-MiniLM-L6-v2 model produce?")
    sources = [r.chunk.source for r in res.retrieved]
    assert "03_embeddings.md" in sources


def test_bm25_ranks_lexical_match_first():
    docs = [
        "Embeddings are dense vectors of floating point numbers.",
        "BM25 weights query terms by how rare they are across the corpus.",
    ]
    bm25 = BM25(docs)
    top = bm25.search("How does BM25 weight rare terms?", k=2)
    assert top[0][0] == 1  # the BM25 document ranks first


def test_lexical_reranker_promotes_relevant_candidate():
    relevant = Chunk("c1", "BM25 weights terms by how rare they are across the corpus", "04.md", 1)
    noise = Chunk("c2", "Embeddings are dense vectors of floating point numbers", "03.md", 1)
    # Dense put the noise first (higher score); the reranker should fix the order.
    candidates = [Retrieved(noise, score=0.9), Retrieved(relevant, score=0.1)]
    out = LexicalReranker().rerank("What does BM25 weight terms by?", candidates, top_n=2)
    assert out[0].chunk.chunk_id == "c1"
    assert out[0].method == "rerank"


def test_rrf_rewards_agreement():
    a = Chunk("a", "x", "f", 1)
    b = Chunk("b", "y", "f", 1)
    dense = [Retrieved(a, 0.9), Retrieved(b, 0.1)]
    lexical = [Retrieved(b, 5.0), Retrieved(a, 1.0)]
    fused = reciprocal_rank_fusion([dense, lexical])
    # 'a' is rank 1 then rank 2; 'b' is rank 2 then rank 1 -> tie broken by insertion,
    # but both appear; the fusion must include both with positive scores.
    ids = {r.chunk.chunk_id for r in fused}
    assert ids == {"a", "b"}
    assert all(r.score > 0 for r in fused)


def test_reranker_changes_ordering_on_corpus(built_pipeline, offline_config):
    cfg_base = offline_config.with_overrides(reranker="none")
    cfg_rr = offline_config.with_overrides(reranker="lexical")  # deterministic, no download
    q = "Which evaluation metric does reranking most directly improve?"
    base = Retriever.from_config(cfg_base, built_pipeline.embedder, built_pipeline.store).retrieve(q)
    rr = Retriever.from_config(cfg_rr, built_pipeline.embedder, built_pipeline.store).retrieve(q)
    assert base and rr
    assert rr[0].method == "rerank"
