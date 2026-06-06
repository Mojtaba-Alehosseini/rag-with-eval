"""The deterministic hashing embedder: reproducible, normalized, lexically meaningful."""

from __future__ import annotations

import numpy as np

from rag.providers.embeddings import HashingEmbedder


def test_deterministic_and_normalized():
    e1, e2 = HashingEmbedder(dim=256), HashingEmbedder(dim=256)
    v1 = e1.embed_query("hybrid search fuses BM25 and dense retrieval")
    v2 = e2.embed_query("hybrid search fuses BM25 and dense retrieval")
    assert np.allclose(v1, v2)                       # deterministic across instances/runs
    assert v1.shape == (256,)
    assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-5


def test_empty_text_is_zero_vector():
    e = HashingEmbedder(dim=64)
    assert not np.any(e.embed_query("   "))


def test_similar_texts_score_higher():
    e = HashingEmbedder(dim=512)
    a = e.embed_query("BM25 is a lexical ranking function from information retrieval")
    b = e.embed_query("BM25 scores documents by term frequency and rarity")
    c = e.embed_query("the boiling point of water in Fahrenheit")
    assert float(a @ b) > float(a @ c)               # vectors are unit-norm -> dot == cosine


def test_batch_matches_single():
    e = HashingEmbedder(dim=128)
    texts = ["context precision", "cross encoder reranker"]
    batch = e.embed_texts(texts)
    assert batch.shape == (2, 128)
    assert np.allclose(batch[0], e.embed_query(texts[0]))
