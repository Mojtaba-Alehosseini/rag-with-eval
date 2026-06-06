"""Step 3 verify: cited answers for in-corpus questions; refusal for out-of-corpus ones."""

from __future__ import annotations

import re

from rag.prompts import NO_ANSWER

_MARKER = re.compile(r"\[\d+\]")


def test_answer_includes_resolvable_citations(built_pipeline):
    res = built_pipeline.query("What does the BM25 ranking function score a document by?")
    assert res.answered
    assert res.citations, "an answered query must carry citations"
    assert _MARKER.search(res.answer), "answer text must contain a [n] citation marker"
    assert any(c.cited for c in res.citations)
    for c in res.citations:
        assert c.source.endswith(".md")
        assert c.page >= 1
        assert c.snippet


def test_citation_points_at_correct_source(built_pipeline):
    res = built_pipeline.query("What dataset is the cross-encoder reranker trained on?")
    assert res.answered
    assert any(c.source == "05_reranking.md" for c in res.citations)


def test_out_of_corpus_is_refused(built_pipeline):
    res = built_pipeline.query("What is the capital of France?")
    assert not res.answered
    assert res.answer == NO_ANSWER
    assert res.citations == []


def test_italian_question_answers_from_italian_doc(built_pipeline):
    res = built_pipeline.query("Che cos'è il RAG?")
    assert res.answered
    assert any(c.source == "08_rag_in_italiano.md" for c in res.citations)


def test_latency_and_providers_recorded(built_pipeline):
    res = built_pipeline.query("What is HNSW and when is it used?")
    assert res.latency_ms >= 0.0
    assert res.providers["llm"] == "extractive"
    assert res.providers["embed"] == "hashing"
