"""Tests for the text utility helpers in rag.textutil."""

from __future__ import annotations

from rag.textutil import STOPWORDS, content_tokens, jaccard, split_sentences, tokenize


def test_tokenize_lowercases():
    assert tokenize("Hello World") == ["hello", "world"]


def test_tokenize_unicode_italian():
    tokens = tokenize("cos'è la semantica")
    assert "semantica" in tokens


def test_tokenize_numbers_and_words():
    tokens = tokenize("BM25 scores 3 documents")
    assert "bm25" in tokens
    assert "scores" in tokens


def test_content_tokens_removes_stopwords():
    tokens = content_tokens("what is the capital of France")
    assert "what" not in tokens
    assert "is" not in tokens
    assert "france" in tokens


def test_content_tokens_removes_single_chars():
    tokens = content_tokens("a b c ranking")
    assert "a" not in tokens
    assert "b" not in tokens
    assert "ranking" in tokens


def test_content_tokens_empty_string():
    assert content_tokens("") == []


def test_stopwords_not_empty():
    assert len(STOPWORDS) > 0


def test_stopwords_contains_english():
    for word in ("the", "a", "is", "and"):
        assert word in STOPWORDS


def test_stopwords_contains_italian():
    for word in ("il", "la", "di", "che"):
        assert word in STOPWORDS


def test_split_sentences_on_period():
    parts = split_sentences("First sentence. Second sentence.")
    assert len(parts) == 2
    assert parts[0] == "First sentence."


def test_split_sentences_on_newline():
    parts = split_sentences("Line one\nLine two")
    assert len(parts) == 2


def test_split_sentences_strips_whitespace():
    parts = split_sentences("  Hello world.  ")
    assert parts[0] == "Hello world."


def test_split_sentences_empty():
    assert split_sentences("") == []


def test_jaccard_identical():
    s = {"a", "b", "c"}
    assert jaccard(s, s) == 1.0


def test_jaccard_disjoint():
    assert jaccard({"a", "b"}, {"c", "d"}) == 0.0


def test_jaccard_partial():
    assert jaccard({"a", "b", "c"}, {"b", "c", "d"}) == 0.5  # inter=2, union=4


def test_jaccard_empty():
    assert jaccard(set(), {"a"}) == 0.0
    assert jaccard({"a"}, set()) == 0.0
