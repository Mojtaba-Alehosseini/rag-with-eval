"""Small, dependency-free text helpers shared across providers and metrics."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
# Split on sentence-ending punctuation (keeping it) or on hard line breaks.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# Minimal EN+IT stopword set. Used to focus embeddings/metrics on content words so a
# question made mostly of glue words ("what is the capital of France?") doesn't match
# the corpus through shared stopwords and slip past the no-context guard.
STOPWORDS = frozenset(
    {
        # English
        "the", "a", "an", "of", "to", "in", "is", "are", "and", "or", "it", "its", "that",
        "this", "for", "on", "as", "by", "with", "be", "can", "does", "do", "how", "what",
        "which", "when", "into", "from", "at", "they", "their", "than", "such", "so", "was",
        "were", "we", "you", "your", "i", "but", "not", "no", "if", "then", "there", "here",
        # Italian
        "il", "lo", "la", "gli", "le", "un", "uno", "una", "di", "che", "è", "e", "ed",
        "o", "per", "con", "su", "al", "del", "della", "dei", "delle", "se", "cosa",
        "cos", "come", "quando", "non", "ai", "agli", "alle", "ha", "si", "ne",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. ``\\w`` is Unicode-aware, so Italian works too."""
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens with stopwords and single characters removed (content words only)."""
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def split_sentences(text: str) -> list[str]:
    """Heuristic sentence/line splitter good enough for txt/markdown corpora."""
    parts = _SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard overlap of two token sets (0..1)."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0
