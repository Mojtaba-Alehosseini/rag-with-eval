"""Shared data structures passed between pipeline stages.

Kept in one module (instead of stdlib-shadowing ``types.py``) so ingest/index/
retrieve/generate can share them without circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A unit of text to embed/retrieve, with provenance for citations."""

    chunk_id: str
    text: str
    source: str          # file name, e.g. "intro_to_rag.md"
    page: int            # 1-based page (PDF) or logical section index (txt/md)
    metadata: dict = field(default_factory=dict)


@dataclass
class Retrieved:
    """A chunk returned by retrieval, with the score that ranked it."""

    chunk: Chunk
    score: float
    # How it was found, for transparency in the UI / debugging:
    # "dense", "bm25", "hybrid", or "rerank".
    method: str = "dense"


@dataclass
class Citation:
    """A numbered source attached to an answer ([1], [2], ...)."""

    index: int           # the [n] marker used in the answer text
    source: str
    page: int
    snippet: str
    score: float
    cited: bool = False  # True if the answer text actually referenced [index]
