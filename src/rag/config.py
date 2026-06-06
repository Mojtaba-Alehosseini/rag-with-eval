"""Central configuration.

Everything is driven from a single immutable :class:`Config` built from environment
variables (optionally loaded from a local ``.env``). Defaults are chosen so the project
runs fully offline and deterministically with no API key and no model download.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path


def _load_dotenv() -> None:
    """Best-effort load of a local .env. Never fails if python-dotenv is absent."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(override=False)


def _get(name: str, default: str) -> str:
    val = os.environ.get(name)
    return default if val is None or val == "" else val


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration for the whole pipeline + eval harness."""

    # Providers ("auto" resolves lazily at construction time in the factories).
    llm_provider: str = "auto"          # auto | offline | ollama | gemini
    embed_provider: str = "auto"        # auto | hashing | sentence-transformers | ollama
    vector_store: str = "auto"          # auto | numpy | chroma

    # Retrieval / upgrade knobs — these are the A/B switches the eval harness flips.
    reranker: str = "none"              # none | cross-encoder
    hybrid: bool = False                # fuse BM25 (lexical) with dense retrieval
    top_k: int = 5                      # candidates retrieved before reranking
    rerank_top_n: int = 5              # kept after reranking
    # No-context guard. Semantic embedders use a cosine floor; the lexical (hashing)
    # embedder uses query-term recall, which is robust to length-diluted cosines.
    no_answer_threshold: float = 0.18   # cosine floor (semantic embedders)
    no_answer_recall: float = 0.40      # min query-term recall (lexical embedder)

    # Chunking (LlamaIndex SentenceSplitter semantics).
    chunk_size: int = 1024
    chunk_overlap: int = 200

    # Paths.
    corpus_dir: Path = Path("data/corpus")
    persist_dir: Path = Path(".rag_index")

    # Model names.
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ollama_host: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.1:8b"
    ollama_embed_model: str = "nomic-embed-text"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # Embedding dimension used by the deterministic hashing embedder. Larger ->
    # fewer hash collisions -> a cleaner cosine signal for the no-context guard.
    hashing_dim: int = 2048

    @classmethod
    def from_env(cls) -> Config:
        """Build a Config from environment variables (loading .env first)."""
        _load_dotenv()
        return cls(
            llm_provider=_get("RAG_LLM_PROVIDER", "auto").lower(),
            embed_provider=_get("RAG_EMBED_PROVIDER", "auto").lower(),
            vector_store=_get("RAG_VECTOR_STORE", "auto").lower(),
            reranker=_get("RAG_RERANKER", "none").lower(),
            hybrid=_get_bool("RAG_HYBRID", False),
            top_k=_get_int("RAG_TOP_K", 5),
            rerank_top_n=_get_int("RAG_RERANK_TOP_N", 5),
            no_answer_threshold=_get_float("RAG_NO_ANSWER_THRESHOLD", 0.18),
            no_answer_recall=_get_float("RAG_NO_ANSWER_RECALL", 0.40),
            chunk_size=_get_int("RAG_CHUNK_SIZE", 1024),
            chunk_overlap=_get_int("RAG_CHUNK_OVERLAP", 200),
            corpus_dir=Path(_get("RAG_CORPUS_DIR", "data/corpus")),
            persist_dir=Path(_get("RAG_PERSIST_DIR", ".rag_index")),
            embed_model=_get("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            rerank_model=_get("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            ollama_host=_get("OLLAMA_HOST", "http://localhost:11434"),
            ollama_llm_model=_get("RAG_OLLAMA_LLM_MODEL", "llama3.1:8b"),
            ollama_embed_model=_get("RAG_OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            gemini_api_key=_get("GEMINI_API_KEY", ""),
            gemini_model=_get("RAG_GEMINI_MODEL", "gemini-1.5-flash"),
        )

    def with_overrides(self, **kwargs: object) -> Config:
        """Return a copy with some fields replaced (used by eval to A/B variants)."""
        return replace(self, **kwargs)  # type: ignore[arg-type]
