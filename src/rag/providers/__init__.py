"""Pluggable providers (embeddings / LLM / reranker).

Each ``get_*`` factory resolves the ``"auto"`` setting to a concrete provider,
preferring the real heavy implementation when its dependency is importable and
falling back to a deterministic, dependency-free offline provider otherwise.
"""

from rag.providers.embeddings import Embedder, get_embedder
from rag.providers.llm import LLM, get_llm
from rag.providers.reranker import Reranker, get_reranker

__all__ = ["Embedder", "get_embedder", "LLM", "get_llm", "Reranker", "get_reranker"]
