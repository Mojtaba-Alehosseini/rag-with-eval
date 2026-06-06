"""RAG over documents with cited answers and a real evaluation harness.

Public API:
    from rag import Config, RAGPipeline
    pipe = RAGPipeline.from_config(Config.from_env())
    pipe.build()                      # ingest + index the corpus
    result = pipe.query("...")        # -> QueryResult(answer, citations, ...)

``RAGPipeline`` / ``QueryResult`` are imported lazily (PEP 562) so that
``python -m rag.pipeline`` doesn't import the module twice (which triggers a
RuntimeWarning) and so ``import rag`` stays cheap.
"""

from typing import TYPE_CHECKING

from rag.config import Config

if TYPE_CHECKING:
    from rag.pipeline import QueryResult, RAGPipeline

__all__ = ["Config", "RAGPipeline", "QueryResult"]
__version__ = "1.0.0"


def __getattr__(name: str) -> object:
    if name in ("RAGPipeline", "QueryResult"):
        from rag import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
