"""End-to-end RAG pipeline + command-line entry point.

    python -m rag.pipeline build                 # ingest + index data/corpus/
    python -m rag.pipeline query "a question"     # answer with citations
    python -m rag.pipeline query "q" --rerank cross-encoder --hybrid
    python -m rag.pipeline info                    # show config + index status

The retrieval *upgrades* (rerank / hybrid) act at query time, so flipping them
never requires re-indexing — exactly what lets the eval harness A/B them.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING

from rag.config import Config
from rag.generate import generate_answer
from rag.index import build_index, load_index
from rag.ingest import ingest
from rag.models import Citation, Retrieved
from rag.providers.embeddings import get_embedder
from rag.providers.llm import get_llm
from rag.retrieve import Retriever

if TYPE_CHECKING:
    from rag.index import VectorStore
    from rag.providers.embeddings import Embedder
    from rag.providers.llm import LLM


@dataclass
class QueryResult:
    question: str
    answer: str
    answered: bool
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[Retrieved] = field(default_factory=list)
    latency_ms: float = 0.0
    providers: dict[str, str] = field(default_factory=dict)

    def pretty(self) -> str:
        lines = [f"Q: {self.question}", "", f"A: {self.answer}", ""]
        if self.citations:
            lines.append("Sources:")
            for c in self.citations:
                mark = "*" if c.cited else " "
                lines.append(f"  [{c.index}]{mark} {c.source} (p.{c.page})  ~{c.score}")
                lines.append(f"       {c.snippet}")
        lines.append("")
        prov = ", ".join(f"{k}={v}" for k, v in self.providers.items())
        lines.append(f"({prov}; {self.latency_ms:.0f} ms)")
        return "\n".join(lines)


class RAGPipeline:
    """Holds the configured providers + store and runs queries."""

    def __init__(
        self,
        config: Config,
        embedder: Embedder,
        llm: LLM,
        store: VectorStore | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.llm = llm
        self.store = store
        self.retriever = retriever

    @classmethod
    def from_config(cls, config: Config) -> RAGPipeline:
        return cls(config, get_embedder(config), get_llm(config))

    def build(self) -> RAGPipeline:
        """Ingest + index the corpus (idempotent: overwrites the index)."""
        chunks = ingest(self.config)
        if not chunks:
            raise RuntimeError(f"No chunks produced from corpus at {self.config.corpus_dir}.")
        self.store = build_index(self.config, chunks, self.embedder)
        self.retriever = Retriever.from_config(self.config, self.embedder, self.store)
        return self

    def load(self) -> RAGPipeline:
        """Load a previously built index."""
        self.store = load_index(self.config, self.embedder)
        self.retriever = Retriever.from_config(self.config, self.embedder, self.store)
        return self

    def _ensure_ready(self) -> None:
        if self.retriever is None or self.store is None:
            self.load()

    def query(self, question: str) -> QueryResult:
        self._ensure_ready()
        assert self.retriever is not None
        t0 = perf_counter()
        retrieved = self.retriever.retrieve(question)
        result = generate_answer(
            self.config,
            self.llm,
            question,
            retrieved,
            self.retriever.last_dense_top_score,
            lexical_embedder=(self.embedder.name == "hashing"),
        )
        latency_ms = (perf_counter() - t0) * 1000.0
        return QueryResult(
            question=question,
            answer=result.answer,
            answered=result.answered,
            citations=result.citations,
            retrieved=retrieved,
            latency_ms=latency_ms,
            providers={
                "llm": self.llm.name,
                "embed": self.embedder.name,
                "rerank": self.config.reranker,
                "hybrid": str(self.config.hybrid).lower(),
            },
        )


# --------------------------------------------------------------------------- CLI


def _add_upgrade_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rerank", choices=["none", "cross-encoder", "lexical"], default=None)
    p.add_argument("--hybrid", action="store_true", default=None)
    p.add_argument("--top-k", type=int, default=None)


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    overrides: dict[str, object] = {}
    if getattr(args, "rerank", None) is not None:
        overrides["reranker"] = args.rerank
    if getattr(args, "hybrid", None):
        overrides["hybrid"] = True
    if getattr(args, "top_k", None) is not None:
        overrides["top_k"] = args.top_k
    return config.with_overrides(**overrides) if overrides else config


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # UTF-8 output on Windows consoles
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass
    parser = argparse.ArgumentParser(prog="rag", description="RAG with citations + evaluation.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="Ingest data/corpus/ and build the vector index.")

    q = sub.add_parser("query", help="Answer a question with citations.")
    q.add_argument("question", help="The question to answer.")
    _add_upgrade_flags(q)

    sub.add_parser("info", help="Show resolved config and index status.")

    args = parser.parse_args(argv)
    config = _apply_overrides(Config.from_env(), args)

    if args.cmd == "build":
        pipe = RAGPipeline.from_config(config).build()
        assert pipe.store is not None
        print(
            f"Built index: {pipe.store.count()} chunks "
            f"(store={config.vector_store}, embedder={pipe.embedder.name}) -> {config.persist_dir}"
        )
        return 0

    if args.cmd == "query":
        pipe = RAGPipeline.from_config(config)
        print(pipe.query(args.question).pretty())
        return 0

    if args.cmd == "info":
        print(f"corpus_dir   : {config.corpus_dir}")
        print(f"persist_dir  : {config.persist_dir}")
        print(f"llm_provider : {config.llm_provider}")
        print(f"embed_provider: {config.embed_provider}")
        print(f"vector_store : {config.vector_store}")
        print(f"reranker     : {config.reranker}   hybrid: {config.hybrid}")
        print(f"top_k        : {config.top_k}   rerank_top_n: {config.rerank_top_n}")
        manifest = config.persist_dir / "manifest.json"
        print(f"index        : {'present' if manifest.exists() else 'NOT BUILT'} ({manifest})")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
