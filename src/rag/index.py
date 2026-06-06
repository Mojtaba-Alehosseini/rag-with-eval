"""Embed chunks and store them in a vector store.

``NumpyVectorStore`` is the deterministic default: a brute-force cosine index
(fine for the modest corpora this project targets) persisted as ``.npy`` + JSON,
with zero heavy dependencies. ``ChromaVectorStore`` is used when chromadb is
installed (the spec-mentioned real store). Both share one interface.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from rag.models import Chunk, Retrieved

if TYPE_CHECKING:
    from rag.config import Config
    from rag.providers.embeddings import Embedder

_MANIFEST = "manifest.json"


class VectorStore:
    """Base store interface."""

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        raise NotImplementedError

    def query(self, embedding: np.ndarray, k: int) -> list[Retrieved]:
        raise NotImplementedError

    def all_chunks(self) -> list[Chunk]:
        """Every stored chunk (used to build the BM25 index for hybrid search)."""
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def persist(self) -> None:
        raise NotImplementedError


def _chunk_payload(chunk: Chunk) -> dict:
    d = asdict(chunk)
    d.pop("metadata", None)  # source/page already top-level; keep payload small
    return d


class NumpyVectorStore(VectorStore):
    """Brute-force cosine store backed by a numpy matrix."""

    def __init__(self, persist_dir, dim: int) -> None:
        from pathlib import Path

        self.dir = Path(persist_dir) / "numpy_store"
        self.dim = dim
        self._emb = np.zeros((0, dim), dtype=np.float32)
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings length mismatch")
        self._chunks.extend(chunks)
        self._emb = np.vstack([self._emb, embeddings.astype(np.float32)])

    def query(self, embedding: np.ndarray, k: int) -> list[Retrieved]:
        if self._emb.shape[0] == 0:
            return []
        q = embedding.astype(np.float32)
        qn = float(np.linalg.norm(q))
        if qn > 0:
            q = q / qn
        # Stored vectors are L2-normalised by the embedders, so dot == cosine.
        sims = self._emb @ q
        k = min(k, sims.shape[0])
        # Deterministic top-k: rank by descending similarity, breaking ties by
        # ascending index so results are reproducible across numpy versions.
        order = np.lexsort((np.arange(sims.shape[0]), -sims))[:k]
        return [
            Retrieved(chunk=self._chunks[i], score=float(sims[i]), method="dense") for i in order
        ]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def count(self) -> int:
        return len(self._chunks)

    def persist(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self.dir / "embeddings.npy", self._emb)
        payload = [_chunk_payload(c) for c in self._chunks]
        (self.dir / "chunks.json").write_text(json.dumps(payload, ensure_ascii=False), "utf-8")

    @classmethod
    def load(cls, persist_dir, dim: int) -> NumpyVectorStore:
        store = cls(persist_dir, dim)
        store._emb = np.load(store.dir / "embeddings.npy")
        raw = json.loads((store.dir / "chunks.json").read_text("utf-8"))
        store._chunks = [
            Chunk(
                chunk_id=r["chunk_id"],
                text=r["text"],
                source=r["source"],
                page=r["page"],
                metadata={"source": r["source"], "page": r["page"]},
            )
            for r in raw
        ]
        return store


class ChromaVectorStore(VectorStore):
    """Persistent cosine store backed by ChromaDB."""

    def __init__(self, persist_dir, dim: int) -> None:
        from pathlib import Path

        import chromadb

        self.dim = dim
        path = str(Path(persist_dir) / "chroma")
        self._client = chromadb.PersistentClient(path=path)
        self._col = self._client.get_or_create_collection(
            name="corpus", metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        self._col.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=[e.tolist() for e in embeddings],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "page": c.page} for c in chunks],
        )

    def query(self, embedding: np.ndarray, k: int) -> list[Retrieved]:
        if self.count() == 0:
            return []
        res = cast(
            "dict[str, Any]",
            self._col.query(
                query_embeddings=[embedding.tolist()],
                n_results=min(k, self.count()),
                include=["documents", "metadatas", "distances"],
            ),
        )
        out = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=False):
            chunk = Chunk(
                chunk_id=cid,
                text=doc,
                source=str(meta["source"]),
                page=int(meta["page"]),
                metadata=dict(meta),
            )
            out.append(Retrieved(chunk=chunk, score=1.0 - float(dist), method="dense"))
        return out

    def all_chunks(self) -> list[Chunk]:
        res = cast("dict[str, Any]", self._col.get(include=["documents", "metadatas"]))
        chunks = []
        for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=False):
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    text=doc,
                    source=str(meta["source"]),
                    page=int(meta["page"]),
                    metadata=dict(meta),
                )
            )
        return chunks

    def count(self) -> int:
        return int(self._col.count())

    def persist(self) -> None:
        # PersistentClient writes through on add(); nothing extra to do.
        pass


def _store_kind(config: Config) -> str:
    if config.vector_store == "numpy":
        return "numpy"
    if config.vector_store == "chroma":
        return "chroma"
    # auto
    try:
        import chromadb  # noqa: F401
    except Exception:
        return "numpy"
    return "chroma"


def _new_store(kind: str, config: Config, dim: int) -> VectorStore:
    if kind == "chroma":
        return ChromaVectorStore(config.persist_dir, dim)
    return NumpyVectorStore(config.persist_dir, dim)


def build_index(config: Config, chunks: list[Chunk], embedder: Embedder) -> VectorStore:
    """Embed chunks, write them to a fresh store, and record a manifest."""
    from pathlib import Path

    kind = _store_kind(config)
    embeddings = embedder.embed_texts([c.text for c in chunks])
    store = _new_store(kind, config, embedder.dim)
    store.add(chunks, embeddings)
    store.persist()

    persist = Path(config.persist_dir)
    persist.mkdir(parents=True, exist_ok=True)
    manifest = {
        "store": kind,
        "embedder": embedder.name,
        "dim": embedder.dim,
        "count": store.count(),
    }
    (persist / _MANIFEST).write_text(json.dumps(manifest, indent=2), "utf-8")
    return store


def load_index(config: Config, embedder: Embedder) -> VectorStore:
    """Load a previously built store, validating it matches the embedder."""
    from pathlib import Path

    persist = Path(config.persist_dir)
    manifest_path = persist / _MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No index found at {persist}. Build it first: python -m rag.pipeline build"
        )
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest["embedder"] != embedder.name or manifest["dim"] != embedder.dim:
        raise RuntimeError(
            f"Index was built with embedder={manifest['embedder']} dim={manifest['dim']} but the "
            f"current embedder is {embedder.name} dim={embedder.dim}. Rebuild: "
            "python -m rag.pipeline build"
        )
    if manifest["store"] == "chroma":
        return ChromaVectorStore(config.persist_dir, embedder.dim)
    return NumpyVectorStore.load(config.persist_dir, embedder.dim)
