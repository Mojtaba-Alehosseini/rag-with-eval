"""Embedding providers.

``HashingEmbedder`` is the default offline provider: a deterministic signed
hashing-trick embedding (numpy only, no model download). It behaves like an
L2-normalised weighted bag-of-words, which is a perfectly good lexical signal
for a small corpus and makes retrieval/eval reproducible bit-for-bit.

``SentenceTransformerEmbedder`` / ``OllamaEmbedder`` are the real semantic
providers, lazy-imported so they never burden the core install.
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING

import numpy as np

from rag.textutil import content_tokens as _content_tokens
from rag.textutil import tokenize as _tokenize

if TYPE_CHECKING:
    from rag.config import Config


class Embedder:
    """Base embedder. Subclasses implement :meth:`embed_texts`."""

    name: str = "base"
    dim: int = 0

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]


def _stable_hash(token: str) -> int:
    """Process-independent hash (Python's built-in hash() is salted)."""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


class HashingEmbedder(Embedder):
    """Deterministic signed hashing-trick embedding over word uni/bi-grams."""

    name = "hashing"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        # Content words drive the signal; fall back to all tokens if a text is
        # nothing but stopwords (rare) so it still gets a non-zero vector.
        tokens = _content_tokens(text) or _tokenize(text)
        if not tokens:
            return vec
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        counts: dict[str, int] = {}
        for g in grams:
            counts[g] = counts.get(g, 0) + 1
        for gram, count in counts.items():
            h = _stable_hash(gram)
            idx = h % self.dim
            sign = 1.0 if (h >> 1) & 1 else -1.0
            vec[idx] += sign * (1.0 + math.log(count))
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed_one(t) for t in texts])


class SentenceTransformerEmbedder(Embedder):
    """Dense semantic embeddings via sentence-transformers (lazy-imported)."""

    name = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - exercised only with extra installed
            raise RuntimeError(
                "sentence-transformers not installed. Run: pip install -e \".[heavy]\" "
                "or set RAG_EMBED_PROVIDER=hashing for the offline embedder."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        emb = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return emb.astype(np.float32)


class OllamaEmbedder(Embedder):
    """Embeddings via a local Ollama server (e.g. nomic-embed-text)."""

    name = "ollama"

    def __init__(self, model_name: str, host: str) -> None:
        try:
            import ollama
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "ollama not installed. Run: pip install -e \".[ollama]\" "
                "or set RAG_EMBED_PROVIDER=hashing."
            ) from exc
        self._client = ollama.Client(host=host)
        self._model = model_name
        # Probe once to learn the dimension (and fail fast if the server is down).
        probe = self._client.embeddings(model=model_name, prompt="dimension probe")
        self.dim = len(probe["embedding"])

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        for t in texts:
            resp = self._client.embeddings(model=self._model, prompt=t)
            v = np.asarray(resp["embedding"], dtype=np.float32)
            n = float(np.linalg.norm(v))
            out.append(v / n if n > 0 else v)
        return np.vstack(out)


def get_embedder(config: Config) -> Embedder:
    """Resolve config.embed_provider to a concrete embedder.

    ``auto`` prefers sentence-transformers if importable, else the offline hasher.
    """
    provider = config.embed_provider
    if provider == "hashing":
        return HashingEmbedder(dim=config.hashing_dim)
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedder(config.embed_model)
    if provider == "ollama":
        return OllamaEmbedder(config.ollama_embed_model, config.ollama_host)
    if provider == "auto":
        try:
            import sentence_transformers  # noqa: F401
        except Exception:
            return HashingEmbedder(dim=config.hashing_dim)
        return SentenceTransformerEmbedder(config.embed_model)
    raise ValueError(f"Unknown embed_provider: {provider!r}")
