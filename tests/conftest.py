"""Shared fixtures. Tests run on deterministic offline providers (no downloads, no keys)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import Config
from rag.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"


@pytest.fixture(scope="session")
def offline_config(tmp_path_factory) -> Config:
    persist = tmp_path_factory.mktemp("rag_index")
    return Config(
        llm_provider="offline",
        embed_provider="hashing",
        vector_store="numpy",
        corpus_dir=CORPUS,
        persist_dir=persist,
    )


@pytest.fixture(scope="session")
def built_pipeline(offline_config: Config) -> RAGPipeline:
    return RAGPipeline.from_config(offline_config).build()
