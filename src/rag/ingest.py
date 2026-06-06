"""Load a document corpus and chunk it, preserving source + page/section metadata.

Loaders:
  * ``.pdf``       -> one logical page per PDF page (via pypdf), real page numbers.
  * ``.md``        -> split on markdown headings; each section is a "page".
  * ``.txt``       -> the whole file is page 1 (or paragraph blocks if very long).

Chunker: LlamaIndex ``SentenceSplitter`` when installed (the spec-mandated chunker),
otherwise a sentence-packing fallback with the same size/overlap semantics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rag.models import Chunk
from rag.textutil import split_sentences

if TYPE_CHECKING:
    from rag.config import Config

_SUPPORTED = {".pdf", ".txt", ".md", ".markdown"}
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_HEADING_LINE_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)


def _headings_to_sentences(section: str) -> str:
    """Turn ``## Heading`` lines into short sentences (``Heading.``).

    Keeps the heading's words (useful for retrieval) while restoring a sentence
    boundary, so downstream sentence-splitting doesn't glue a heading onto the
    following prose.
    """
    return _HEADING_LINE_RE.sub(lambda m: m.group(1).rstrip(".!?:") + ".", section)


@dataclass
class Page:
    source: str
    page: int
    text: str


def _load_pdf(path: Path) -> list[Page]:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pypdf is required to read PDFs (it is a core dependency).") from exc
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Page(source=path.name, page=i, text=text))
    return pages


def _split_markdown_sections(text: str) -> list[str]:
    """Split markdown into heading-delimited sections (preamble kept as section 1)."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [text]
    sections = []
    starts = [m.start() for m in matches]
    if starts[0] > 0:
        pre = text[: starts[0]].strip()
        if pre:
            sections.append(pre)
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)
    return sections


def _load_text(path: Path) -> list[Page]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in {".md", ".markdown"}:
        sections = _split_markdown_sections(text)
        return [
            Page(source=path.name, page=i, text=_headings_to_sentences(s))
            for i, s in enumerate(sections, start=1)
        ]
    text = text.strip()
    return [Page(source=path.name, page=1, text=text)] if text else []


def load_corpus(corpus_dir: Path) -> list[Page]:
    """Load every supported document under ``corpus_dir`` (recursively)."""
    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: {corpus_dir}. "
            "Add documents to data/corpus/ or run the fetch script."
        )
    pages: list[Page] = []
    for path in sorted(corpus_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in _SUPPORTED:
            if path.suffix.lower() == ".pdf":
                pages.extend(_load_pdf(path))
            else:
                pages.extend(_load_text(path))
    return pages


def _builtin_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Sentence-packing splitter (token≈word) mirroring SentenceSplitter semantics."""
    sentences = split_sentences(text)
    if not sentences:
        return []
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for sent in sentences:
        slen = len(sent.split())
        if cur and cur_len + slen > chunk_size:
            chunks.append(" ".join(cur))
            # Carry over a tail of ~chunk_overlap tokens for context continuity.
            overlap: list[str] = []
            olen = 0
            for s in reversed(cur):
                olen += len(s.split())
                overlap.insert(0, s)
                if olen >= chunk_overlap:
                    break
            cur = overlap
            cur_len = sum(len(s.split()) for s in cur)
        cur.append(sent)
        cur_len += slen
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _split_text(text: str, config: Config) -> list[str]:
    try:
        from llama_index.core.node_parser import SentenceSplitter
    except Exception:
        return _builtin_split(text, config.chunk_size, config.chunk_overlap)
    splitter = SentenceSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    return [c for c in splitter.split_text(text) if c.strip()]


def chunk_pages(pages: list[Page], config: Config) -> list[Chunk]:
    """Chunk loaded pages into retrievable units with stable ids + provenance."""
    chunks: list[Chunk] = []
    for page in pages:
        for ci, piece in enumerate(_split_text(page.text, config)):
            chunk_id = f"{page.source}::p{page.page}::c{ci}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=piece,
                    source=page.source,
                    page=page.page,
                    metadata={"source": page.source, "page": page.page},
                )
            )
    return chunks


def ingest(config: Config) -> list[Chunk]:
    """Load + chunk the whole corpus."""
    return chunk_pages(load_corpus(config.corpus_dir), config)


def _main() -> None:
    from rag.config import Config

    config = Config.from_env()
    chunks = ingest(config)
    sources = sorted({c.source for c in chunks})
    print(f"Ingested {len(chunks)} chunks from {len(sources)} documents in {config.corpus_dir}")
    for s in sources:
        n = sum(1 for c in chunks if c.source == s)
        print(f"  {s}: {n} chunks")


if __name__ == "__main__":
    _main()
