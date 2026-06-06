"""Step 1 verify: chunks carry source+page; a sample doc yields the expected count."""

from __future__ import annotations

from rag.ingest import ingest, load_corpus


def test_pages_have_provenance(offline_config):
    pages = load_corpus(offline_config.corpus_dir)
    assert pages, "corpus should not be empty"
    for p in pages:
        assert p.source
        assert p.page >= 1
        assert p.text.strip()


def test_chunks_carry_source_and_page(offline_config):
    chunks = ingest(offline_config)
    assert chunks
    for c in chunks:
        assert c.source.endswith((".md", ".txt", ".pdf"))
        assert c.page >= 1
        assert c.metadata["source"] == c.source
        assert c.metadata["page"] == c.page
        assert "::p" in c.chunk_id and "::c" in c.chunk_id


def test_expected_chunk_count_for_sample(tmp_path, offline_config):
    # A 5-section markdown file -> 5 sections -> 5 chunks (each well under chunk_size).
    body = "\n\n".join(f"# Section {i}\nSome body text for section {i}." for i in range(1, 6))
    (tmp_path / "sample.md").write_text(body, encoding="utf-8")
    chunks = ingest(offline_config.with_overrides(corpus_dir=tmp_path))
    assert len(chunks) == 5
    assert {c.page for c in chunks} == {1, 2, 3, 4, 5}


def test_italian_document_is_loaded(offline_config):
    sources = {p.source for p in load_corpus(offline_config.corpus_dir)}
    assert "08_rag_in_italiano.md" in sources, "bilingual corpus should include the Italian doc"
