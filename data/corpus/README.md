# Corpus

This folder is the document corpus the RAG system indexes.

## What ships in the repo
The committed `.md` files (`01_…` to `08_…`) are an **original, MIT-licensed sample
corpus** written for this project. They cover RAG, chunking, embeddings, retrieval,
reranking, evaluation, and vector databases, and one document is in **Italian** to
demonstrate bilingual retrieval. They double as the ground truth for `eval/testset.yaml`.

Because the corpus is original, no third-party copyright is committed to the repo.

## Bringing your own documents
Drop your own `.pdf`, `.md`, or `.txt` files into this folder (or a subfolder) and rebuild:

```bash
python -m rag.pipeline build
```

- PDFs are loaded per page (real page numbers in citations) via `pypdf`.
- Markdown is split into heading sections; plain text is one section.

**Do not commit copyrighted PDFs.** Put private documents under `data/corpus/private/`
(git-ignored) or fetch them with the script below.

## Fetching open-access papers
`data/fetch_corpus.py` downloads a few open-access arXiv PDFs into
`data/corpus/private/` so you can try the pipeline on real papers without committing them:

```bash
python data/fetch_corpus.py
python -m rag.pipeline build
```

Respect each source's license; arXiv articles keep their authors' chosen license.
