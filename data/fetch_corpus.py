"""Download a few open-access arXiv PDFs into data/corpus/private/ (git-ignored).

These are foundational RAG / evaluation papers, handy for trying the pipeline on real
documents. They are NOT committed — respect each paper's license. Re-build after fetching:

    python data/fetch_corpus.py
    python -m rag.pipeline build
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# (arXiv id, friendly filename) — open-access abstracts/PDFs on arxiv.org.
PAPERS = [
    ("2005.11401", "lewis-2020-rag.pdf"),            # original RAG paper
    ("2309.15217", "es-2023-ragas.pdf"),             # RAGAS
    ("2212.10496", "gao-2022-hyde.pdf"),             # HyDE
    ("2104.08663", "thakur-2021-beir.pdf"),          # BEIR retrieval benchmark
]

OUT_DIR = Path(__file__).resolve().parent / "corpus" / "private"


def fetch(arxiv_id: str, filename: str, out_dir: Path) -> bool:
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    dest = out_dir / filename
    if dest.exists():
        print(f"  skip (exists): {filename}")
        return True
    req = urllib.request.Request(url, headers={"User-Agent": "rag-with-eval/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            data = resp.read()
        dest.write_bytes(data)
        print(f"  saved: {filename} ({len(data) // 1024} KB)")
        return True
    except Exception as exc:
        print(f"  FAILED {arxiv_id}: {exc}")
        return False


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(PAPERS)} papers into {OUT_DIR}")
    ok = sum(fetch(aid, name, OUT_DIR) for aid, name in PAPERS)
    print(f"Done: {ok}/{len(PAPERS)} available. Now run: python -m rag.pipeline build")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
