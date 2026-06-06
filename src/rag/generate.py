"""Answer generation with mandatory citations and a no-context guard.

Contract (enforced here, tested in tests/):
  * If the best retrieved chunk is too weak (dense similarity < threshold) or there
    is nothing retrieved, we refuse with :data:`NO_ANSWER` — no hallucination.
  * Otherwise we ask the LLM to answer using ONLY the retrieved context and to cite
    passages with ``[n]`` markers. Every answered result carries >=1 citation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag.models import Citation
from rag.prompts import NO_ANSWER
from rag.textutil import content_tokens

if TYPE_CHECKING:
    from rag.config import Config
    from rag.models import Retrieved
    from rag.providers.llm import LLM

_CITATION_RE = re.compile(r"\[(\d+)\]")
_SNIPPET_LEN = 240


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation]
    answered: bool


def make_citations(retrieved: list[Retrieved]) -> list[Citation]:
    """Number retrieved chunks [1..n] and build Citation rows."""
    citations = []
    for i, r in enumerate(retrieved, start=1):
        snippet = r.chunk.text.strip().replace("\n", " ")
        if len(snippet) > _SNIPPET_LEN:
            snippet = snippet[:_SNIPPET_LEN].rstrip() + "..."
        citations.append(
            Citation(
                index=i,
                source=r.chunk.source,
                page=r.chunk.page,
                snippet=snippet,
                score=round(r.score, 4),
            )
        )
    return citations


def _mark_cited(answer: str, citations: list[Citation]) -> bool:
    """Flag citations referenced by ``[n]`` markers; return whether any were found."""
    referenced = {int(m) for m in _CITATION_RE.findall(answer)}
    found = False
    for c in citations:
        if c.index in referenced:
            c.cited = True
            found = True
    return found


def _query_term_recall(question: str, retrieved: list[Retrieved]) -> float:
    """Fraction of the question's content words that appear in the retrieved context.

    Robust out-of-corpus signal for a lexical retriever: a short, distinctive query
    can have a low cosine against long chunks even when relevant, but its key terms
    will still be present — and entirely absent for a truly out-of-corpus question.
    """
    q_tokens = set(content_tokens(question))
    if not q_tokens:
        return 1.0
    ctx_tokens: set[str] = set()
    for r in retrieved:
        ctx_tokens |= set(content_tokens(r.chunk.text))
    return len(q_tokens & ctx_tokens) / len(q_tokens)


def _is_answerable(
    config: Config,
    question: str,
    retrieved: list[Retrieved],
    dense_top_score: float,
    lexical_embedder: bool,
) -> bool:
    if not retrieved:
        return False
    if lexical_embedder:
        return _query_term_recall(question, retrieved) >= config.no_answer_recall
    return dense_top_score >= config.no_answer_threshold


def generate_answer(
    config: Config,
    llm: LLM,
    question: str,
    retrieved: list[Retrieved],
    dense_top_score: float,
    lexical_embedder: bool = False,
) -> AnswerResult:
    """Produce a cited answer, or refuse when the corpus doesn't support one."""
    if not _is_answerable(config, question, retrieved, dense_top_score, lexical_embedder):
        return AnswerResult(answer=NO_ANSWER, citations=[], answered=False)

    citations = make_citations(retrieved)
    contexts = [r.chunk.text for r in retrieved]
    answer = llm.answer(question, contexts).strip()

    # The LLM may decide, even with context, that it can't answer.
    if not answer or answer.strip() == NO_ANSWER or "couldn't find an answer" in answer.lower():
        return AnswerResult(answer=NO_ANSWER, citations=[], answered=False)

    has_citation = _mark_cited(answer, citations)
    if not has_citation:
        # Enforce the "every answer is cited" contract: attribute to the top source.
        answer = f"{answer} [1]"
        citations[0].cited = True
    return AnswerResult(answer=answer, citations=citations, answered=True)
