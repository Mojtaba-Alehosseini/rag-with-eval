"""Prompt templates shared by the real LLM providers.

The whole project's anti-hallucination contract lives here: answer ONLY from the
provided context, cite every claim with a ``[n]`` marker, and refuse with a fixed
sentinel when the context does not contain the answer.
"""

from __future__ import annotations

# Returned verbatim when the corpus does not contain the answer. The pipeline also
# detects this to flag a "no answer" result in the UI / eval.
NO_ANSWER = "I couldn't find an answer to that in the provided documents."

_INSTRUCTION = (
    "You are a careful retrieval assistant. Answer the question using ONLY the "
    "numbered context passages below. Cite the passages you use with their bracket "
    "markers like [1] or [2] — every factual claim must carry at least one citation. "
    "Do not use any outside knowledge. If the answer is not contained in the context, "
    f'reply with exactly: "{NO_ANSWER}"'
)


def format_contexts(contexts: list[str]) -> str:
    """Render passages as ``[1] ...`` blocks the model is told to cite."""
    return "\n\n".join(f"[{i + 1}] {c.strip()}" for i, c in enumerate(contexts))


def build_prompt(question: str, contexts: list[str]) -> str:
    """Full single-string prompt for completion-style models."""
    return (
        f"{_INSTRUCTION}\n\n"
        f"Context passages:\n{format_contexts(contexts)}\n\n"
        f"Question: {question}\n\n"
        f"Answer (with [n] citations):"
    )
