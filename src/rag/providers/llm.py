"""LLM providers.

``ExtractiveLLM`` is the default offline provider: it composes an answer by
selecting the context sentences most relevant to the question and attaching
citation markers. Being purely extractive, it is faithful by construction (it
never states anything not in the retrieved context) and fully deterministic,
which is exactly what we want for hermetic tests and a key-free demo.

``OllamaLLM`` / ``GeminiLLM`` are the real generative providers, lazy-imported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.prompts import build_prompt
from rag.textutil import content_tokens, split_sentences, tokenize

if TYPE_CHECKING:
    from rag.config import Config


class LLM:
    """Base LLM. Subclasses implement :meth:`answer`."""

    name: str = "base"

    def answer(self, question: str, contexts: list[str]) -> str:
        raise NotImplementedError


class ExtractiveLLM(LLM):
    """Deterministic, citation-grounded extractive 'generation'.

    Picks the best-matching sentences from the top contexts (by token overlap
    with the question) and returns them with ``[n]`` citations.
    """

    name = "extractive"

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def answer(self, question: str, contexts: list[str]) -> str:
        if not contexts:
            from rag.prompts import NO_ANSWER

            return NO_ANSWER
        q_tokens = set(tokenize(question))

        # Score substantial sentences across the contexts; remember which context
        # (=> which citation index) each came from. Skip headings/fragments.
        scored: list[tuple[float, int, str]] = []
        for ctx_idx, ctx in enumerate(contexts):
            for sent in split_sentences(ctx):
                if len(content_tokens(sent)) < 3:
                    continue
                s_tokens = set(tokenize(sent))
                overlap = len(q_tokens & s_tokens)
                if overlap == 0:
                    continue
                # Normalise by sentence length to avoid favouring very long lines.
                scored.append((overlap / (len(s_tokens) ** 0.5), ctx_idx, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            # No overlap with any sentence — fall back to the first substantial
            # sentence of the top context so we still answer something grounded.
            fallback = next(
                (s for s in split_sentences(contexts[0]) if len(content_tokens(s)) >= 3),
                contexts[0],
            )
            scored = [(0.0, 0, fallback)]

        top_score = scored[0][0]
        picked = [scored[0]]
        for cand in scored[1 : self.max_sentences]:
            # Add a supporting sentence only if it is clearly relevant too.
            if cand[0] > 0 and cand[0] >= 0.45 * top_score:
                picked.append(cand)

        picked.sort(key=lambda x: x[1])  # keep original context order for readability
        parts = []
        for _score, ctx_idx, sent in picked:
            sent = sent.strip().lstrip("#").strip()
            if not sent.endswith((".", "!", "?")):
                sent += "."
            parts.append(f"{sent} [{ctx_idx + 1}]")
        return " ".join(parts)


class OllamaLLM(LLM):
    """Local generation via Ollama (e.g. llama3.1:8b)."""

    name = "ollama"

    def __init__(self, model_name: str, host: str) -> None:
        try:
            import ollama
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "ollama not installed. Run: pip install -e \".[ollama]\" "
                "or set RAG_LLM_PROVIDER=offline."
            ) from exc
        self._client = ollama.Client(host=host)
        self._model = model_name

    def answer(self, question: str, contexts: list[str]) -> str:
        prompt = build_prompt(question, contexts)
        resp = self._client.generate(model=self._model, prompt=prompt, options={"temperature": 0.0})
        return str(resp["response"]).strip()


class GeminiLLM(LLM):
    """Hosted generation via the free Gemini tier (used by the HF Spaces demo)."""

    name = "gemini"

    def __init__(self, model_name: str, api_key: str) -> None:
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is empty; set it or use a different provider.")
        try:
            import google.generativeai as genai
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "google-generativeai not installed. Run: pip install -e \".[demo]\"."
            ) from exc
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)

    def answer(self, question: str, contexts: list[str]) -> str:
        prompt = build_prompt(question, contexts)
        resp = self._model.generate_content(prompt)
        return str(resp.text).strip()


def _ollama_reachable(host: str) -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen(host, timeout=0.5) as r:  # noqa: S310 (local host)
            return r.status == 200
    except Exception:
        return False


def get_llm(config: Config) -> LLM:
    """Resolve config.llm_provider to a concrete LLM.

    ``auto`` order: Gemini (if key) -> Ollama (if reachable) -> offline extractive.
    """
    provider = config.llm_provider
    if provider == "offline":
        return ExtractiveLLM()
    if provider == "ollama":
        return OllamaLLM(config.ollama_llm_model, config.ollama_host)
    if provider == "gemini":
        return GeminiLLM(config.gemini_model, config.gemini_api_key)
    if provider == "auto":
        if config.gemini_api_key:
            return GeminiLLM(config.gemini_model, config.gemini_api_key)
        if _ollama_reachable(config.ollama_host):
            return OllamaLLM(config.ollama_llm_model, config.ollama_host)
        return ExtractiveLLM()
    raise ValueError(f"Unknown llm_provider: {provider!r}")
