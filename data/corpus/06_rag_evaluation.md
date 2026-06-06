# Evaluating RAG Systems

## Why evaluate
Plain RAG is now expected, so what differentiates an engineer is measuring whether the
system actually works. A RAG evaluation harness reports numbers for retrieval quality and
answer quality, and shows a measurable improvement when a component is upgraded.

## Faithfulness
Faithfulness measures whether the generated answer is supported by the retrieved context.
An answer is faithful when every claim it makes can be inferred from the context; an
unfaithful answer contains statements the context does not support, which is a
hallucination. Faithfulness is a property of the generation step.

## Answer relevancy
Answer relevancy measures how well the answer addresses the question that was asked. An
answer can be perfectly faithful to the context yet still score poorly on answer
relevancy if it is evasive, incomplete, or off-topic.

## Context precision
Context precision measures the quality of retrieval: of the chunks that were retrieved,
how many are actually relevant, and are the relevant ones ranked near the top. Because it
is rank-sensitive, context precision is the metric that improves most when a reranker
reorders the retrieved chunks.

## RAGAS
RAGAS is an open-source framework that computes these metrics using a judge language
model and an embedding model. By default RAGAS uses OpenAI for both, so it raises an
error without an OpenAI key; the fix is to configure the judge and embeddings to a local
model such as Ollama, or to a free hosted tier, before calling evaluate.
