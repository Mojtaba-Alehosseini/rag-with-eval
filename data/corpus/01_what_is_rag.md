# Retrieval-Augmented Generation (RAG)

## Definition
Retrieval-Augmented Generation (RAG) is a technique that grounds a large language
model's answer in external documents retrieved at query time. Instead of relying only
on the parameters learned during training, a RAG system first retrieves relevant
passages from a corpus and then conditions the model's generation on those passages.

## Why use RAG
RAG addresses two core weaknesses of standalone language models: stale knowledge and
hallucination. Because the corpus can be updated independently of the model, RAG keeps
answers current without retraining. Because the answer is grounded in retrieved text,
the system can cite its sources, which makes the output verifiable.

## The core pipeline
A RAG pipeline has five stages: load documents, split them into chunks, embed the
chunks into vectors, store those vectors in a vector store, and at query time retrieve
the most similar chunks and pass them to the language model as context. An optional
reranking stage can reorder the retrieved chunks before generation.

## Citations and grounding
A trustworthy RAG system attaches a citation to every claim, identifying the source
file and page or section. The guiding rule is simple: no hallucinated claims without a
citation. When the retrieved context does not contain the answer, the system should say
so explicitly rather than inventing a response.
