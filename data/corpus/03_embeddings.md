# Embeddings

## What an embedding is
An embedding is a dense vector of floating-point numbers that represents the meaning of
a piece of text. Texts with similar meaning map to vectors that are close together in
the vector space, which lets a system find related passages by comparing vectors.

## Measuring similarity
The most common similarity measure for text embeddings is cosine similarity, which
compares the angle between two vectors and ignores their magnitude. Cosine similarity
ranges from -1 to 1, where 1 means the vectors point in the same direction. When vectors
are normalized to unit length, cosine similarity equals their dot product.

## Embedding models
Sentence-transformers is a popular library of embedding models. The model
all-MiniLM-L6-v2 produces 384-dimensional vectors and is fast enough to run on a CPU,
which makes it a common default. Larger models such as BGE or E5 can improve retrieval
quality at the cost of speed. Ollama can serve the nomic-embed-text model locally.

## Dimensionality
The dimensionality of an embedding is the length of its vector. Higher dimensionality
can capture more nuance but increases storage and computation. The dimensionality is
fixed by the embedding model, so all chunks in one index must use the same model.
