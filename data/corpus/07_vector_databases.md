# Vector Databases

## What a vector store does
A vector store, or vector database, indexes embeddings so that the nearest neighbours of
a query vector can be found quickly. It stores each vector together with the original
text and its metadata, and exposes a similarity search over the vectors.

## Exact versus approximate search
For a small corpus, a brute-force search that compares the query against every stored
vector is exact and fast enough. For large corpora, an approximate nearest neighbour
index such as HNSW trades a small amount of recall for a large speed-up by searching a
navigable graph instead of every vector.

## Common options
ChromaDB is a lightweight embedded vector database that persists to local disk and is
easy to start with. FAISS is a high-performance similarity-search library from Meta.
pgvector adds vector search to PostgreSQL, which is convenient when the rest of the
application already uses a relational database.

## Choosing a distance
Vector stores let you choose the distance metric used for search, commonly cosine
distance, dot product, or Euclidean distance. The choice should match how the embedding
model was trained; most sentence-transformers models are tuned for cosine similarity.
