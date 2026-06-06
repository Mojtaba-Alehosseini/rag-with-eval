# Retrieval: Lexical, Dense, and Hybrid

## Lexical retrieval with BM25
BM25 is a ranking function from classical information retrieval that scores a document
by how often the query terms appear in it, weighted by how rare each term is across the
corpus. BM25 is a lexical method: it matches exact words and does not understand
synonyms, but it is fast, needs no training, and is a strong baseline.

## Dense retrieval
Dense retrieval embeds the query and the chunks into the same vector space and returns
the chunks whose vectors are most similar to the query vector. Unlike BM25, dense
retrieval can match passages that share meaning even when they use different words, but
it can miss exact keyword matches such as product codes or rare names.

## Hybrid search
Hybrid search combines lexical and dense retrieval to get the strengths of both. A
common way to merge the two ranked lists is Reciprocal Rank Fusion (RRF), which scores
each chunk by the sum of 1 divided by a constant plus its rank in each list. RRF needs
no score normalization, which makes it robust when the two retrievers use different
score scales.

## Top-k
The parameter k, or top-k, is the number of chunks returned by the retriever. A small k
keeps the context focused and cheap, while a larger k improves recall at the risk of
adding noise that can distract the language model.
