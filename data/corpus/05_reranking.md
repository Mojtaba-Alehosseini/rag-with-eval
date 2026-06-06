# Reranking

## Bi-encoders versus cross-encoders
A bi-encoder embeds the query and each chunk separately and compares their vectors; this
is fast and is what powers first-stage dense retrieval. A cross-encoder instead feeds the
query and a chunk together into the model and outputs a single relevance score. The
cross-encoder is far more accurate because it can attend to interactions between the two
texts, but it is too slow to run over an entire corpus.

## The two-stage pattern
Reranking uses bi-encoders and cross-encoders together in a two-stage pattern. The fast
bi-encoder retrieves a candidate pool of perhaps the top 20 chunks, and then the slower
cross-encoder rescores only those candidates and keeps the best few. This gives most of
the cross-encoder's accuracy at a fraction of the cost.

## A common reranker
The model cross-encoder/ms-marco-MiniLM-L-6-v2 is a widely used cross-encoder reranker
trained on the MS MARCO passage ranking dataset. It takes a query and a passage and
returns a relevance score used to sort the candidates.

## Why reranking helps evaluation
Reranking most directly improves context precision, because it pushes the truly relevant
chunks to the top of the list. Reporting a metric such as context precision rising from
0.61 to 0.84 after adding a cross-encoder reranker is concrete evidence that the upgrade
worked.
