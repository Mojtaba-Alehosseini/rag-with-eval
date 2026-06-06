# Chunking Documents

## Why chunk
Documents are split into smaller passages called chunks before embedding. Chunking
matters because embedding models have a limited context window, and because retrieval
is more precise when each vector represents a focused unit of meaning rather than an
entire document.

## Chunk size and overlap
Two parameters control chunking: chunk size and chunk overlap. Chunk size is the
maximum length of a chunk, commonly measured in tokens. Overlap is the number of tokens
repeated between consecutive chunks so that a sentence split across a boundary still
appears intact in at least one chunk. A common starting point is a chunk size of 1024
tokens with an overlap of 200 tokens.

## Sentence-aware splitting
A naive splitter cuts text at a fixed character count and can break sentences in half.
A sentence-aware splitter, such as LlamaIndex's SentenceSplitter, packs whole sentences
into a chunk until the size limit is reached, which keeps each chunk readable and
semantically coherent.

## Keeping metadata
Each chunk should carry metadata describing where it came from, at minimum the source
file name and the page or section number. This provenance is what makes citations
possible: when a chunk is retrieved, the system can point the user back to the exact
location in the original document.
