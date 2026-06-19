# Proposal: pluggable BM25 analyzer

**Status:** implemented as a dependency-free tokenizer hook.

## Problem

The default BM25 analyzer is intentionally simple: lowercase plus Unicode word
tokenization. That is a good baseline for exact identifiers, filenames, hashes
and short technical tokens, but it is too rigid for many corpora:

- morphologically rich languages may use different surface forms for the same
  underlying word;
- domain corpora may need custom normalization for aliases, product names or
  spelling variants;
- exact technical tokens must still be preserved.

When BM25 misses these lexical variants, relevant documents may never enter the
candidate set that later fusion or reranking can reorder.

## API

BM25 now accepts an injectable tokenizer/analyzer:

```python
from mnemostack.recall import BM25Retriever, Tokenizer

def analyzer(text: str) -> list[str]:
    ...

retriever = BM25Retriever(docs, tokenizer=analyzer)
```

The same tokenizer is applied to both corpus and query text. `from_qdrant` also
threads the analyzer while building the BM25 corpus and avoids a second corpus
tokenization pass:

```python
retriever = BM25Retriever.from_qdrant(
    client,
    "memory",
    tokenizer=analyzer,
)
```

For manually pre-tokenized `BM25Doc` objects, pass `retokenize=False` to
`BM25` or `BM25Retriever` when the stored tokens already came from the same
analyzer.

## Dependency Policy

Core remains dependency-free. Lemmatizers, stemmers and domain-specific
normalizers belong in application code or optional adapters. This keeps the
default exact-token behavior unchanged while giving multilingual and
domain-heavy corpora a stable extension point.

## Notes

Analyzer hooks improve candidate recall. They do not replace reranking: an
analyzer can normalize lexical variants, while a scoring reranker can still
improve ordering for paraphrase, synonymy and buried facts.
