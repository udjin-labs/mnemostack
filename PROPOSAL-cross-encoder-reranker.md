# Proposal: first-class ScoringReranker

**Status:** implemented as a backend-agnostic scoring reranker abstraction.

## Problem

The existing `Reranker(llm=...)` is a generative LLM reranker: it asks a model
to output candidate IDs. That is flexible, but it can fail in ways that are
specific to text generation: malformed output, missing IDs, partial lists or
ambiguous formatting. When that happens, mnemostack correctly falls back to the
original order, but the rerank step may not improve recall quality.

Some applications need a reranker that returns numeric relevance scores instead
of generated IDs.

## API

`ScoringReranker` accepts any backend that scores each candidate document:

```python
from mnemostack.recall import RelevanceScorer, ScoringReranker

class MyScorer:
    def score(self, query: str, documents: list[str]) -> Iterable[float]:
        ...

reranker = ScoringReranker(MyScorer(), max_items=100)
reranked = reranker.rerank(query, results)
```

The scorer must return exactly one numeric score per document, in the same
order. Lists and one-shot iterables are accepted. Scores are used only
relatively: higher score ranks earlier. Items past `max_items` keep their
original order. If scoring fails or returns malformed output, reranking is
fail-open and the original order is preserved.

## Recommended Backends

Recommended:

- local cross-encoder scorers;
- hosted rerank endpoints;
- dedicated rerank services such as TEI-compatible services, Cohere, Voyage,
  Jina or similar providers.

Allowed but not recommended as a default:

- generative LLM prompts that ask for numeric scores.

Generative LLM scoring can be wrapped behind `RelevanceScorer`, but it
reintroduces prompt/output parsing and score-calibration fragility in a
different shape. The stable default should be a model or service trained to
produce relevance scores.

## Relationship to BM25 Analyzers

Analyzer hooks and scoring rerankers solve different stages. A BM25 analyzer
improves which candidates enter the pool; a scoring reranker improves the order
of candidates already retrieved. Large corpora generally need both: good
candidate generation first, bounded scoring rerank second.
