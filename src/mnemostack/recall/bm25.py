"""BM25 search over a corpus of pre-tokenized documents.

Implementation is intentionally simple and standalone — no dependency on
external BM25 libraries. Good enough for <100K documents.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
Tokenizer = Callable[[str], list[str]]


def tokenize(text: str) -> list[str]:
    """Lowercase + word-character tokenization (Unicode-aware)."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Doc:
    """Document representation for BM25.

    `payload` is user-provided metadata that is round-tripped to search results.
    `tokens` is computed automatically if you pass raw text.
    """

    id: str | int
    text: str
    payload: dict[str, Any] | None = None
    tokens: list[str] | None = None

    def __post_init__(self):
        if self.tokens is None:
            self.tokens = tokenize(self.text)
        if self.payload is None:
            self.payload = {}


class BM25:
    """Standard BM25 (Okapi) implementation.

    Parameters:
        k1: term frequency saturation (1.2–2.0 typical)
        b:  length normalization (0.75 typical)
        tokenizer: analyzer applied to query text and, by default, corpus text.
        retokenize: set to ``False`` when ``documents`` already carry tokens
            produced by ``tokenizer``.
    """

    def __init__(
        self,
        documents: list[BM25Doc],
        k1: float = 1.5,
        b: float = 0.75,
        tokenizer: Tokenizer = tokenize,
        *,
        retokenize: bool | None = None,
    ):
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        if retokenize is None:
            retokenize = tokenizer is not tokenize
        if not retokenize:
            self.docs = documents
        else:
            # Existing BM25Doc instances may already carry tokens produced by
            # the default analyzer. A custom analyzer must be applied to both
            # corpus and query text, so rebuild the token lists consistently.
            self.docs = [
                BM25Doc(id=d.id, text=d.text, payload=d.payload, tokens=tokenizer(d.text))
                for d in documents
            ]
        self.doc_count = len(self.docs)
        self.avgdl = sum(len(d.tokens or []) for d in self.docs) / max(self.doc_count, 1)
        # Document frequency per term
        self.df: dict[str, int] = defaultdict(int)
        for d in self.docs:
            for term in set(d.tokens or []):
                self.df[term] += 1
        # IDF per term
        self.idf: dict[str, float] = {}
        for term, df in self.df.items():
            self.idf[term] = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))

    def _score(self, query_tokens: list[str], doc: BM25Doc) -> float:
        score = 0.0
        tf = Counter(doc.tokens)
        doc_len = len(doc.tokens or [])
        for term in query_tokens:
            if term not in self.idf:
                continue
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            num = freq * (self.k1 + 1)
            denom = freq + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1))
            score += self.idf[term] * num / denom
        return score

    def search(
        self,
        query: str,
        limit: int = 10,
        predicate: Callable[[BM25Doc], bool] | None = None,
    ) -> list[tuple[BM25Doc, float]]:
        """Return top-K (doc, score) pairs sorted by score desc.

        *predicate* restricts the candidate set BEFORE the top-K cut: a
        post-cut filter would return fewer than K matching results whenever
        foreign docs out-scored them, losing recall.
        """
        q_tokens = self.tokenizer(query)
        if not q_tokens:
            return []
        candidates = self.docs if predicate is None else [d for d in self.docs if predicate(d)]
        scored = [(d, self._score(q_tokens, d)) for d in candidates]
        scored = [x for x in scored if x[1] > 0]
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]
