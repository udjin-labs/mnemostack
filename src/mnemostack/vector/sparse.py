"""Sparse text encoding for Qdrant sparse vectors (server-side lexical search).

A dependency-free BM25-style encoder: tokens hash to stable ``uint32`` indices
(CRC-32), document values carry the term frequency, query values are all ``1``.
Stored in a sparse vector space configured with Qdrant's ``Modifier.IDF``, the
server multiplies each matched term by its collection-wide IDF at query time —
so scoring behaves like the classic *tf·idf* family without the corpus ever
living in client memory. This is what lets the lexical arm scale past the
in-process BM25's documented "<100K documents" ceiling: the index lives in
Qdrant, next to the dense vectors.

Honest approximation notes (documented, deliberate):

- **tf, not BM25's saturated tf.** Values store the raw term frequency; the
  BM25 ``k1``/``b`` saturation and length normalization are not applied. For
  short memory chunks (the ingest default) the difference is small.
- **Hash collisions.** Two tokens may share a CRC-32 bucket (~1 in 4 billion
  per pair); a collision makes two unrelated terms score as one. Accepted for
  a dependency-free encoder.
- The tokenizer mirrors the in-process BM25 default (lowercased ``\\w+`` runs)
  and is injectable, so a deployment using a custom BM25 analyzer can keep
  both arms consistent.

This module must stay importable without the recall layer (the vector layer
never imports recall — same rule as the validity keys).
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from collections.abc import Callable

#: Name of the sparse vector space `VectorStore(sparse_text=True)` maintains.
SPARSE_TEXT_VECTOR = "text_sparse"

_WORD_RE = re.compile(r"\w+", re.UNICODE)

#: A tokenizer takes raw text and returns lexical tokens.
SparseTokenizer = Callable[[str], list[str]]


def _default_tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def token_index(token: str) -> int:
    """Stable uint32 index for a token (CRC-32 of its UTF-8 bytes)."""
    return zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF


class SparseTextEncoder:
    """Encode text into Qdrant sparse-vector ``(indices, values)`` pairs."""

    def __init__(self, tokenizer: SparseTokenizer | None = None):
        self.tokenize = tokenizer or _default_tokenize

    def encode_document(self, text: str) -> tuple[list[int], list[float]]:
        """Document encoding: one entry per distinct token, value = its tf.
        Colliding tokens aggregate into one bucket (values add)."""
        counts: Counter[int] = Counter()
        for token in self.tokenize(text or ""):
            counts[token_index(token)] += 1
        if not counts:
            return [], []
        indices = sorted(counts)
        return indices, [float(counts[i]) for i in indices]

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        """Query encoding: distinct tokens with value 1 — the server's IDF
        modifier supplies the discriminative weighting."""
        seen = {token_index(t) for t in self.tokenize(text or "")}
        if not seen:
            return [], []
        indices = sorted(seen)
        return indices, [1.0] * len(indices)
