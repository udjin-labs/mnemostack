"""Role dispatch and document-space guard helpers.

Internal call sites go through these helpers instead of calling the role
methods directly so that third-party / duck-typed providers implementing only
the legacy ``embed`` / ``embed_batch`` surface keep working unchanged (the
same defensive posture as the Ingestor's ``embed_batch`` AttributeError
fallback). A provider subclassing :class:`EmbeddingProvider` inherits the
role methods, so the fallback only fires for objects that never had them.
"""

from __future__ import annotations

from itertools import islice
from typing import Any

from .profiles import EMBEDDING_SPACE_KEY

__all__ = [
    "EMBEDDING_SPACE_KEY",
    "check_document_space",
    "document_space_fingerprint_via",
    "embed_document_via",
    "embed_documents_via",
    "embed_queries_via",
    "embed_query_via",
]


def embed_query_via(provider: Any, text: str) -> list[float]:
    """Embed a retrieval query via the provider's role method when available."""
    method = getattr(provider, "embed_query", None)
    if method is not None:
        return method(text)
    return provider.embed(text)


def embed_queries_via(provider: Any, texts: list[str]) -> list[list[float]]:
    """Embed retrieval queries via the provider's role method when available."""
    method = getattr(provider, "embed_queries", None)
    if method is not None:
        return method(texts)
    return provider.embed_batch(texts)


def embed_document_via(provider: Any, text: str) -> list[float]:
    """Embed a document/chunk via the provider's role method when available."""
    method = getattr(provider, "embed_document", None)
    if method is not None:
        return method(text)
    return provider.embed(text)


def embed_documents_via(provider: Any, texts: list[str]) -> list[list[float]]:
    """Embed documents/chunks via the provider's role method when available."""
    method = getattr(provider, "embed_documents", None)
    if method is not None:
        return method(texts)
    return provider.embed_batch(texts)


def document_space_fingerprint_via(provider: Any) -> str | None:
    """The provider's document-space fingerprint, or None for legacy providers."""
    method = getattr(provider, "document_space_fingerprint", None)
    return method() if method is not None else None


def check_document_space(
    store: Any, provider: Any, *, sample_size: int = 16
) -> tuple[str, str | None, str | None]:
    """Compare the provider's document-space fingerprint against stored points.

    Samples up to ``sample_size`` points (lazy scroll — never loads the
    collection). Returns ``(status, expected, found)`` where status is:

    - ``"match"`` — a sampled point carries the current fingerprint;
    - ``"mismatch"`` — a sampled point was embedded under a DIFFERENT
      document space; indexing must stop (recreate or use a new collection);
    - ``"legacy"`` — points exist but none of the sample carries a
      fingerprint (indexed before fingerprints existed); proceeding adopts
      the current space for new points;
    - ``"empty"`` — no points yet;
    - ``"unknown"`` — nothing to compare (provider without fingerprints or
      store without scroll).

    A small sample is a deliberate trade-off: scanning everything on every
    index run would punish large corpora. Be honest about its limits: the
    scroll order is stable, so repeated runs examine the SAME prefix — a
    mixed state introduced outside the CLI (direct store writes, a restored
    snapshot) that sits beyond the sample window is not detected here. The
    refresh paths therefore never overwrite a conflicting stamp: an
    undetected mismatch stays visible instead of being laundered into the
    current space.

    Callers that stamp fingerprints (the CLI index commands) run this check
    FIRST. Library consumers driving `Ingestor` or `upsert_markdown_chunks`
    directly are responsible for calling it themselves — those building
    blocks stamp but do not guard.
    """
    expected = document_space_fingerprint_via(provider)
    scroll = getattr(store, "scroll", None)
    if expected is None or scroll is None:
        return "unknown", expected, None
    sampled = 0
    found: str | None = None
    for hit in islice(scroll(), sample_size):
        sampled += 1
        fp = (getattr(hit, "payload", None) or {}).get(EMBEDDING_SPACE_KEY)
        if fp is None:
            continue
        found = str(fp)
        if found != expected:
            return "mismatch", expected, found
    if sampled == 0:
        return "empty", expected, None
    if found is None:
        return "legacy", expected, None
    return "match", expected, found
