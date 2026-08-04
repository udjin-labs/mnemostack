"""Role dispatch and document-space guard helpers.

Internal call sites go through these helpers instead of calling the role
methods directly so that third-party / duck-typed providers implementing only
the legacy ``embed`` / ``embed_batch`` surface keep working unchanged (the
same defensive posture as the Ingestor's ``embed_batch`` AttributeError
fallback). A provider subclassing :class:`EmbeddingProvider` inherits the
role methods, so the fallback only fires for objects that never had them.
"""

from __future__ import annotations

import logging
import threading
import time
from itertools import islice
from typing import Any

from .profiles import EMBEDDING_SPACE_KEY

logger = logging.getLogger(__name__)

__all__ = [
    "EMBEDDING_SPACE_KEY",
    "EmbeddingSpaceError",
    "SpaceGuard",
    "check_document_space",
    "document_space_fingerprint_via",
    "embed_document_via",
    "embed_documents_via",
    "embed_queries_via",
    "embed_query_via",
    "recall_space_error",
]


class EmbeddingSpaceError(RuntimeError):
    """Recalling or indexing would cross incompatible embedding spaces."""


class SpaceGuard:
    """Lazy, revalidating embedding-space guard for one (store, provider) pair.

    ``ensure()`` is cheap after the first call: a conclusive verdict is
    cached and revalidated after ``recheck_seconds``. Verdicts age out on
    purpose — a live process must notice a recreated/reindexed collection
    (both a NEW mismatch and a HEALED one) without a restart. This is the
    guard's freshness contract: incompatibility introduced under a running
    process is detected at the next revalidation or process start, not
    instantaneously. Fingerprint-resolution failures raise and are never
    cached (fail closed, retried); store hiccups log, fail open and retry.
    """

    def __init__(self, store: Any, provider: Any, *, recheck_seconds: float = 300.0):
        self._store = store
        self._provider = provider
        self._recheck = recheck_seconds
        self._lock = threading.Lock()
        self._error: str | None = None
        self._validated: str | None = None
        self._checked_at: float | None = None

    def _stale(self) -> bool:
        return (
            self._checked_at is None
            or time.monotonic() - self._checked_at > self._recheck
        )

    def ensure(self) -> str | None:
        """Raise :class:`EmbeddingSpaceError` when the pair is incompatible.

        Returns the fingerprint the verdict was computed FROM (None when the
        pair is unguardable/duck-typed). Write paths must stamp exactly this
        value — re-resolving separately would open a window where the guard
        validated space A and the stamp records space B.
        """
        if self._stale():
            with self._lock:
                if self._stale():
                    try:
                        error, validated = _space_verdict(self._store, self._provider)
                    except EmbeddingSpaceError:
                        # Fingerprint unresolvable: fail closed for this
                        # call, cache nothing, retry on the next one.
                        raise
                    except Exception as exc:  # noqa: BLE001 — store hiccup: fail open, retry
                        logger.warning(
                            "embedding-space check inconclusive (%s) — will retry",
                            exc,
                        )
                    else:
                        self._error = error
                        self._validated = validated
                        self._checked_at = time.monotonic()
        if self._error:
            raise EmbeddingSpaceError(self._error)
        return self._validated


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
    """The provider's document-space fingerprint.

    None means the provider has no fingerprint support at all (legacy duck
    type) — a permanent property, safe to treat as "no stamping, no guard".
    A fingerprint that EXISTS but cannot be resolved right now (e.g. the
    Ollama digest lookup failing while embeddings still work) raises
    :class:`EmbeddingSpaceError` instead: writing unstamped vectors or
    recalling unverified in that window could mix repointed weights into one
    collection, so indexing and recall must fail closed and retry — only
    diagnostic callers (doctor) catch and report it.
    """
    method = getattr(provider, "document_space_fingerprint", None)
    if method is None:
        return None
    try:
        return method()
    except Exception as exc:
        raise EmbeddingSpaceError(
            f"document-space fingerprint unavailable: {exc}"
        ) from exc


def recall_space_error(store: Any, provider: Any) -> str | None:
    """Explain why recalling with *provider* against *store* would cross
    embedding spaces — or None when compatible (or not determinable).

    The recall-side counterpart of the index guard: a mismatch means query
    vectors and stored vectors come from different spaces, and a legacy
    (unstamped) collection under a DOCUMENT-transforming profile means the
    stored vectors were embedded from raw text while the profile's queries
    target transformed documents. Both degrade retrieval silently, so recall
    should fail loud with this message instead.

    A query-only transform (e.g. Qwen3's instruction) is NOT an
    incompatibility: legacy raw document vectors are byte-identical to what
    the active profile would produce, and the transformed query is exactly
    how the family is meant to search them — that is the contract's
    "adding a query transform never requires reindexing".
    """
    return _space_verdict(store, provider)[0]


def _space_verdict(store: Any, provider: Any) -> tuple[str | None, str | None]:
    """(error, validated_fingerprint) for one (store, provider) pair.

    The fingerprint is the SAME resolution the verdict was computed from —
    callers that stamp points must use it verbatim instead of re-resolving,
    or a tag repointed between two lookups would pass the guard under space
    A and stamp space B.
    """
    status, expected, found = check_document_space(store, provider)
    if status == "mismatch":
        return (
            f"collection points are stamped with embedding space {found}, but "
            f"the active provider embeds in {expected} — operating on this "
            "collection would cross vectors from different spaces; fix the "
            "embedding config or reindex the collection"
        ), expected
    if status == "legacy":
        profile = getattr(provider, "profile", None)
        doc_tf = getattr(profile, "document_transform", None) or {}
        if doc_tf.get("kind", "identity") != "identity":
            return (
                "existing points carry no embedding-space fingerprint (embedded "
                "from raw text by an older mnemostack), but the active profile "
                f"'{getattr(profile, 'name', 'unknown')}' transforms documents "
                "before embedding — the stored vectors are not in this space. "
                "Reindex the collection, or register an identity profile for "
                "this model to keep the legacy behavior"
            ), expected
        legacy_ok = getattr(provider, "_legacy_space_compatible", None)
        if legacy_ok is not None and not legacy_ok():
            return (
                "existing points carry no embedding-space fingerprint (embedded "
                "by an older mnemostack under its then-default provider "
                "settings), and the active provider configuration does not "
                "reproduce that default (e.g. last-token pooling replacing the "
                "old mean default) — the stored vectors are not in this space. "
                "Reindex the collection, or configure the provider to the "
                "legacy settings to keep the old behavior"
            ), expected
    return None, expected


def check_document_space(
    store: Any, provider: Any, *, sample_size: int = 16
) -> tuple[str, str | None, str | None]:
    """Compare the provider's document-space fingerprint against stored points.

    Samples up to ``sample_size`` points (lazy scroll — never loads the
    collection). Returns ``(status, expected, found)`` where status is:

    - ``"match"`` — every sampled point carries the current fingerprint;
    - ``"mismatch"`` — a sampled point was embedded under a DIFFERENT
      document space; indexing must stop (recreate or use a new collection);
    - ``"legacy"`` — the sample contains at least one point WITHOUT a
      fingerprint (indexed before fingerprints existed) and no mismatching
      one. Legacy dominates match on purpose: an unstamped point may have
      been embedded from raw text, so under a document-transforming profile
      the caller must treat the collection as legacy even when other points
      already match;
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

    Every built-in path that stamps fingerprints guards first: the CLI index
    commands call this before embedding, and `Ingestor` /
    `upsert_markdown_chunks` self-guard per flush/invocation via
    :class:`SpaceGuard`, so a mutable tag repointed under a long-lived
    writer is refused at the next batch instead of writing mixed spaces.
    """
    # Scroll support first: a store that cannot be inspected cannot be
    # guarded at all, so the (potentially failing) fingerprint lookup must
    # not run — otherwise an Ollama-compatible endpoint without /api/tags
    # would break the advertised legacy-store fallback.
    scroll = getattr(store, "scroll", None)
    if scroll is None:
        return "unknown", None, None
    expected = document_space_fingerprint_via(provider)
    if expected is None:
        return "unknown", None, None
    sampled = 0
    unstamped = 0
    found: str | None = None
    for hit in islice(scroll(), sample_size):
        sampled += 1
        fp = (getattr(hit, "payload", None) or {}).get(EMBEDDING_SPACE_KEY)
        if fp is None:
            unstamped += 1
            continue
        found = str(fp)
        if found != expected:
            return "mismatch", expected, found
    if sampled == 0:
        return "empty", expected, None
    if unstamped:
        return "legacy", expected, found
    return "match", expected, found
