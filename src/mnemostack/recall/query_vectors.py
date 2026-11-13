"""Query vectors shared only within one recall."""

from __future__ import annotations

from collections.abc import Callable
from threading import Event, Lock
from typing import Any

from ..embeddings import EmbeddingProvider
from ..embeddings.roles import embed_query_via


class QueryVectors:
    def __init__(self, *, wait_for_pending: bool = True):
        self._wait_for_pending = wait_for_pending
        self._lock = Lock()
        self._entries: dict[tuple[int, str], tuple[Event, list[float] | None]] = {}

    def embed(
        self, provider: EmbeddingProvider, query: str,
        embed: Callable[[Any, str], list[float]],
    ) -> list[float]:
        key = (id(provider), query)
        with self._lock:
            first = key not in self._entries
            ready, _ = self._entries.setdefault(key, (Event(), None))
        if not first:
            if self._wait_for_pending and not ready.is_set():
                ready.wait()
            with self._lock:
                _, vector = self._entries[key]
            if vector:
                return list(vector)
        try:
            vector = embed(provider, query)
            if vector:
                with self._lock:
                    self._entries[key] = (ready, list(vector))
            return vector
        finally:
            if first:
                ready.set()


def query_vector(
    provider: EmbeddingProvider, query: str, recall_scope: dict[Any, Any] | None,
    embed: Callable[[Any, str], list[float]] = embed_query_via,
) -> list[float]:
    cache = recall_scope.get(QueryVectors) if recall_scope is not None else None
    if isinstance(cache, QueryVectors):
        return cache.embed(provider, query, embed)
    return embed(provider, query)
