"""
In-memory cache for Gemini reranker results.

Deterministic hash-based cache with configurable TTL and max size.
Thread-safe for use in the recall daemon.
"""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache hit/miss statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Hit rate as a percentage (0-100)."""
        if self.total == 0:
            return 0.0
        return (self.hits / self.total) * 100


@dataclass
class _CacheEntry:
    """Internal cache entry with expiry."""
    value: Any
    expires_at: float


class RerankerCache:
    """
    Thread-safe in-memory cache for reranker results.

    Cache key is derived from SHA256(query + candidate IDs in input order).
    Uses length-prefixed encoding to prevent separator collisions.
    """

    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()
        self.stats = CacheStats()

    @staticmethod
    def _make_key(query: str, candidate_ids: list[str]) -> str:
        """Create a deterministic cache key from query + ordered candidate IDs.

        Uses length-prefixed encoding to avoid collisions when IDs
        contain delimiter characters.
        """
        # Length-prefix each component to prevent collisions
        parts = [f"{len(query)}:{query}"]
        for cid in candidate_ids:
            parts.append(f"{len(cid)}:{cid}")
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, query: str, candidate_ids: list[str]) -> Any | None:
        """Look up a cached reranker result. Returns None on miss or expiry."""
        key = self._make_key(query, candidate_ids)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self.stats.misses += 1
                return None
            if time.monotonic() > entry.expires_at:
                del self._cache[key]
                self.stats.misses += 1
                return None
            self.stats.hits += 1
            return entry.value

    def put(self, query: str, candidate_ids: list[str], value: Any) -> None:
        """Store a reranker result in the cache."""
        key = self._make_key(query, candidate_ids)
        expires_at = time.monotonic() + self.ttl_seconds
        with self._lock:
            # Evict oldest entries if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_oldest()
            self._cache[key] = _CacheEntry(value=value, expires_at=expires_at)

    def _evict_oldest(self) -> None:
        """Evict the entry closest to expiry (called under lock)."""
        if not self._cache:
            return
        oldest_key = min(self._cache, key=lambda k: self._cache[k].expires_at)
        del self._cache[oldest_key]
        self.stats.evictions += 1

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
