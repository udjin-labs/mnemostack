"""Per-tenant request rate limiting for the authenticated HTTP surface.

A tenant's ``max_rps`` (with an optional ``burst``) from the quota store is
enforced by a token bucket: each request spends one token, tokens refill at
``max_rps`` per second up to ``burst`` capacity. A tenant without a rate quota
(or an unauthenticated, tenant-less request) is never limited.

**Per-process, best-effort.** The buckets live in memory in one process. Running
N HTTP workers multiplies the effective ceiling by N (each keeps its own bucket),
exactly like the per-process metrics recorder and the storage-quota concurrency
caveat. For a hard global limit, front the service with a shared limiter (a
reverse proxy or a Redis token bucket). This layer stops a single worker from
being trivially flooded and gives each tenant an isolated share.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from .quotas import QuotaStore, RateLimitExceededError


class TokenBucket:
    """A thread-safe token bucket: ``rate`` tokens/sec, up to ``capacity`` tokens."""

    def __init__(self, rate: float, capacity: float, clock: Callable[[], float]):
        self.rate = rate
        self.capacity = capacity
        self._clock = clock
        self._tokens = float(capacity)  # start full: an idle tenant may burst
        self._updated = clock()
        self._lock = threading.Lock()

    def reconfigure(self, rate: float, capacity: float) -> None:
        """Apply a changed rate/capacity in place, clamping current tokens to it."""
        with self._lock:
            self._refill_locked()
            self.rate = rate
            self.capacity = capacity
            if self._tokens > capacity:
                self._tokens = capacity

    def _refill_locked(self) -> None:
        now = self._clock()
        elapsed = now - self._updated
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._updated = now

    def acquire(self, cost: float = 1.0) -> float:
        """Spend ``cost`` tokens if available. Returns 0.0 on success, else the
        seconds to wait before that many tokens will have refilled (no tokens are
        consumed when denied)."""
        with self._lock:
            self._refill_locked()
            if self._tokens >= cost:
                self._tokens -= cost
                return 0.0
            if self.rate <= 0:
                return float("inf")
            return (cost - self._tokens) / self.rate


class RateLimiter:
    """Enforces each tenant's ``max_rps`` via a per-tenant :class:`TokenBucket`.

    Tenant rate config is resolved from the ``QuotaStore`` and cached briefly
    (``cache_ttl`` seconds) so a live ``quota set`` takes effect without a restart
    while a hot path doesn't read the quota file on every request.

    The per-tenant bucket and config-cache maps grow one entry per distinct tenant
    seen and are not evicted — bounded by the number of authenticated tenants (a
    closed set issued via ``keys``), so not attacker-amplifiable.
    """

    def __init__(
        self,
        store: QuotaStore,
        *,
        clock: Callable[[], float] = time.monotonic,
        cache_ttl: float = 5.0,
    ):
        self._store = store
        self._clock = clock
        self._cache_ttl = cache_ttl
        self._buckets: dict[str, TokenBucket] = {}
        #: tenant -> (expiry_monotonic, (rate|None, capacity|None))
        self._cfg: dict[str, tuple[float, tuple[float | None, float | None]]] = {}
        self._lock = threading.Lock()

    def _resolve(self, tenant: str) -> tuple[float | None, float | None]:
        """(rate, capacity) for a tenant, cached for ``cache_ttl`` seconds."""
        now = self._clock()
        cached = self._cfg.get(tenant)
        if cached is not None and now < cached[0]:
            return cached[1]
        q = self._store.get(tenant)
        if q is None or q.max_rps is None:
            cfg: tuple[float | None, float | None] = (None, None)
        else:
            cfg = (q.max_rps, float(q.effective_burst() or 1))
        self._cfg[tenant] = (now + self._cache_ttl, cfg)
        return cfg

    def check(self, tenant: str | None) -> None:
        """Raise :class:`RateLimitExceededError` if ``tenant`` is over its rate.

        No-op for an unscoped (``None``) request or a tenant without a rate quota.
        """
        if tenant is None:
            return
        with self._lock:
            rate, capacity = self._resolve(tenant)
            if rate is None or capacity is None:
                self._buckets.pop(tenant, None)  # limit cleared — drop stale bucket
                return
            bucket = self._buckets.get(tenant)
            if bucket is None:
                # A tenant whose limit was cleared (or a fail-open on a transiently
                # unreadable quota file) drops its bucket above, so re-adding a limit
                # starts a fresh full bucket — one burst of leniency at the boundary,
                # acceptable for a best-effort guardrail.
                bucket = TokenBucket(rate, capacity, self._clock)
                self._buckets[tenant] = bucket
            elif bucket.rate != rate or bucket.capacity != capacity:
                bucket.reconfigure(rate, capacity)
        wait = bucket.acquire()
        if wait > 0:
            raise RateLimitExceededError(tenant, rate, wait)
