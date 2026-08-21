"""Per-tenant request rate limiting — token bucket + RateLimiter."""

from __future__ import annotations

import pytest

from mnemostack.quotas import RateLimitExceededError, TenantQuota
from mnemostack.ratelimit import RateLimiter, TokenBucket


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class _Store:
    """Minimal mutable QuotaStore for tests."""

    def __init__(self, quotas=None):
        self.quotas = dict(quotas or {})

    def get(self, tenant):
        return self.quotas.get(tenant)


# ---------- TokenBucket ----------


def test_token_bucket_burst_then_throttles():
    clock = _Clock()
    b = TokenBucket(rate=2.0, capacity=3.0, clock=clock)
    assert b.acquire() == 0.0  # starts full: 3 -> 2
    assert b.acquire() == 0.0  # 2 -> 1
    assert b.acquire() == 0.0  # 1 -> 0
    wait = b.acquire()  # empty: need 1 token at 2/s
    assert wait == pytest.approx(0.5)
    clock.t = 0.5  # one token refilled
    assert b.acquire() == 0.0


def test_token_bucket_refill_caps_at_capacity():
    clock = _Clock()
    b = TokenBucket(rate=1.0, capacity=2.0, clock=clock)
    b.acquire()
    b.acquire()  # empty
    clock.t = 100.0  # long idle — refill must cap at capacity, not 100 tokens
    assert b.acquire() == 0.0
    assert b.acquire() == 0.0
    assert b.acquire() > 0  # only 2 were banked


def test_token_bucket_reconfigure_clamps_tokens():
    clock = _Clock()
    b = TokenBucket(rate=1.0, capacity=10.0, clock=clock)  # starts full (10)
    b.reconfigure(rate=1.0, capacity=3.0)  # shrink capacity -> clamp to 3
    assert b.acquire() == 0.0
    assert b.acquire() == 0.0
    assert b.acquire() == 0.0
    assert b.acquire() > 0


# ---------- RateLimiter ----------


def test_rate_limiter_no_quota_never_limits():
    lim = RateLimiter(_Store(), clock=_Clock())
    for _ in range(100):
        lim.check("acme")  # tenant has no quota -> unlimited


def test_rate_limiter_none_tenant_is_noop():
    store = _Store({"acme": TenantQuota(max_rps=1.0, burst=1)})
    lim = RateLimiter(store, clock=_Clock())
    for _ in range(100):
        lim.check(None)  # unscoped request -> never limited


def test_rate_limiter_throttles_over_rate():
    clock = _Clock()
    store = _Store({"acme": TenantQuota(max_rps=1.0, burst=2)})
    lim = RateLimiter(store, clock=clock)
    lim.check("acme")
    lim.check("acme")  # burst of 2
    with pytest.raises(RateLimitExceededError) as ei:
        lim.check("acme")
    assert ei.value.tenant == "acme"
    assert ei.value.retry_after == pytest.approx(1.0)
    clock.t = 1.0  # one token refilled at 1/s
    lim.check("acme")


def test_rate_limiter_isolates_tenants():
    clock = _Clock()
    store = _Store({
        "a": TenantQuota(max_rps=1.0, burst=1),
        "b": TenantQuota(max_rps=1.0, burst=1),
    })
    lim = RateLimiter(store, clock=clock)
    lim.check("a")
    lim.check("b")  # b's bucket is independent of a's
    with pytest.raises(RateLimitExceededError):
        lim.check("a")


def test_rate_limiter_picks_up_raised_quota_after_ttl():
    clock = _Clock()
    store = _Store({"acme": TenantQuota(max_rps=1.0, burst=1)})
    lim = RateLimiter(store, clock=clock, cache_ttl=5.0)
    lim.check("acme")  # burst of 1 consumed at t=0
    with pytest.raises(RateLimitExceededError):
        lim.check("acme")  # empty at t=0
    store.quotas["acme"] = TenantQuota(max_rps=100.0, burst=100)  # operator raises it
    # Still t=0: within the cache TTL the old (tight) config is used and the
    # bucket is empty, so the raised quota isn't visible yet.
    with pytest.raises(RateLimitExceededError):
        lim.check("acme")
    clock.t = 6.0  # past the TTL -> the raised quota now applies
    lim.check("acme")


def test_rate_limiter_dropping_quota_removes_bucket():
    clock = _Clock()
    store = _Store({"acme": TenantQuota(max_rps=1.0, burst=1)})
    lim = RateLimiter(store, clock=clock, cache_ttl=0.0)  # always re-resolve
    lim.check("acme")
    with pytest.raises(RateLimitExceededError):
        lim.check("acme")
    store.quotas["acme"] = TenantQuota(max_rps=None)  # rate limit cleared
    lim.check("acme")  # no longer limited
    assert "acme" not in lim._buckets
