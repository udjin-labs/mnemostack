"""
Tests for the reranker cache.
"""

import threading
import time

from mnemostack.reranker_cache import RerankerCache


class TestRerankerCache:
    def test_cache_miss(self):
        """Missing key returns None and increments misses."""
        cache = RerankerCache(ttl_seconds=60)
        assert cache.get("q", ["1"]) is None
        assert cache.stats.misses == 1
        assert cache.stats.hits == 0

    def test_cache_hit(self):
        """Stored value is returned on hit."""
        cache = RerankerCache(ttl_seconds=60)
        data = [{"id": "1", "score": 0.9}]
        cache.put("q", ["1"], data)
        result = cache.get("q", ["1"])
        assert result == data
        assert cache.stats.hits == 1

    def test_order_sensitive(self):
        """Candidate ID order affects cache key."""
        cache = RerankerCache(ttl_seconds=60)
        cache.put("q", ["b", "a"], [1])
        assert cache.get("q", ["a", "b"]) is None
        assert cache.get("q", ["b", "a"]) == [1]

    def test_expiry(self):
        """Entries expire after TTL."""
        cache = RerankerCache(ttl_seconds=0.1)
        cache.put("q", ["1"], [1])
        assert cache.get("q", ["1"]) == [1]
        time.sleep(0.15)
        assert cache.get("q", ["1"]) is None

    def test_max_size_eviction(self):
        """Cache evicts oldest entries when full."""
        cache = RerankerCache(ttl_seconds=60, max_size=3)
        for i in range(5):
            cache.put(f"q{i}", [f"{i}"], [i])
        assert len(cache) <= 3
        assert cache.stats.evictions >= 2

    def test_clear(self):
        """Clear empties the cache."""
        cache = RerankerCache(ttl_seconds=60)
        cache.put("q", ["1"], [1])
        cache.clear()
        assert len(cache) == 0
        assert cache.get("q", ["1"]) is None

    def test_thread_safety(self):
        """Concurrent writes don't raise."""
        cache = RerankerCache(ttl_seconds=60, max_size=100)
        errors = []

        def worker(thread_id):
            try:
                for i in range(50):
                    cache.put(f"q{thread_id}_{i}", [f"{i}"], [i])
                    cache.get(f"q{thread_id}_{i}", [f"{i}"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_deterministic_key(self):
        """Same inputs always produce the same cache key."""
        key1 = RerankerCache._make_key("query", ["a", "b", "c"])
        key2 = RerankerCache._make_key("query", ["a", "b", "c"])
        assert key1 == key2

        key3 = RerankerCache._make_key("different", ["a", "b", "c"])
        assert key1 != key3  # different query

    def test_candidate_order_affects_key(self):
        """Candidate order matters because reranked output preserves input order."""
        key1 = RerankerCache._make_key("query", ["a", "b", "c"])
        key2 = RerankerCache._make_key("query", ["c", "a", "b"])
        assert key1 != key2

    def test_no_separator_collision(self):
        """IDs containing delimiter chars don't cause key collisions."""
        key1 = RerankerCache._make_key("q", ["a|b", "c"])
        key2 = RerankerCache._make_key("q", ["a", "b|c"])
        assert key1 != key2

        # Also test with colon (used in length prefix)
        key3 = RerankerCache._make_key("q", ["a:1", "b"])
        key4 = RerankerCache._make_key("q", ["a", "1:b"])
        assert key3 != key4

    def test_hit_rate(self):
        """Hit rate is computed correctly."""
        cache = RerankerCache(ttl_seconds=60)
        assert cache.stats.hit_rate == 0.0  # no calls yet

        cache.put("q", ["1"], [1])
        cache.get("q", ["1"])  # hit
        cache.get("q", ["2"])  # miss
        assert cache.stats.hit_rate == 50.0
