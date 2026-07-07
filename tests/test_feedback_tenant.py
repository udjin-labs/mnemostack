"""Per-tenant learning state — Q-table and IoR log partitioned by tenant.

The learning state is a shared store keyed by stage name (``q_table``,
``ior_log``); ``q_table`` is keyed by ``(source, query_type)`` — labels identical
across tenants — so without partitioning one tenant's feedback shifts the Q-value
every other tenant reranks against. These pin that each tenant reads/writes its
own partition, and that ``tenant=None`` keeps the bare (single-tenant) keys.
"""

from __future__ import annotations

from mnemostack.recall.pipeline.base import PipelineContext
from mnemostack.recall.pipeline.stages import InhibitionOfReturn, QLearningReranker
from mnemostack.recall.pipeline.state import InMemoryStateStore, tenant_state_key
from mnemostack.recall.recaller import RecallResult


def test_tenant_state_key_namespacing():
    assert tenant_state_key("q_table", None) == "q_table"  # unscoped = bare key
    assert tenant_state_key("q_table", "acme") == "q_table:acme"
    assert tenant_state_key("q_table", "a") != tenant_state_key("q_table", "b")
    # tenant containing the delimiter can't forge another tenant's key
    assert tenant_state_key("q_table", "a:b") != tenant_state_key("q_table", "a")


def _ctx(tenant=None, query_type="general"):
    c = PipelineContext(query="q")
    c.query_type = query_type
    if tenant is not None:
        c.extras["tenant"] = tenant
    return c


def test_qlearning_feedback_is_tenant_isolated():
    store = InMemoryStateStore()
    q = QLearningReranker(store, use_blend=True)
    # Tenant A gives strong positive feedback for the vector source.
    for _ in range(20):
        q.record_feedback("vector", "general", reward=1.0, tenant="acme")

    def _score(tenant):
        r = RecallResult(id="x", text="t", score=0.5, payload={}, sources=["vector"])
        q.apply(_ctx(tenant=tenant), [r])
        return r.payload["q_value"]

    # A sees the lifted Q-value; B (and unscoped) see the untouched cold-start
    # value (identical to a fresh table — A's feedback didn't leak into theirs).
    q_a = _score("acme")
    q_b = _score("other")
    q_unscoped = _score(None)
    assert q_a > 0.7  # moved up by A's own feedback
    assert q_b == q_unscoped  # B behaves exactly like a fresh single-tenant table
    assert q_a > q_b  # A's feedback did not leak into B's ranking
    # The store holds a per-tenant partition, not a shared q_table.
    assert "q_table:acme" in store._data
    assert store.get("q_table") in (None, {})  # unscoped table untouched


def test_qlearning_unscoped_uses_bare_key():
    store = InMemoryStateStore()
    q = QLearningReranker(store)
    q.record_feedback("vector", "general", reward=1.0, tenant=None)
    assert "q_table" in store._data
    assert not any(k.startswith("q_table:") for k in store._data)


def test_ior_is_tenant_isolated():
    store = InMemoryStateStore()
    ior = InhibitionOfReturn(store, penalty_per_recall=0.1, max_penalty=0.5)
    # Tenant A recalls id "m" several times.
    for _ in range(3):
        ior.record_recall("m", tenant="acme")

    def _penalized_score(tenant):
        r = RecallResult(id="m", text="t", score=1.0, payload={}, sources=["vector"])
        ior.apply(_ctx(tenant=tenant), [r])
        return r.score

    # A's own recalls penalize A; B (and unscoped) are untouched.
    assert _penalized_score("acme") < 1.0
    assert _penalized_score("other") == 1.0
    assert _penalized_score(None) == 1.0
    assert "ior_log:acme" in store._data


def test_legacy_stage_signature_works_unscoped():
    # A custom stage with the old record_recall(memory_id) / record_feedback(
    # source, query_type, reward) signature must still work in a single-tenant
    # (tenant=None) run — no tenant kwarg is passed.
    from mnemostack.feedback import record_feedback_events, record_recall_events

    recalls: list[str] = []
    fb: list[tuple] = []

    class _LegacyStage:
        def record_recall(self, memory_id):  # no tenant kwarg
            recalls.append(memory_id)

        def record_feedback(self, source, query_type, reward):  # no tenant kwarg
            fb.append((source, query_type, reward))

    r = RecallResult(id="x", text="t", score=1.0, payload={}, sources=["vector"])
    record_recall_events([_LegacyStage()], [r], tenant=None)
    record_feedback_events([_LegacyStage()], ["vector"], "general", 1.0, tenant=None)
    assert recalls == ["x"] and fb == [("vector", "general", 1.0)]


def test_apply_feedback_threads_tenant_into_state():
    from mnemostack.feedback import apply_feedback

    store = InMemoryStateStore()
    pipeline = [QLearningReranker(store), InhibitionOfReturn(store)]
    apply_feedback(
        pipeline, hit_id="h", signal="clicked", query="who is alice",
        sources=["vector"], reward=1.0, tenant="acme",
    )
    # Both blobs landed in the tenant's partition, none in the bare keys.
    assert "q_table:acme" in store._data and "ior_log:acme" in store._data
    assert "q_table" not in store._data and "ior_log" not in store._data
