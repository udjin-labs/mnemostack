"""`serve --retry-on-weak` — ask a nearly-empty recall again, in other words.

The paraphrasing machinery has always been in the stack and the answer
path already retried its own sub-recalls; `/recall` had no policy for
"this returned nothing, try saying it differently". This adds one, opt-in,
bounded, and with the scoping of the original call carried through
unchanged.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from test_remote_ingest import _ingest_app

from mnemostack.recall.retry import (
    DEFAULT_WEAK_BELOW,
    MAX_VARIANTS,
    is_weak,
    retry_weak_recall,
)


class _Hit:
    def __init__(self, pid, text="a memory"):
        self.id = pid
        self.text = text
        self.score = 0.9
        self.payload = {}
        self.sources = ["vector"]


class _LLM:
    """Returns paraphrases, one per line, like the real expander expects."""

    def __init__(self, text="how did we decide auth\nwhat was chosen for login"):
        self.text = text
        self.calls = 0

    def generate(self, prompt, max_tokens=200, temperature=0.0):
        self.calls += 1
        return type("R", (), {"ok": True, "text": self.text, "tokens_used": 5})()


def _flow(monkeypatch, by_query):
    """Patch recall_flow inside the retry module; record what it was given."""
    seen: list[tuple[str, dict]] = []

    def _fake(_recaller, query, _limit, **kwargs):
        seen.append((query, kwargs))
        return list(by_query.get(query, []))

    import mnemostack.recall.retry as retry_mod

    monkeypatch.setattr(retry_mod, "recall_flow", _fake)
    return seen


# ------------------------------------------------------------- the policy


def test_weakness_is_a_count_not_a_score():
    """Fused scores are RRF values — 1/(k+rank) — so they encode position,
    not confidence: one irrelevant hit scores exactly as high as one
    perfect hit. The trigger is how little came back."""
    assert is_weak([]) is True
    assert is_weak([_Hit(1)]) is False
    assert is_weak([_Hit(1)], below=2) is True
    assert is_weak([_Hit(1), _Hit(2)], below=2) is False
    assert DEFAULT_WEAK_BELOW == 1  # only an EMPTY recall, by default


def test_a_healthy_recall_is_never_retried(monkeypatch):
    llm = _LLM()
    seen = _flow(monkeypatch, {})
    results = [_Hit(1)]
    out, retried = retry_weak_recall(None, "q", 10, llm=llm, results=results)
    assert out == results and retried is False
    assert llm.calls == 0 and seen == []  # not a token spent


def test_an_empty_recall_is_asked_again_in_other_words(monkeypatch):
    llm = _LLM()
    seen = _flow(monkeypatch, {"how did we decide auth": [_Hit(7)]})
    out, retried = retry_weak_recall(None, "auth decision?", 10, llm=llm, results=[])
    assert retried is True
    assert [r.id for r in out] == [7]
    assert [q for q, _ in seen] == ["how did we decide auth", "what was chosen for login"]


def test_the_retry_carries_the_callers_scope_unchanged(monkeypatch):
    """A retry that widened scope would be a tenant leak wearing a
    feature's clothes."""
    seen = _flow(monkeypatch, {})
    scope = {
        "filters": {"index_root": "/a"},
        "tenant": "acme",
        "as_of": "2026-01-01T00:00:00+00:00",
        "include_invalidated": False,
        "token_budget": 512,
    }
    retry_weak_recall(None, "q", 10, llm=_LLM(), results=[], **scope)
    assert seen, "the retry never recalled"
    for _query, kwargs in seen:
        for key, value in scope.items():
            assert kwargs[key] == value, key


def test_the_retry_is_bounded_to_one_round(monkeypatch):
    """Two paraphrases, one pass — never a ladder that escalates while a
    query keeps failing."""
    llm = _LLM("one\ntwo\nthree\nfour\nfive")
    seen = _flow(monkeypatch, {})
    retry_weak_recall(None, "q", 10, llm=_LLM(), results=[])
    assert len(seen) <= MAX_VARIANTS
    seen.clear()
    retry_weak_recall(None, "q", 10, llm=llm, results=[])
    assert len(seen) == MAX_VARIANTS


def test_duplicates_across_variants_are_merged_once(monkeypatch):
    _flow(
        monkeypatch,
        {"how did we decide auth": [_Hit(1), _Hit(2)], "what was chosen for login": [_Hit(2)]},
    )
    out, retried = retry_weak_recall(None, "q", 10, llm=_LLM(), results=[])
    assert retried and [r.id for r in out] == [1, 2]


def test_the_merged_result_respects_the_limit(monkeypatch):
    _flow(monkeypatch, {"how did we decide auth": [_Hit(i) for i in range(10)]})
    out, _retried = retry_weak_recall(None, "q", 3, llm=_LLM(), results=[])
    assert len(out) == 3


def test_without_an_llm_the_recall_stands(monkeypatch):
    """The paraphrase needs a model; a deployment without one gets its
    original results, not an error."""
    seen = _flow(monkeypatch, {})
    out, retried = retry_weak_recall(None, "q", 10, llm=None, results=[])
    assert out == [] and retried is False and seen == []


def test_a_failing_retry_is_not_a_failing_recall(monkeypatch):
    """Whatever goes wrong in the second pass, the caller still gets the
    first pass's answer."""

    class _Boom:
        def generate(self, *_a, **_k):
            raise RuntimeError("llm exploded")

    out, retried = retry_weak_recall(None, "q", 10, llm=_Boom(), results=[])
    assert out == [] and retried is False

    def _explode(*_a, **_k):
        raise RuntimeError("store down")

    import mnemostack.recall.retry as retry_mod

    monkeypatch.setattr(retry_mod, "recall_flow", _explode)
    out, retried = retry_weak_recall(None, "q", 10, llm=_LLM(), results=[])
    assert out == [] and retried is True  # asked, found nothing, said so


def test_asking_again_and_still_finding_nothing_is_an_answer(monkeypatch):
    _flow(monkeypatch, {})
    out, retried = retry_weak_recall(None, "q", 10, llm=_LLM(), results=[])
    assert out == [] and retried is True


# ------------------------------------------------------------- the surface


def _recall(client, keys, **body):
    return client.post(
        "/recall",
        json={"query": "anything", "limit": 5, **body},
        headers={"X-API-Key": keys["read"]},
    )


def test_the_server_does_not_retry_unless_asked(monkeypatch, tmp_path):
    import mnemostack.server as srv

    calls: list = []
    monkeypatch.setattr(
        srv, "retry_weak_recall", lambda *a, **k: calls.append(k) or ([], False)
    )
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    monkeypatch.setattr(srv, "recall_flow", lambda *_a, **_k: [])
    assert _recall(TestClient(app), keys).status_code == 200
    assert calls == []


def test_the_operator_switch_turns_it_on(monkeypatch, tmp_path):
    import mnemostack.server as srv

    calls: list = []
    monkeypatch.setattr(
        srv, "retry_weak_recall", lambda *a, **k: calls.append(k) or ([_Hit(3)], True)
    )
    app, _store, _emb, keys = _ingest_app(
        monkeypatch, tmp_path, cfg_extra={"retry_on_weak": True}
    )
    monkeypatch.setattr(srv, "recall_flow", lambda *_a, **_k: [])
    r = _recall(TestClient(app), keys)
    assert r.status_code == 200
    assert [m["id"] for m in r.json()["results"]] == ["3"]
    assert len(calls) == 1
    assert calls[0]["tenant"] == "alpha"  # the key's tenant, carried through


def test_a_request_can_opt_out_where_the_operator_opted_in(monkeypatch, tmp_path):
    import mnemostack.server as srv

    calls: list = []
    monkeypatch.setattr(
        srv, "retry_weak_recall", lambda *a, **k: calls.append(k) or ([], False)
    )
    app, _store, _emb, keys = _ingest_app(
        monkeypatch, tmp_path, cfg_extra={"retry_on_weak": True}
    )
    monkeypatch.setattr(srv, "recall_flow", lambda *_a, **_k: [])
    assert _recall(TestClient(app), keys, retry_on_weak=False).status_code == 200
    assert calls == []


def test_a_request_cannot_turn_it_on_where_the_operator_did_not(monkeypatch, tmp_path):
    """The client pays nothing for this; the SERVER pays for the LLM call
    and the extra retrieval, so enabling it is the operator's decision."""
    import mnemostack.server as srv

    calls: list = []
    monkeypatch.setattr(
        srv, "retry_weak_recall", lambda *a, **k: calls.append(k) or ([], False)
    )
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    monkeypatch.setattr(srv, "recall_flow", lambda *_a, **_k: [])
    assert _recall(TestClient(app), keys, retry_on_weak=True).status_code == 200
    assert calls == []
