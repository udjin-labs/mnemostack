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
    has_room,
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


def test_a_memory_two_phrasings_agree_on_ranks_first(monkeypatch):
    """The passes are fused, not concatenated — so a hit both paraphrases
    found outranks one only a single phrasing did, and it still appears
    exactly once."""
    _flow(
        monkeypatch,
        {"how did we decide auth": [_Hit(1), _Hit(2)], "what was chosen for login": [_Hit(2)]},
    )
    out, retried = retry_weak_recall(None, "q", 10, llm=_LLM(), results=[])
    assert retried and [r.id for r in out] == [2, 1]


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
    monkeypatch.setattr(srv, "retry_weak_recall", lambda *a, **k: calls.append(k) or ([], False))
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
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path, cfg_extra={"retry_on_weak": True})
    monkeypatch.setattr(srv, "recall_flow", lambda *_a, **_k: [])
    r = _recall(TestClient(app), keys)
    assert r.status_code == 200
    assert [m["id"] for m in r.json()["results"]] == ["3"]
    assert len(calls) == 1
    assert calls[0]["tenant"] == "alpha"  # the key's tenant, carried through


def test_a_request_can_opt_out_where_the_operator_opted_in(monkeypatch, tmp_path):
    import mnemostack.server as srv

    calls: list = []
    monkeypatch.setattr(srv, "retry_weak_recall", lambda *a, **k: calls.append(k) or ([], False))
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path, cfg_extra={"retry_on_weak": True})
    monkeypatch.setattr(srv, "recall_flow", lambda *_a, **_k: [])
    assert _recall(TestClient(app), keys, retry_on_weak=False).status_code == 200
    assert calls == []


def test_a_request_cannot_turn_it_on_where_the_operator_did_not(monkeypatch, tmp_path):
    """The client pays nothing for this; the SERVER pays for the LLM call
    and the extra retrieval, so enabling it is the operator's decision."""
    import mnemostack.server as srv

    calls: list = []
    monkeypatch.setattr(srv, "retry_weak_recall", lambda *a, **k: calls.append(k) or ([], False))
    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path)
    monkeypatch.setattr(srv, "recall_flow", lambda *_a, **_k: [])
    assert _recall(TestClient(app), keys, retry_on_weak=True).status_code == 200
    assert calls == []


def test_no_room_means_no_retry(monkeypatch):
    """R1 (codex P2): with a threshold above the caller's limit, a FULL
    page still counts as "below the threshold" — and every hit the retry
    found would be appended past the limit and cut away again. Real spend,
    identical answer, and a recovery counter that lied about it."""
    llm = _LLM()
    seen = _flow(monkeypatch, {"how did we decide auth": [_Hit(9)]})
    full_page = [_Hit(1), _Hit(2)]
    out, retried = retry_weak_recall(None, "q", 2, llm=llm, results=full_page, below=5)
    assert out == full_page and retried is False
    assert llm.calls == 0 and seen == []
    # ...but a page that is short DOES get asked again.
    out, retried = retry_weak_recall(None, "q", 5, llm=llm, results=full_page, below=5)
    # Fused, so the retry's top hit ties with the original's top hit and
    # the original's second falls below both.
    assert retried is True and sorted(r.id for r in out) == [1, 2, 9]
    assert [r.id for r in out][:2] == [1, 9]


def test_the_token_budget_holds_after_the_merge(monkeypatch):
    """R1 (codex P2): each variant's flow capped its OWN results, and two
    lists that each fit the budget concatenate into one that does not. The
    budget is documented as a hard cap on the response."""
    long_hit = lambda pid: _Hit(pid, text="word " * 200)  # noqa: E731
    _flow(
        monkeypatch,
        {
            "how did we decide auth": [long_hit(1)],
            "what was chosen for login": [long_hit(2)],
        },
    )
    from mnemostack.recall.tokens import sum_tokens

    out, retried = retry_weak_recall(None, "q", 10, llm=_LLM(), results=[], token_budget=120)
    assert retried is True
    assert sum_tokens(out, None) <= 120, [r.id for r in out]


def test_answer_does_not_pay_for_a_paraphrase_round(monkeypatch, tmp_path):
    """R1 (codex P2): the generator already runs inference and expansion
    retries over a fresh sub-recall, and AnswerRequest has no field to
    decline this one — so /answer must not add a paraphrase round in
    front of its own."""
    import mnemostack.server as srv

    calls: list = []
    monkeypatch.setattr(srv, "retry_weak_recall", lambda *a, **k: calls.append(k) or ([], False))

    class _Gen:
        def generate(self, *_a, **_k):
            from mnemostack.recall.answer import Answer

            return Answer(text="an answer", confidence=0.9)

    monkeypatch.setattr(srv, "AnswerGenerator", lambda *_a, **_k: _Gen())
    app, _store, _emb, keys = _ingest_app(
        monkeypatch, tmp_path, cfg_extra={"retry_on_weak": True}, llm=object()
    )
    monkeypatch.setattr(srv, "recall_flow", lambda *_a, **_k: [])
    r = TestClient(app).post(
        "/answer", json={"query": "anything"}, headers={"X-API-Key": keys["read"]}
    )
    assert r.status_code == 200, r.text
    assert calls == []


def test_the_trace_describes_what_was_returned(monkeypatch):
    """R2 (codex P2): every variant recalled into the CALLER's trace, so
    `fused` ended up describing the last paraphrase while the response
    carried the merge — and `trace.fused` is documented as the order
    recall returned."""
    from mnemostack.recall.trace import RecallTrace

    def _fake(_recaller, query, _limit, **kwargs):
        trace = kwargs.get("trace")
        if trace is not None:  # each pass writes its own order, as flow does
            trace.fused = [(f"{query}-1", 0.5)]
            trace.retrievers.append(
                type("RT", (), {"name": "vector", "ranked": [], "to_dict": dict})()
            )
        return [_Hit(9)] if query == "how did we decide auth" else []

    import mnemostack.recall.retry as retry_mod

    monkeypatch.setattr(retry_mod, "recall_flow", _fake)
    trace = RecallTrace()
    trace.fused = [("original", 0.1)]
    first_pass = type("RT", (), {"name": "vector", "ranked": [], "to_dict": dict})()
    trace.retrievers.append(first_pass)  # what the ORIGINAL recall recorded
    out, retried = retry_weak_recall(None, "q", 10, llm=_LLM(), results=[], trace=trace)
    assert retried is True and [r.id for r in out] == [9]
    # The order actually returned, carrying the FUSED score rather than
    # the one the pass happened to assign before fusion.
    assert trace.fused == [("9", out[0].score)]
    # the retry's retrieval work is visible, labelled as such...
    assert sum(rt.name.endswith(":retry") for rt in trace.retrievers) == MAX_VARIANTS
    # ...and the first pass's own entry is untouched, which only holds if
    # the variants recalled into traces of their own.
    assert first_pass.name == "vector"


def test_a_degradation_during_the_retry_is_not_swallowed(monkeypatch):
    """An arm that failed in the second pass failed; hiding it because of
    WHEN it happened is the opposite of what a trace is for."""
    from mnemostack.recall.trace import RecallTrace

    def _fake(_recaller, _query, _limit, **kwargs):
        trace = kwargs.get("trace")
        if trace is not None:
            trace.mark("bm25:down")
        return []

    import mnemostack.recall.retry as retry_mod

    monkeypatch.setattr(retry_mod, "recall_flow", _fake)
    trace = RecallTrace()
    retry_weak_recall(None, "q", 10, llm=_LLM(), results=[], trace=trace)
    assert "bm25:down" in trace.degraded


def test_the_caller_trace_is_restored_for_later_stages(monkeypatch):
    """The helper borrows the trace keyword while it runs; whoever called
    it must get its own object back in the kwargs it passed."""
    from mnemostack.recall.trace import RecallTrace

    _flow(monkeypatch, {})
    trace = RecallTrace()
    kwargs = {"trace": trace, "tenant": "acme"}
    retry_weak_recall(None, "q", 10, llm=_LLM(), results=[], **kwargs)
    assert kwargs["trace"] is trace


def test_every_paraphrase_gets_to_compete(monkeypatch):
    """R6 (review agent P2): once the passes are FUSED, a later paraphrase
    can outrank an earlier one — so stopping the loop when the page looks
    full (which rounds 3 and 4 did, correctly, while the merge was
    first-fit) would discard the better answer unread."""
    seen = _flow(
        monkeypatch,
        {
            "how did we decide auth": [_Hit(1), _Hit(2)],
            "what was chosen for login": [_Hit(3)],
        },
    )
    out, retried = retry_weak_recall(None, "q", 2, llm=_LLM(), results=[])
    assert retried is True
    assert len(seen) == MAX_VARIANTS  # both phrasings were asked
    assert [r.id for r in out] == [1, 3]  # each list's top hit outranks a second


def test_a_full_token_budget_leaves_no_room(monkeypatch):
    """R4 (codex P2): the room question has TWO dimensions, because the
    response is cut by both. A page under the item limit can still be
    spending the whole token budget, and every hit a paraphrase found
    would then be trimmed away — the same guaranteed-useless spend the
    count rule already refused."""
    long_hit = _Hit(1, text="word " * 200)
    from mnemostack.recall.tokens import sum_tokens

    used = sum_tokens([long_hit], None)
    assert has_room([long_hit], limit=10) is True  # room by count...
    assert has_room([long_hit], limit=10, token_budget=used) is False  # ...not by budget
    assert has_room([long_hit], limit=10, token_budget=used * 3) is True

    llm = _LLM()
    seen = _flow(monkeypatch, {"how did we decide auth": [_Hit(2)]})
    out, retried = retry_weak_recall(
        None, "q", 10, llm=llm, results=[long_hit], below=5, token_budget=used
    )
    assert out == [long_hit] and retried is False
    assert llm.calls == 0 and seen == []


def test_the_loop_stops_when_the_budget_fills_mid_retry(monkeypatch):
    """Same question inside the loop: the first paraphrase can fill the
    budget without filling the page."""
    from mnemostack.recall.tokens import sum_tokens

    big = _Hit(1, text="word " * 200)
    budget = sum_tokens([big], None)
    seen = _flow(
        monkeypatch,
        {"how did we decide auth": [big], "what was chosen for login": [_Hit(2)]},
    )
    out, retried = retry_weak_recall(None, "q", 10, llm=_LLM(), results=[], token_budget=budget)
    assert retried is True
    assert len(seen) == MAX_VARIANTS  # both phrasings compete for the budget
    assert sum_tokens(out, None) <= budget  # ...and the cap still holds


def test_a_retry_degradation_is_counted_once(monkeypatch):
    """R5 (codex P2): the variant's own trace already emitted the
    process-wide degradation counter, and folding the tag back with
    `mark()` emitted it a second time — `/status.degraded_events` and
    `/metrics` overreporting every retry-time failure."""
    from mnemostack.observability.recorder import (
        InMemoryRecorder,
        NullRecorder,
        set_recorder,
    )
    from mnemostack.recall.trace import DEGRADED_COUNTER, RecallTrace

    def _fake(_recaller, _query, _limit, **kwargs):
        trace = kwargs.get("trace")
        if trace is not None:
            trace.mark("bm25:down")  # a real fault, in the second pass
        return []

    import mnemostack.recall.retry as retry_mod

    monkeypatch.setattr(retry_mod, "recall_flow", _fake)
    rec = InMemoryRecorder()
    set_recorder(rec)
    try:
        trace = RecallTrace()
        retry_weak_recall(None, "q", 10, llm=_LLM(), results=[], trace=trace)
        emitted = rec.counter_value(DEGRADED_COUNTER, labels={"reason": "bm25:down"})
    finally:
        set_recorder(NullRecorder())
    assert "bm25:down" in trace.degraded  # still reported to the caller...
    assert emitted == MAX_VARIANTS  # ...and counted once per pass, not twice


def test_a_routine_note_from_a_retry_stays_a_note(monkeypatch):
    """Copying both lists keeps the variant's own classification, so this
    module never re-decides what counts as routine."""
    from mnemostack.recall.trace import RecallTrace

    def _fake(_recaller, _query, _limit, **kwargs):
        trace = kwargs.get("trace")
        if trace is not None:
            trace.mark("temporal:no_parse")  # routine, not a fault
        return []

    import mnemostack.recall.retry as retry_mod

    monkeypatch.setattr(retry_mod, "recall_flow", _fake)
    trace = RecallTrace()
    retry_weak_recall(None, "q", 10, llm=_LLM(), results=[], trace=trace)
    assert "temporal:no_parse" in trace.notes


def test_a_better_hit_from_a_later_phrasing_wins(monkeypatch):
    """R6 (review agent P2), the case that named the defect: the merge was
    arrival-ordered, so a poor hit from the first paraphrase held the only
    slot and the far better hit from the second was never even fetched."""
    _flow(
        monkeypatch,
        {
            "how did we decide auth": [_Hit("low", text="barely related")],
            "what was chosen for login": [_Hit("high", text="exactly the answer")],
        },
    )
    seen_ids = []
    for limit in (1, 2):
        out, retried = retry_weak_recall(None, "q", limit, llm=_LLM(), results=[])
        assert retried is True
        seen_ids.append([r.id for r in out])
    # With one slot the two tie at rank 1 and the first is kept; with two
    # slots BOTH are returned — the point is that the later phrasing is
    # fetched and competes, instead of being cut off unread.
    assert seen_ids[0] == ["low"]
    assert sorted(seen_ids[1]) == ["high", "low"]


def test_an_empty_paraphrase_list_is_not_a_retry(monkeypatch):
    """R6 (review agent P3): an LLM that answers with nothing usable left
    this branch untested — no test double ever returned an empty variant
    list without raising."""

    class _Silent:
        def generate(self, *_a, **_k):
            return type("R", (), {"ok": True, "text": "   \n\n  ", "tokens_used": 1})()

    seen = _flow(monkeypatch, {})
    out, retried = retry_weak_recall(None, "q", 10, llm=_Silent(), results=[])
    assert out == [] and retried is False
    assert seen == []  # nothing recalled on the strength of no paraphrase


def test_a_paraphrase_identical_to_the_query_is_not_asked_again(monkeypatch):
    """The expander dedups against the original, so a model that just
    echoes the question costs one LLM call and no extra retrieval."""

    class _Echo:
        def generate(self, *_a, **_k):
            return type("R", (), {"ok": True, "text": "q", "tokens_used": 1})()

    seen = _flow(monkeypatch, {})
    out, retried = retry_weak_recall(None, "q", 10, llm=_Echo(), results=[])
    assert out == [] and retried is False and seen == []


def test_the_returned_scores_describe_the_returned_order(monkeypatch):
    """R7 (codex P2): the fused score was computed and then thrown away,
    so results came back carrying their pre-fusion scores — numbers that
    do not describe the order they are printed in, and that a client
    sorting by score would use to undo the ranking."""
    _flow(
        monkeypatch,
        {
            "how did we decide auth": [_Hit(1), _Hit(2)],
            "what was chosen for login": [_Hit(2), _Hit(3)],
        },
    )
    out, retried = retry_weak_recall(None, "q", 10, llm=_LLM(), results=[])
    assert retried is True
    scores = [r.score for r in out]
    assert scores == sorted(scores, reverse=True), scores
    assert out[0].id == 2  # found by both phrasings, and it says so in its score
    assert scores[0] > scores[1]


def test_the_trace_scores_match_the_result_scores(monkeypatch):
    from mnemostack.recall.trace import RecallTrace

    _flow(
        monkeypatch,
        {"how did we decide auth": [_Hit(1)], "what was chosen for login": [_Hit(2)]},
    )
    trace = RecallTrace()
    out, _retried = retry_weak_recall(None, "q", 10, llm=_LLM(), results=[], trace=trace)
    assert trace.fused == [(str(r.id), r.score) for r in out]


def test_one_memory_stays_one_memory_across_id_types(monkeypatch):
    """R8 (review P2): a store may hand back `1` in one pass and `"1"` in
    the next — Qdrant point ids are `str | int` — and both times it means
    the same memory. Two notions of identity (a str-keyed "is this new"
    dict beside RRF's raw-id deduplication) split that memory in two: it
    takes two slots of the caller's page, so the page carries the same
    text twice and the trace reports the duplicate as the order recall
    returned."""
    seen = _flow(
        monkeypatch,
        {
            "how did we decide auth": [_Hit("1"), _Hit("2")],
            "what was chosen for login": [],
        },
    )
    from mnemostack.recall.trace import RecallTrace

    trace = RecallTrace()
    out, retried = retry_weak_recall(
        None, "q", 10, llm=_LLM(), results=[_Hit(1)], below=5, trace=trace
    )
    assert retried is True and len(seen) == MAX_VARIANTS
    assert [str(r.id) for r in out] == ["1", "2"]  # not ["1", "1", "2"]
    assert trace.fused == [(str(r.id), r.score) for r in out]


def test_a_second_phrasing_of_a_known_memory_is_one_memory(monkeypatch):
    """The same disagreement in its other direction: when a paraphrase
    returns a memory the caller already has under the OTHER id type, that
    is one memory corroborated, not two found. It takes one slot."""
    _flow(
        monkeypatch,
        {
            "how did we decide auth": [_Hit(1)],
            "what was chosen for login": [_Hit("1")],
        },
    )
    original = _Hit("1")
    out, retried = retry_weak_recall(None, "q", 10, llm=_LLM(), results=[original], below=5)
    assert retried is True
    assert out == [original]  # one row, and it is the caller's own object


def test_corroboration_re_ranks_even_when_nothing_new_is_found(monkeypatch):
    """R9 (codex P2): with `--retry-weak-below` above 1 the original list
    is not empty, and the paraphrases can come back holding only memories
    the caller already had — but in a DIFFERENT order. That is not
    "nothing": a memory both phrasings put first is better evidenced than
    one only the original found, which is the whole point of fusing the
    rounds. Gating the fusion on "did we see an unseen id" skipped it and
    returned the original order with pre-fusion scores."""
    a, b = _Hit("A"), _Hit("B")
    a.score, b.score = 0.9, 0.5  # the original pass preferred A
    seen = _flow(
        monkeypatch,
        {
            "how did we decide auth": [_Hit("B")],  # both paraphrases...
            "what was chosen for login": [_Hit("B")],  # ...prefer B
        },
    )
    out, retried = retry_weak_recall(None, "q", 10, llm=_LLM(), results=[a, b], below=5)
    assert retried is True and len(seen) == MAX_VARIANTS
    assert [r.id for r in out] == ["B", "A"]  # corroboration wins the top slot
    assert out[0].score > out[1].score  # ...and the scores say so


def test_nothing_at_all_leaves_the_caller_untouched(monkeypatch):
    """The other side of that rule: paraphrases that come back EMPTY have
    contributed no evidence, so there is nothing to fuse and the caller's
    own results and scores must survive verbatim."""
    a = _Hit("A")
    _flow(monkeypatch, {})  # neither phrasing retrieves anything
    out, retried = retry_weak_recall(None, "q", 10, llm=_LLM(), results=[a], below=5)
    assert retried is True
    assert out == [a] and a.score == 0.9  # no fused score written over it


def test_the_trace_says_which_paraphrase_produced_which_entry(monkeypatch):
    """R9 (codex P2): `:retry` in the name says an entry came from the
    retry, not WHICH of the two paraphrases produced its hits, latency or
    error. `RetrieverTrace.query` is the field for that."""
    from mnemostack.recall.trace import RecallTrace, RetrieverTrace

    def _fake(_recaller, query, _limit, **kwargs):
        trace = kwargs.get("trace")
        if trace is not None:
            trace.retrievers.append(RetrieverTrace(name="vector"))
            if query == "what was chosen for login":
                # an inner expansion recorded the text IT sent
                trace.retrievers.append(RetrieverTrace(name="bm25", query="inner"))
        return [_Hit(query)]

    import mnemostack.recall.retry as retry_mod

    monkeypatch.setattr(retry_mod, "recall_flow", _fake)
    trace = RecallTrace()
    retry_weak_recall(None, "q", 10, llm=_LLM(), results=[], trace=trace)
    labelled = [(e.name, e.query) for e in trace.retrievers]
    assert labelled == [
        ("vector:retry", "how did we decide auth"),
        ("vector:retry", "what was chosen for login"),
        ("bm25:retry", "inner"),  # the text that actually reached it, kept
    ]


def test_recovery_counts_memories_not_rows(monkeypatch):
    """The recovery counter answers "how much did the retry add", so it
    has to count what the merge counts. A caller list holding one memory
    under two id types is two rows and one memory: measuring the merge
    against the row count hides a genuine recovery behind the dedup."""
    from mnemostack.observability.recorder import (
        InMemoryRecorder,
        NullRecorder,
        set_recorder,
    )

    _flow(monkeypatch, {"how did we decide auth": [_Hit("N")]})
    rec = InMemoryRecorder()
    set_recorder(rec)
    try:
        out, _retried = retry_weak_recall(
            None, "q", 10, llm=_LLM(), results=[_Hit(1), _Hit("1")], below=5
        )
    finally:
        set_recorder(NullRecorder())
    assert [str(r.id) for r in out] == ["1", "N"]  # one memory, plus the new one
    assert rec.counters.get(("mnemostack.recall.weak_retry_recovered",)) == 1.0
