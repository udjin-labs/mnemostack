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


def _floor_recaller(n=1):
    """A recaller exposing the REAL vector-floor logic and nothing else."""
    from mnemostack.recall.recaller import Recaller

    class _R:
        vector_floor = n
        _apply_vector_floor = Recaller._apply_vector_floor
        _vector_floor_candidates_from_results = staticmethod(
            Recaller._vector_floor_candidates_from_results
        )
        apply_vector_floor_after_rerank = Recaller.apply_vector_floor_after_rerank

    return _R()


def _with_candidates(hit, *candidates):
    hit.payload["_vector_floor_candidates"] = [
        {"id": c, "text": f"floored {c}", "score": 0.99, "payload": {}, "sources": ["vector"]}
        for c in candidates
    ]
    return hit


def test_the_vector_floor_survives_the_fusion(monkeypatch):
    """R10 (codex P2): `vector_floor` is a promise that N raw vector hits
    reach the caller even when the ranking stages would drop them, and
    each pass keeps it by returning MORE than `limit` — the extras are
    appended past the cut. Fusing those lists back down to `limit` revoked
    a guarantee the operator configured, so a retried recall honoured a
    weaker contract than the same recall left alone."""
    _flow(
        monkeypatch,
        {
            "how did we decide auth": [_with_candidates(_Hit("A"), "F")],
            "what was chosen for login": [_with_candidates(_Hit("B"), "F")],
        },
    )
    out, retried = retry_weak_recall(_floor_recaller(), "q", 1, llm=_LLM(), results=[])
    assert retried is True
    assert [r.id for r in out] == ["A", "F"]  # past `limit`, exactly as the floor intends
    assert out[0].score > out[1].score  # ...and still descending


def test_the_budget_still_caps_a_floored_retry(monkeypatch):
    """The floor is applied BEFORE the budget, the same order `recall_flow`
    uses: a guarantee about which memories reach the caller does not get to
    overrun the hard cap on how many tokens do."""
    from mnemostack.recall.tokens import sum_tokens

    one = _Hit("A", text="word " * 50)
    budget = sum_tokens([one], None)
    _flow(
        monkeypatch,
        {
            "how did we decide auth": [_with_candidates(_Hit("A", "word " * 50), "F")],
            "what was chosen for login": [],
        },
    )
    out, _retried = retry_weak_recall(
        _floor_recaller(), "q", 1, llm=_LLM(), results=[], token_budget=budget
    )
    assert [r.id for r in out] == ["A"]  # the floored extra did not fit
    assert sum_tokens(out, None) <= budget


def test_a_retry_never_hands_back_less_than_it_was_given(monkeypatch):
    """R11 (review P1): the budget is a hard cap and the fusion reorders by
    rank, so together they can COST the caller memories they already had.
    A corroborated newcomer takes the front, and the greedy trim — which
    stops at the first item that would overflow — then evicts the smaller
    memories that fitted perfectly well in the original order. A feature
    whose entire premise is "that was too little" must not answer by
    returning less: no shrinkage, and the caller's own scores survive."""
    from mnemostack.recall.tokens import sum_tokens

    small, large = _Hit("S", text="word " * 8), _Hit("H", text="word " * 100)
    newcomer_text = "word " * 104
    budget = sum_tokens([small, large], None) + 2
    # The setup this defect needs: the caller's two memories fit with room
    # to spare, the newcomer fits on its own, and the newcomer plus the
    # smallest of theirs does not.
    assert sum_tokens([small, large], None) < budget
    assert sum_tokens([_Hit("N", text=newcomer_text)], None) <= budget
    assert sum_tokens([_Hit("N", text=newcomer_text), small], None) > budget
    _flow(
        monkeypatch,
        {  # both phrasings corroborate N, so RRF ranks it above S and H
            "how did we decide auth": [_Hit("N", text=newcomer_text)],
            "what was chosen for login": [_Hit("N", text=newcomer_text)],
        },
    )
    out, retried = retry_weak_recall(
        None,
        "q",
        10,
        llm=_LLM(),
        results=[small, large],
        below=5,
        token_budget=budget,
    )
    assert retried is True
    assert [r.id for r in out] == ["S", "H"]  # not ["N"]
    assert (small.score, large.score) == (0.9, 0.9)  # ...with their own scores


def test_the_env_deployment_can_set_the_threshold_too(monkeypatch):
    """R11 (review P3): `MNEMOSTACK_RETRY_ON_WEAK` let an env-configured
    deployment (`uvicorn mnemostack.server:app`, no CLI) switch the feature
    on while the threshold stayed pinned at the default — a knob you can
    turn on but not tune. Bad values fall back rather than fail startup:
    a typo in one tuning knob must not take the service down."""
    from mnemostack.server import ServerConfig

    monkeypatch.setenv("MNEMOSTACK_RETRY_ON_WEAK", "1")
    monkeypatch.setenv("MNEMOSTACK_RETRY_WEAK_BELOW", "3")
    cfg = ServerConfig.from_env()
    assert cfg.retry_on_weak is True and cfg.retry_weak_below == 3

    for bad in ("nonsense", "0", "-2", ""):
        monkeypatch.setenv("MNEMOSTACK_RETRY_WEAK_BELOW", bad)
        assert ServerConfig.from_env().retry_weak_below == 1


def _serve_args(**overrides):
    """A `serve` namespace like the CLI builds, minus what a test varies."""
    import argparse

    base = dict(
        provider="fake",
        embedding_model=None,
        llm="fake-llm",
        llm_model=None,
        collection="test",
        qdrant="http://localhost:6333",
        memgraph_uri=None,
        graph_timeout=5.0,
        qdrant_health_timeout=2,
        bm25_path=[],
        state_path="/tmp/state.json",
        vector_floor=0,
        rerank_mode="relevant_only",
        token_budget=None,
        auto_record_ior=False,
        auth=False,
        keys_file=None,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _serve_cfg(monkeypatch, **overrides):
    import sys
    from unittest.mock import MagicMock, patch

    import mnemostack.server as srv
    from mnemostack.cli import cmd_serve

    monkeypatch.setitem(sys.modules, "uvicorn", MagicMock())
    captured = {}

    def _fake_build_app(cfg):
        captured["cfg"] = cfg
        return MagicMock()

    with patch.object(srv, "build_app", _fake_build_app):
        cmd_serve(_serve_args(**overrides))
    return captured["cfg"]


def test_serve_honors_the_weak_threshold_env(monkeypatch):
    """R12 (codex P2): round 11 gave the threshold an env var and wired it
    into `ServerConfig.from_env()` — the programmatic-ASGI path. `serve`
    builds its config explicitly and never calls `from_env`, so the knob
    the README had just documented did nothing on the entry point the
    README points at first. CLI wins, env fills in, default is last."""
    monkeypatch.setenv("MNEMOSTACK_RETRY_ON_WEAK", "1")
    monkeypatch.setenv("MNEMOSTACK_RETRY_WEAK_BELOW", "4")

    cfg = _serve_cfg(monkeypatch, retry_on_weak=False, retry_weak_below=None)
    assert cfg.retry_on_weak is True and cfg.retry_weak_below == 4  # env fills in

    cfg = _serve_cfg(monkeypatch, retry_on_weak=True, retry_weak_below=2)
    assert cfg.retry_weak_below == 2  # ...but an explicit flag wins over it

    monkeypatch.delenv("MNEMOSTACK_RETRY_WEAK_BELOW")
    cfg = _serve_cfg(monkeypatch, retry_on_weak=True, retry_weak_below=None)
    assert cfg.retry_weak_below == 1  # neither: the conservative default


def test_a_better_evidenced_hit_takes_a_weak_one_s_slot(monkeypatch):
    """R13 (review P1): the never-shrink rule is about the SIZE of the
    response, not its membership, and this pins the difference so nobody
    later "fixes" it into strict containment. A memory the original pass
    ranked LAST can lose its slot to hits that each rank first in their own
    pass — that is RRF doing exactly what fusing the rounds is for. The
    alternative rules are both worse: pinning every original would seat the
    caller's weakest hits ahead of better ones, and making room for them
    would overrun the `limit` they asked for."""
    early, weak = _Hit("E"), _Hit("W")
    early.score, weak.score = 0.95, 0.5
    _flow(
        monkeypatch,
        {
            "how did we decide auth": [_Hit("N")],  # each new hit ranks
            "what was chosen for login": [_Hit("M")],  # first in its own pass
        },
    )
    out, retried = retry_weak_recall(None, "q", 3, llm=_LLM(), results=[early, weak], below=3)
    assert retried is True
    assert [r.id for r in out] == ["E", "N", "M"]  # W displaced, not lost to a bug
    assert len(out) > len([early, weak])  # ...and the caller ends up with MORE


def test_one_pass_gets_one_vote(monkeypatch):
    """R14 (codex P2): RRF adds `1/(k+rank)` once per list an item appears
    in, so a pass that listed the same memory twice — `1` here and `"1"`
    there — handed it two contributions and could out-vote a hit two
    separate phrasings genuinely corroborated. Canonicalising the objects
    is not enough: the fusion then sees the same OBJECT twice and scores
    it twice."""
    seen = _flow(
        monkeypatch,
        {
            "how did we decide auth": [_Hit("A"), _Hit("C")],
            "what was chosen for login": [_Hit("C")],  # C: corroborated
        },
    )
    out, retried = retry_weak_recall(
        None,
        "q",
        3,
        llm=_LLM(),
        results=[_Hit(1), _Hit("1")],  # one memory, listed twice
        below=5,
    )
    assert retried is True and len(seen) == MAX_VARIANTS
    assert [str(r.id) for r in out] == ["C", "1", "A"]  # not ["1", "C", "A"]


def test_a_later_pass_contributes_its_metadata_to_the_shared_memory(monkeypatch):
    """PR #167 (bot P2): keeping the first arrival must not mean discarding
    what later passes learned about that memory. A paraphrase may reach it
    through a different arm, and may be the pass whose results carry the
    vector floor's candidate list — dropping either under-credits the
    documented `sources` field (so feedback reinforces the wrong arms) or
    loses the floor's pool when only the later pass held it."""
    first = _Hit("A")
    first.sources = ["bm25"]
    later = _with_candidates(_Hit("A"), "F")
    later.sources = ["vector"]
    later.payload["raw_vector_score"] = 0.77
    _flow(
        monkeypatch,
        {"how did we decide auth": [first], "what was chosen for login": [later]},
    )
    out, _retried = retry_weak_recall(_floor_recaller(), "q", 1, llm=_LLM(), results=[])
    kept = out[0]
    assert kept is first  # the first arrival is still the object returned...
    assert kept.sources == ["bm25", "vector"]  # ...credited with both arms
    assert kept.payload["raw_vector_score"] == 0.77  # ...and the later payload
    assert [r.id for r in out] == ["A", "F"]  # the floor's pool survived too


def test_the_caller_s_objects_come_back_as_they_arrived(monkeypatch):
    """The merge mutates the caller's own objects in place — score, and now
    arms and payload — so the no-op path has to undo all three, not just
    the score it used to."""
    from mnemostack.recall.tokens import sum_tokens

    small, large = _Hit("S", text="word " * 8), _Hit("H", text="word " * 100)
    small.sources = ["bm25"]
    budget = sum_tokens([small, large], None) + 2
    newcomer_text = "word " * 104
    _flow(
        monkeypatch,
        {
            "how did we decide auth": [_Hit("N", text=newcomer_text)],
            "what was chosen for login": [_with_candidates(_Hit("S"), "F")],
        },
    )
    out, retried = retry_weak_recall(
        None, "q", 10, llm=_LLM(), results=[small, large], below=5, token_budget=budget
    )
    assert retried is True and [r.id for r in out] == ["S", "H"]
    assert small.score == 0.9  # score restored...
    assert small.sources == ["bm25"]  # ...arms restored...
    assert "_vector_floor_candidates" not in small.payload  # ...payload restored


def test_a_failing_retry_shows_up_as_degraded_service(monkeypatch):
    """PR #167 (bot P2): `/status.degraded_events` sums an explicit
    allowlist, so an enabled retry policy could fail on every single
    request while `/metrics` counted the failures and `/status` reported
    healthy — next to `followup_rewrite_failed`, which is the same class of
    failure and was already covered."""
    from mnemostack.server import _DEGRADED_METRICS

    assert "mnemostack.recall.weak_retry_failed" in _DEGRADED_METRICS
    assert "mnemostack.recall.weak_retry_unavailable" in _DEGRADED_METRICS

    from mnemostack.observability.recorder import (
        InMemoryRecorder,
        NullRecorder,
        set_recorder,
    )

    def _boom(*_a, **_k):
        raise RuntimeError("paraphrase arm down")

    import mnemostack.recall.retry as retry_mod

    monkeypatch.setattr(retry_mod, "recall_flow", _boom)
    rec = InMemoryRecorder()
    set_recorder(rec)
    try:
        out, retried = retry_weak_recall(None, "q", 10, llm=_LLM(), results=[])
    finally:
        set_recorder(NullRecorder())
    assert out == [] and retried is True  # fail-open: still not an error
    failed = rec.counters.get(("mnemostack.recall.weak_retry_failed",))
    assert failed == float(MAX_VARIANTS)  # once per arm that raised


def test_a_floor_extra_does_not_vote_in_the_fusion(monkeypatch):
    """PR #167 (bot P2): `recall_flow` returns the ranked page AND, past
    it, the vector floor's guaranteed candidates — items placed there
    precisely BECAUSE the ranking did not choose them. Letting them vote
    inverts the floor: the same candidate appended to both passes collects
    two RRF contributions, out-votes each pass's actual rank-one winner,
    and at `limit=1` becomes the sole survivor — after which the final
    floor step has nothing left to add and BOTH real winners are gone."""
    floored = _floor_recaller()

    def _pass(winner):
        # what recall_flow returns: the ranked page, then the floor's
        # extra — marked, as the floor marks what it appends
        extra = _Hit("F", text="floored F")
        extra.from_vector_floor = True
        return [_with_candidates(_Hit(winner), "F"), extra]

    _flow(
        monkeypatch,
        {"how did we decide auth": _pass("A"), "what was chosen for login": _pass("B")},
    )
    out, retried = retry_weak_recall(floored, "q", 1, llm=_LLM(), results=[])
    assert retried is True
    assert [r.id for r in out] == ["A", "F"]  # not ["F"]
    assert out[0].score > out[1].score  # the floor rides behind the winner


def test_a_pass_that_dies_still_reports_what_it_saw(monkeypatch):
    """PR #167 (bot P2): a variant that records retriever entries or marks
    a degradation and THEN raises had all of it discarded by `continue`.
    The request returned success with a trace and a `degraded` field that
    mentioned nothing, leaving a process-wide counter as the only evidence
    that a whole pass had collapsed."""
    from mnemostack.recall.trace import RecallTrace, RetrieverTrace

    def _fake(_recaller, query, _limit, **kwargs):
        trace = kwargs.get("trace")
        if trace is not None:
            trace.retrievers.append(RetrieverTrace(name="vector"))
            trace.mark("bm25:down")
        raise RuntimeError("pipeline stage exploded")

    import mnemostack.recall.retry as retry_mod

    monkeypatch.setattr(retry_mod, "recall_flow", _fake)
    trace = RecallTrace()
    out, retried = retry_weak_recall(None, "q", 10, llm=_LLM(), results=[], trace=trace)
    assert out == [] and retried is True  # still fail-open
    assert [e.name for e in trace.retrievers] == ["vector:retry"] * MAX_VARIANTS
    assert [e.query for e in trace.retrievers] == [
        "how did we decide auth",
        "what was chosen for login",
    ]
    assert "bm25:down" in trace.degraded  # the fault the pass actually hit


def test_both_passes_floor_pools_reach_the_floor(monkeypatch):
    """PR #167 (bot P2): with the floor on, EVERY result of a pass carries
    that pass's whole candidate pool, so two passes both holding the key is
    the normal case — not a collision to settle by seniority. Keeping only
    the incumbent's stranded a stronger vector hit that a later paraphrase
    alone saw: a floor extra is not a ranking, so it cannot vote, and the
    pool was the one route it had left."""
    first = _with_candidates(_Hit("A"), "F1")
    later = _Hit("A")
    later.payload["_vector_floor_candidates"] = [
        {"id": "F2", "text": "stronger", "score": 0.995, "payload": {}, "sources": ["vector"]}
    ]
    _flow(
        monkeypatch,
        {"how did we decide auth": [first], "what was chosen for login": [later]},
    )
    out, _retried = retry_weak_recall(_floor_recaller(n=2), "q", 1, llm=_LLM(), results=[])
    assert [r.id for r in out] == ["A", "F2", "F1"]  # strongest of the union first


def test_the_reranker_is_not_credited_with_the_merge(monkeypatch):
    """PR #167 (bot P2): `post_rerank` means "the reranker's order when a
    reranker ran" — a claim about one component's output on the candidates
    it was given, and its contract already allows differing from the
    response. No reranker ever saw the cross-query merge, so copying the
    fused order into it credited the reranker with an ordering it never
    produced, precisely when a retry succeeded."""
    from mnemostack.recall.trace import RecallTrace

    trace = RecallTrace()
    trace.post_rerank = []  # the caller's own pass reranked nothing
    _flow(monkeypatch, {"how did we decide auth": [_Hit("N"), _Hit("M")]})
    out, retried = retry_weak_recall(None, "q", 5, llm=_LLM(), results=[], trace=trace)
    assert retried is True and [r.id for r in out] == ["N", "M"]
    assert trace.fused == [(str(r.id), r.score) for r in out]  # the response order
    assert trace.post_rerank == []  # ...and the reranker keeps its own truth


def test_the_retry_spend_names_the_tenant(monkeypatch, tmp_path):
    """PR #167 (bot P2): the only server counter marking a request that
    actually paid for a paraphrase and a second retrieval carried no tenant
    label, and `tenant.requests` cannot recover the attribution — it counts
    healthy recalls and per-request opt-outs alike. An operator could see
    the retry spend but not whose it was."""
    import mnemostack.server as srv
    from mnemostack.observability.recorder import (
        InMemoryRecorder,
        NullRecorder,
        set_recorder,
    )

    app, _store, _emb, keys = _ingest_app(monkeypatch, tmp_path, cfg_extra={"retry_on_weak": True})
    monkeypatch.setattr(srv, "recall_flow", lambda *_a, **_k: [])
    monkeypatch.setattr(srv, "retry_weak_recall", lambda *a, **k: ([_Hit(3)], True))
    rec = InMemoryRecorder()
    set_recorder(rec)
    try:
        assert _recall(TestClient(app), keys).status_code == 200
    finally:
        set_recorder(NullRecorder())
    retried = [k for k in rec.counters if k[0] == "mnemostack.server.recall_retried"]
    assert retried, "the retry counter was never recorded"
    assert ("tenant", "alpha") in retried[0], retried[0]  # the key's tenant


def test_the_floor_weighs_every_pass_that_ran_not_only_the_winners(monkeypatch):
    """PR #167 (bot P2): a pass's floor pool travels on its hits, so when
    fusion cuts that pass's hit the pool goes with it — and the strongest
    raw vector candidate of the whole retry could be discarded because the
    paraphrase that found it lost a tie. Disjoint ids never meet in
    `_merged_candidates`, so the same-id union fixed the other half of this
    and left this half: what the floor is owed is what the retry SAW."""
    a = _with_candidates(_Hit("A"), "F1")  # F1 scores 0.99
    b = _Hit("B")
    b.payload["_vector_floor_candidates"] = [
        {"id": "F2", "text": "strongest", "score": 0.999, "payload": {}, "sources": ["vector"]}
    ]
    _flow(monkeypatch, {"how did we decide auth": [a], "what was chosen for login": [b]})
    out, retried = retry_weak_recall(_floor_recaller(), "q", 1, llm=_LLM(), results=[])
    assert retried is True
    # A wins the single slot on the RRF tie; the floor still owes the
    # caller the strongest candidate the retry saw, which B's pass found.
    assert [r.id for r in out] == ["A", "F2"]


def test_the_floor_pool_speaks_the_survivors_id_language(monkeypatch):
    """PR #167 (bot P2): the floor's candidate pool was the one place in
    this module still keyed on the RAW id. A memory offered as `"1"` by one
    pass and `1` by another therefore kept BOTH entries, and since
    `Recaller._apply_vector_floor` dedupes against the results natively, it
    could not recognise the string candidate as the integer survivor
    already on the page — appending it, and showing one memory twice."""
    int_hit = _Hit(1)
    int_hit.payload["_vector_floor_candidates"] = [
        {"id": 1, "text": "weak", "score": 0.5, "payload": {}, "sources": ["vector"]}
    ]
    other = _Hit("X")
    other.payload["_vector_floor_candidates"] = [
        {"id": "1", "text": "strong", "score": 0.99, "payload": {}, "sources": ["vector"]}
    ]
    _flow(
        monkeypatch,
        {"how did we decide auth": [int_hit], "what was chosen for login": [other]},
    )
    out, retried = retry_weak_recall(_floor_recaller(), "q", 1, llm=_LLM(), results=[])
    assert retried is True
    assert [str(r.id) for r in out] == ["1"]  # one memory, one row — not ["1", "1"]


def test_every_id_this_module_keys_on_goes_through_one_rule(monkeypatch):
    """The class, not the position. Three separate places each shipped a
    bug by answering "same memory?" their own way, so the rule now has one
    name — and this asserts nothing bypasses it, since a fourth place with
    a fresh answer is the shape every one of those bugs had."""
    import ast
    import inspect

    import mnemostack.recall.retry as retry_mod

    def _converts_to_str(node):
        """Python's string-conversion forms — a closed set, so it CAN be
        named exhaustively, where a set of *values* could not be."""
        if isinstance(node, ast.JoinedStr):  # f"{x}"
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):  # "%s" % x
            return isinstance(node.left, ast.Constant) and isinstance(node.left.value, str)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"str", "repr", "format"}:
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr == "format":
                return True
        return False

    def _stringifies(node):
        """...applied to an id. A label built from a retriever's NAME is
        not a second notion of identity; `str(some.id)` is."""
        return _converts_to_str(node) and any(
            isinstance(part, ast.Attribute) and part.attr == "id" for part in ast.walk(node)
        )

    tree = ast.parse(inspect.getsource(retry_mod))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name == "_memory_key":
            continue
        for inner in ast.walk(node):
            if _stringifies(inner):
                offenders.append(f"{node.name}:{inner.lineno}")
    assert not offenders, (
        f"an id is stringified outside `_memory_key`: {offenders} — key through it instead"
    )


def test_the_pool_itself_holds_one_entry_per_memory():
    """The rule at the level it is stated, not only through the floor: a
    pool that keeps `1` and `"1"` as two candidates is a second notion of
    identity, whatever a later stage happens to do about it."""
    from mnemostack.recall.retry import _merged_candidates

    weak = {"id": 1, "text": "weak", "score": 0.5, "payload": {}, "sources": ["vector"]}
    strong = {"id": "1", "text": "strong", "score": 0.99, "payload": {}, "sources": ["vector"]}
    merged = _merged_candidates([weak], [strong])
    assert len(merged) == 1 and merged[0]["text"] == "strong"
    assert _merged_candidates([strong], [weak])[0]["text"] == "strong"  # order-free


def test_a_floor_extra_does_not_vote_on_a_short_page_either(monkeypatch):
    """PR #167 (bot P2): position was the wrong way to spot an appended
    floor hit. `recall_flow` appends them right after the ranked page, so
    when that page is SHORT — and a weak recall, the only kind this module
    sees, is exactly when it is — the extras sit at ordinary indices and
    vote anyway. At `limit=2` an `F` appended to both passes outranked both
    genuine rank-one hits and evicted one of them entirely."""

    def _pass(winner):
        extra = _Hit("F", text="floored F")
        extra.from_vector_floor = True
        return [_with_candidates(_Hit(winner), "F"), extra]  # one ranked hit, one extra

    _flow(
        monkeypatch,
        {"how did we decide auth": _pass("A"), "what was chosen for login": _pass("B")},
    )
    out, retried = retry_weak_recall(_floor_recaller(), "q", 2, llm=_LLM(), results=[])
    assert retried is True
    assert [r.id for r in out] == ["A", "B", "F"]  # not ["F", "A"] — B survives


def test_the_floor_marks_what_it_appended():
    """The producer side of that rule. Anything downstream that treats a
    result list as a ranking depends on this marker to tell the ranked
    page from the guaranteed extras, and position cannot tell them apart
    once the page is short."""
    ranked = _Hit("A")
    ranked.payload["_vector_floor_candidates"] = [
        {"id": "F", "text": "floored", "score": 0.99, "payload": {}, "sources": ["vector"]}
    ]
    out = _floor_recaller().apply_vector_floor_after_rerank([ranked], [ranked])
    assert [r.id for r in out] == ["A", "F"]
    assert not getattr(out[0], "from_vector_floor", False)  # the ranking chose A
    assert out[1].from_vector_floor is True  # the floor appended F


def test_the_pool_merge_weighs_what_the_floor_weighs():
    """Local review P1: `_apply_vector_floor` OVERWRITES `.score` on the
    candidates it appends (`floor_score * 0.999`, seeded by that pass's own
    ranked page) and orders by `payload["raw_vector_score"]` for exactly
    that reason. Deduping the pool by the top-level score compared a real
    similarity in one pass against a rank artefact in another — and could
    drop the stronger observation while promising "strongest kept"."""
    from mnemostack.recall.retry import _merged_candidates

    artefact = {  # weak hit whose .score was inflated by its pass's floor
        "id": "M",
        "text": "weak",
        "score": 0.9,
        "payload": {"raw_vector_score": 0.3},
        "sources": ["vector"],
    }
    genuine = {  # the real, stronger observation of the same memory
        "id": "M",
        "text": "strong",
        "score": 0.4,
        "payload": {"raw_vector_score": 0.95},
        "sources": ["vector"],
    }
    assert _merged_candidates([artefact], [genuine])[0]["text"] == "strong"
    assert _merged_candidates([genuine], [artefact])[0]["text"] == "strong"


def _floor_flow(with_pipeline):
    """Run the real `recall_flow` over a recall whose floor appended one
    item, with and without a stage that re-evaluates the page."""
    from mnemostack.recall.flow import recall_flow
    from mnemostack.recall.recaller import RecallResult

    floored = _floor_recaller()

    class _Recaller:
        vector_floor = 1
        apply_vector_floor_after_rerank = staticmethod(floored.apply_vector_floor_after_rerank)

        def recall(self, _query, limit=10, **_kw):
            ranked = RecallResult(
                id="A",
                text="a",
                score=0.9,
                payload={
                    "_vector_floor_candidates": [
                        {"id": "C", "text": "c", "score": 0.4, "payload": {}, "sources": ["vector"]}
                    ]
                },
                sources=["bm25"],
            )
            # what `recall` really hands back: its OWN floor already ran
            return floored.apply_vector_floor_after_rerank([ranked], [ranked])

    pipeline = None
    if with_pipeline:

        class _Pipeline:
            def apply(self, _q, results, **_kw):
                return results  # scored and placed every candidate it saw

        pipeline = _Pipeline()
    return recall_flow(_Recaller(), "q", 5, pipeline=pipeline)


def test_the_floor_marker_outlives_a_recall_that_did_not_rerank_it():
    """PR #167 (local review P1 + codex P2): the marker may only be cleared
    by something that actually RE-RANKED the item, and nothing on the
    recall path can say that. A pipeline can be composed entirely of
    non-scoring stages (`ClassifyQuery` alone is legal), and
    `apply_rerank_safe` is fail-open — a configured reranker that raised or
    kept the input order leaves a reranker present and nothing reranked.
    Both were tried as proxies for "this was ranked" and both were wrong,
    so the marker now simply persists: a floor-only item stays labelled
    whether or not stages were configured, and the retry clears it the
    moment a pass genuinely ranks that memory."""
    for with_pipeline in (True, False):
        page = {r.id: r for r in _floor_flow(with_pipeline=with_pipeline)}
        assert page["A"].from_vector_floor is False  # the ranking chose A
        assert page["C"].from_vector_floor is True, (
            f"floor-only item lost its marker (pipeline={with_pipeline}); it would "
            "then vote in pass 0 as a ranked hit"
        )


def test_corroboration_clears_the_floor_marker(monkeypatch):
    """Local review P1: a memory that reached the page only because the
    floor guaranteed it is exactly what a paraphrase can overturn. When a
    pass RANKS it, it is not a floor extra any more — whichever pass
    ranked it."""
    floor_only = _Hit("F")
    floor_only.from_vector_floor = True
    floor_only.score = 0.3
    ranked_by_paraphrase = _Hit("F")  # same memory, genuinely ranked
    _flow(monkeypatch, {"how did we decide auth": [ranked_by_paraphrase]})
    out, retried = retry_weak_recall(None, "q", 5, llm=_LLM(), results=[floor_only], below=3)
    assert retried is True
    assert out[0] is floor_only  # the incumbent object is what comes back...
    assert out[0].from_vector_floor is False  # ...no longer calling itself an extra


def test_the_restore_covers_every_field_the_merge_touches(monkeypatch):
    """Local review P1: round 1 promised the no-op path hands back the
    caller's objects as they arrived, then round 6 added a mutable field
    without extending that promise. A restore that covers all but the
    newest channel is how the promise starts quietly meaning something
    else again."""
    from mnemostack.recall.tokens import sum_tokens

    small, large = _Hit("S", text="word " * 8), _Hit("H", text="word " * 100)
    small.from_vector_floor = True  # the caller's own floor-guaranteed row
    budget = sum_tokens([small, large], None) + 2
    # The paraphrase re-finds S as a RANKED hit — which clears the marker
    # mid-merge — behind a newcomer big enough to make the trim shrink the
    # page below what the caller arrived with, so the no-op path runs and
    # has to put back a field something really did change.
    _flow(
        monkeypatch,
        {"how did we decide auth": [_Hit("N", text="word " * 104), _Hit("S")]},
    )
    out, _retried = retry_weak_recall(
        None, "q", 10, llm=_LLM(), results=[small, large], below=5, token_budget=budget
    )
    assert [r.id for r in out] == ["S", "H"]  # the no-op path ran
    assert small.from_vector_floor is True  # ...and gave the field back too


def test_weakness_is_counted_in_memories():
    """Local review P2 / bot P2: the gate deciding whether to ask again was
    the last place counting rows. One memory under two id representations
    is one memory found — reading it as two calls a genuinely weak recall
    healthy and skips the retry entirely."""
    assert is_weak([_Hit(1), _Hit("1")], below=2) is True
    assert is_weak([_Hit(1), _Hit("2")], below=2) is False


def test_the_room_question_still_counts_rows():
    """...and the neighbouring gate is NOT the same oversight: it asks
    whether the RESPONSE has space left, and a response is made of rows.
    Pinned so the next reader does not "fix" it into agreement."""
    assert has_room([_Hit(1), _Hit("1")], limit=2) is False


def test_an_unreadable_raw_score_falls_back_the_way_the_floor_does():
    """PR #167 (codex P2): `Recaller._raw_score` falls back to the
    candidate's own score when `raw_vector_score` will not parse. Returning
    zero here instead made the pool disagree with the floor about which
    observation of a memory is stronger — and the disagreement, not the bad
    value, is what picks a different memory than the floor would."""
    from mnemostack.recall.retry import _candidate_score, _merged_candidates

    unreadable = {
        "id": "M",
        "text": "strong",
        "score": 0.9,
        "payload": {"raw_vector_score": "not-a-number"},
        "sources": ["vector"],
    }
    weaker = {
        "id": "M",
        "text": "weak",
        "score": 0.4,
        "payload": {"raw_vector_score": 0.4},
        "sources": ["vector"],
    }
    assert _candidate_score(unreadable) == 0.9  # not 0.0
    assert _merged_candidates([weaker], [unreadable])[0]["text"] == "strong"


def test_the_documented_cost_is_the_cost_actually_paid(monkeypatch):
    """PR #167 (bot P2): the CLI help, the request field, the README and
    the CHANGELOG all quoted the retry's price, and all quoted it wrong —
    "an LLM call plus another retrieval round". Each variant repeats the
    caller's recall in FULL, reranker included, so the real bill is one
    paraphrase call plus a rerank per variant. A spend knob whose stated
    price is a third of the real one is not an informed opt-in, so the
    number is pinned here and the docs are written from it."""
    calls = {"paraphrase": 0, "rerank": 0, "retrieval": 0}

    class _CountingLLM(_LLM):
        def generate(self, prompt, max_tokens=200, temperature=0.0):
            calls["paraphrase"] += 1
            return super().generate(prompt, max_tokens, temperature)

    class _Reranker:
        def rerank(self, _query, results):
            calls["rerank"] += 1
            return results

    def _fake(_recaller, query, _limit, **kwargs):
        calls["retrieval"] += 1
        reranker = kwargs.get("reranker")
        if reranker is not None:  # what recall_flow does with it
            reranker.rerank(query, [])
        return []

    import mnemostack.recall.retry as retry_mod

    monkeypatch.setattr(retry_mod, "recall_flow", _fake)
    retry_weak_recall(None, "q", 5, llm=_CountingLLM(), results=[], reranker=_Reranker())

    assert calls["paraphrase"] == 1  # one call generates both variants
    assert calls["rerank"] == MAX_VARIANTS  # ...but each variant reranks
    assert calls["retrieval"] == MAX_VARIANTS
    assert calls["paraphrase"] + calls["rerank"] == 3  # the number the docs quote
