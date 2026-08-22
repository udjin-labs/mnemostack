"""Ageing for a memory that carries no event time.

`timestamp` is optional on ingest, so a whole class of writer never sends
one — the hermes connector did not until its 0.9.2. For those, this stage
had no age to work with and substituted a flat 0.5: the same value at a
day old and at a year old, which is not "a neutral default" but a memory
that never ages by anything at all. The write stamp the ingest pipeline
always sets is the honest last resort.
"""

from datetime import datetime, timedelta, timezone

import pytest

from mnemostack.recall import RecallResult
from mnemostack.recall.pipeline import FreshnessBlend, PipelineContext


def _ago(days: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _fresh(payload: dict, **kw) -> float:
    stage = FreshnessBlend(weight=0.2, **kw)
    out = stage.apply(
        PipelineContext(query="q"), [RecallResult(id="m", text="t", score=1.0, payload=payload)]
    )
    return out[0].payload["freshness"]


def test_a_memory_without_an_event_time_ages_by_when_it_was_written():
    """The defect this closes: without a fallback these four are the same
    number, so nothing distinguishes a note from today from one from last
    year."""
    curve = [_fresh({"indexed_at": _ago(d)}) for d in (0, 7, 30, 365)]
    assert all(b < a for a, b in zip(curve, curve[1:], strict=False)), curve
    assert curve[0] > 0.9 and curve[-1] < 0.05


def test_the_event_time_still_wins_when_there_is_one():
    """The fallback is a LAST resort — a memory that says when it happened
    must not be aged by when it was imported instead."""
    old_event_new_write = _fresh({"timestamp": _ago(365), "indexed_at": _ago(0)})
    new_event_old_write = _fresh({"timestamp": _ago(0), "indexed_at": _ago(365)})
    assert old_event_new_write < 0.05, old_event_new_write
    assert new_event_old_write > 0.4, new_event_old_write


def test_a_date_in_the_source_still_wins_over_the_write_stamp():
    """`source` dates were already preferred over nothing; the write stamp
    goes BELOW them, not above — a dated filename is evidence about the
    content, while the write stamp is only evidence about the import."""
    got = _fresh(
        {
            "source": f"notes/{(datetime.now(timezone.utc) - timedelta(days=365)).date()}-plan.md",
            "indexed_at": _ago(0),
        }
    )
    assert got < 0.05, got


def test_re_indexing_an_old_corpus_is_not_treated_as_conversation_echo():
    """The trap in this change. The echo penalty means "probably meta-noise
    from the conversation happening right now" — a claim about when content
    HAPPENED. `mnemostack index` stamps every point with the current time,
    so letting the write stamp arm that penalty would halve the score of an
    entire archive for the crime of being imported today.
    """
    just_written = _fresh({"indexed_at": _ago(0)})
    assert just_written == pytest.approx(1.0, abs=1e-6), just_written
    # ...while a memory that says it HAPPENED moments ago still gets it.
    just_happened = _fresh({"timestamp": _ago(0)})
    assert just_happened == pytest.approx(0.5, abs=1e-6), just_happened


def test_a_memory_with_no_time_information_at_all_is_unchanged():
    """Nothing to age by, so the historical neutral value stands."""
    assert _fresh({}) == 0.5
    assert _fresh({"indexed_at": "not-a-date"}) == 0.5
    assert _fresh({"indexed_at": None}) == 0.5


def test_a_write_stamp_in_a_foreign_numeric_format_does_not_crash():
    """`indexed_at` is written by this stack and is always ISO, but a
    payload is a payload: a hand-edited or migrated point can hold
    anything, and a recall must not 500 over it."""
    for bad in (12345, [1], {"a": 1}, "", "   "):
        assert (
            _fresh({"indexed_at": bad}) in (0.5, 1.0) or 0.0 <= _fresh({"indexed_at": bad}) <= 1.0
        )
