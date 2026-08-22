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


def test_a_non_string_write_stamp_is_ignored_rather_than_read_as_an_epoch():
    """The contract is STRICT ISO, and the strictness is the point.

    The shared instant parser also accepts Unix epochs, because the
    configured `timestamp` key may legitimately be numeric in a foreign
    collection. `indexed_at` has no such freedom — this stack writes it and
    always writes ISO — so a number in it is corruption. Reading it as an
    epoch is worse than ignoring it: `12345` becomes 1970 and buries the
    memory at the bottom of every recall, where declining to age it merely
    leaves it where it was.

    (The assertion this replaced allowed any value in [0, 1] — which is
    every value freshness can take, so it pinned nothing.)
    """
    for corrupt in (12345, 12345.0, [1], {"a": 1}, True, object()):
        assert _fresh({"indexed_at": corrupt}) == 0.5, corrupt

    # An unparseable STRING is likewise neutral — not an error, not 1970.
    # The numeric strings matter most: the shared instant parser accepts
    # those as Unix epochs, so a type check alone left "12345" reading as
    # 1970 while the docstring claimed strict ISO. A type is not a format.
    for text in ("", "   ", "not-a-date", "2026-13-45T99:99:99", "12345", "1719834000"):
        assert _fresh({"indexed_at": text}) == 0.5, text

    # ...and a real ISO stamp still ages, so strictness has not quietly
    # disabled the fallback this file exists to add.
    assert _fresh({"indexed_at": _ago(365)}) < 0.05


class _ConstantProvider:
    dimension = 3

    def embed(self, text):
        return [0.1, 0.2, 0.3]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    def health_check(self):
        return True, "ok"


class _RecordingStore:
    """Just enough VectorStore for `cmd_index` to run against."""

    def __init__(self):
        self.points: dict[str, dict] = {}
        self.set_payloads: list = []
        self.deletes: list = []

    def collection_exists(self):
        return True

    def ensure_collection(self, recreate=False):
        pass

    def count(self):
        return len(self.points)

    def iter_ids(self):
        return iter(self.points)

    def scroll(self, *a, **kw):
        from types import SimpleNamespace

        return iter(SimpleNamespace(id=cid, payload=dict(p)) for cid, p in self.points.items())

    def upsert(self, cid, vec, payload, **kw):
        self.points[cid] = dict(payload)

    def upsert_batch(self, points, **kw):
        for cid, vec, payload in points:
            self.upsert(cid, vec, payload)

    def set_payload(self, cid, payload, **kw):
        self.set_payloads.append((cid, payload))
        self.points[cid].update(payload)

    def delete_payload_keys(self, cid, keys, **kw):
        self.deletes.append((cid, keys))


# ------------------------------------------- the write path that feeds it


def test_the_cli_indexer_writes_the_stamp_the_fallback_reads(tmp_path, monkeypatch):
    """The fallback is only worth having if the core write path fills the
    field. `mnemostack index` built its payloads without `indexed_at` while
    every other write path set it — so documents indexed through the
    command most people use were exactly the ones that still could not age.

    Asserted on what lands in the STORE, and then on what the stage makes
    of it, rather than on the shape of the source: a stamp that is written
    but unreadable by the reader is the same outage as no stamp.
    """
    import mnemostack.cli as cli

    # Long enough to chunk SEVERAL times: with one chunk, "one stamp per
    # run" and "one stamp per chunk" are the same assertion and neither is
    # tested — a mutation moving the clock into the loop survived exactly
    # that way.
    (tmp_path / "doc.md").write_text(
        "\n\n".join(f"paragraph {i} " + "filler words " * 40 for i in range(12)),
        encoding="utf-8",
    )
    store = _RecordingStore()
    monkeypatch.setattr(cli, "get_provider", lambda *a, **kw: _ConstantProvider())
    monkeypatch.setattr(cli, "VectorStore", lambda **kw: store)
    assert cli.main(["index", str(tmp_path), "--chunk-size", "200"]) == 0

    payloads = list(store.points.values())
    assert len(payloads) > 3, f"expected several chunks, got {len(payloads)}"
    stamps = [p.get("indexed_at") for p in payloads]
    assert all(isinstance(v, str) and v for v in stamps), stamps
    # One instant for the whole run, not one per chunk: chunks written
    # together share a write time rather than differing by the wall-clock
    # cost of embedding.
    assert len(set(stamps)) == 1, sorted(set(stamps))

    parsed = datetime.fromisoformat(stamps[0])
    assert parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0), stamps[0]
    # ...and the stage actually ages by it: a just-written point is fresh,
    # which the flat 0.5 could never express.
    assert _fresh({"indexed_at": stamps[0]}) > 0.99


def test_a_payload_refresh_does_not_reset_the_write_time(tmp_path, monkeypatch):
    """A refresh rewrites payload FIELDS; it does not re-write the point.
    Restamping would reset the age of the whole corpus to today — undoing
    the ageing this field exists to provide — and would turn every warm
    refresh into a full payload write."""
    import mnemostack.cli as cli

    (tmp_path / "doc.md").write_text("stable content", encoding="utf-8")
    store = _RecordingStore()
    monkeypatch.setattr(cli, "get_provider", lambda *a, **kw: _ConstantProvider())
    monkeypatch.setattr(cli, "VectorStore", lambda **kw: store)
    argv = ["index", str(tmp_path), "--chunk-size", "2000", "--refresh-payloads"]

    assert cli.main(argv) == 0
    first = {cid: p["indexed_at"] for cid, p in store.points.items()}
    assert first

    # Age the stored stamp, then refresh: the old value must survive.
    aged = _ago(200)
    for payload in store.points.values():
        payload["indexed_at"] = aged
    assert cli.main(argv) == 0

    after = {cid: p["indexed_at"] for cid, p in store.points.items()}
    assert set(after.values()) == {aged}, (first, after)
    assert _fresh({"indexed_at": after[next(iter(after))]}) < 0.05


@pytest.mark.parametrize(
    "flags",
    [
        pytest.param([], id="prose"),
        pytest.param(["--code"], id="code"),
        pytest.param(["--window-size", "3"], id="sliding-window"),
    ],
)
def test_every_chunking_branch_gets_the_write_stamp(tmp_path, monkeypatch, flags):
    """`cmd_index` builds payloads in three places — prose chunks, `--code`
    chunks, and sliding windows — and the first attempt stamped only the
    first, leaving two whole modes unable to age.

    Parametrised over the branches rather than asserting one of them: the
    stamp is applied at the single point they all converge on, and this is
    what makes a fourth branch fail loudly instead of quietly shipping
    another unstampable mode.
    """
    import mnemostack.cli as cli

    # Both kinds of input, so each mode finds something: the prose and
    # window branches scan .md/.txt, `--code` scans sources.
    (tmp_path / "doc.md").write_text(
        "\n\n".join(f"paragraph {i} " + "filler words " * 40 for i in range(12)),
        encoding="utf-8",
    )
    (tmp_path / "doc.py").write_text(
        "\n\n".join(f"def f{i}():\n    return {i}  # " + "x " * 30 for i in range(12)),
        encoding="utf-8",
    )
    store = _RecordingStore()
    monkeypatch.setattr(cli, "get_provider", lambda *a, **kw: _ConstantProvider())
    monkeypatch.setattr(cli, "VectorStore", lambda **kw: store)
    assert cli.main(["index", str(tmp_path), "--chunk-size", "200", *flags]) == 0

    payloads = list(store.points.values())
    assert payloads, f"nothing indexed for {flags}"
    stamps = [p.get("indexed_at") for p in payloads]
    assert all(isinstance(v, str) and v for v in stamps), (flags, stamps)
    assert len(set(stamps)) == 1, (flags, sorted(set(stamps)))
    assert _fresh({"indexed_at": stamps[0]}) > 0.99, (flags, stamps[0])


@pytest.mark.parametrize(
    "stored",
    [
        pytest.param("", id="empty-string"),
        pytest.param("   ", id="blank"),
        pytest.param("not-a-date", id="unparseable"),
        pytest.param("12345", id="numeric-string"),
        pytest.param(12345, id="number"),
        pytest.param(None, id="null"),
    ],
)
def test_a_refresh_does_not_promote_a_corrupt_stamp_to_today(tmp_path, monkeypatch, stored):
    """The reader treats a corrupt stamp as the neutral 0.5. A refresh that
    replaced it with the run's own time would promote exactly those points
    to MAXIMALLY fresh — the opposite of what the reader decided about them,
    reached by a metadata-only operation that is supposed to change nothing
    about when the point was written.

    Carried by presence, not by validity: requiring a well-formed value
    looked like the safe check and was the one that caused this.
    """
    import mnemostack.cli as cli

    (tmp_path / "doc.md").write_text("stable content", encoding="utf-8")
    store = _RecordingStore()
    monkeypatch.setattr(cli, "get_provider", lambda *a, **kw: _ConstantProvider())
    monkeypatch.setattr(cli, "VectorStore", lambda **kw: store)
    argv = ["index", str(tmp_path), "--chunk-size", "2000", "--refresh-payloads"]

    assert cli.main(argv) == 0
    for payload in store.points.values():
        payload["indexed_at"] = stored
    assert cli.main(argv) == 0

    after = [p.get("indexed_at") for p in store.points.values()]
    assert after == [stored] * len(after), after
    assert _fresh({"indexed_at": after[0]}) == 0.5, after[0]


def test_a_refresh_does_not_invent_a_write_time_for_a_legacy_point(tmp_path, monkeypatch):
    """A point predating the field has no write time, and a refresh has no
    way to learn one — it rewrites payload FIELDS, it does not write the
    point. Stamping today would claim knowledge it does not have and make
    an old point look new."""
    import mnemostack.cli as cli

    (tmp_path / "doc.md").write_text("stable content", encoding="utf-8")
    store = _RecordingStore()
    monkeypatch.setattr(cli, "get_provider", lambda *a, **kw: _ConstantProvider())
    monkeypatch.setattr(cli, "VectorStore", lambda **kw: store)
    argv = ["index", str(tmp_path), "--chunk-size", "2000", "--refresh-payloads"]

    assert cli.main(argv) == 0
    for payload in store.points.values():
        payload.pop("indexed_at", None)  # a point from before the field existed
    assert cli.main(argv) == 0

    for payload in store.points.values():
        assert "indexed_at" not in payload, payload
    assert _fresh({}) == 0.5


def test_the_docstrings_do_not_deny_the_fallback_they_now_have():
    """These two passages document the ranking model, and both were written
    to say `freshness` has NO `indexed_at` fallback — which this change made
    false. A wrong explanation of a ranking rule outlives the reviewer who
    would have caught it."""
    import inspect

    from mnemostack import access
    from mnemostack.recall.pipeline.stages import compute_access_boost

    for text in (compute_access_boost.__doc__ or "", inspect.getdoc(access) or ""):
        lowered = text.lower()
        assert "no ``indexed_at`` fallback" not in lowered
        assert "does not fall back to `indexed_at`" not in lowered
        assert "flat 0.5 whatever its age" not in lowered


# ------------------------------------- every path that writes a point


def test_every_write_path_stamps_the_field_this_fallback_reads():
    """Three separate paths write points, and the stamp was added to them
    one review round at a time: the library `Ingestor`, `mnemostack index`,
    and `index-markdown` (which bypasses `Ingestor` entirely). Each miss
    looked like a small omission and was a whole indexing mode frozen at a
    flat middling freshness however old its documents were.

    Enumerated here so a FOURTH writer fails loudly instead of shipping as
    another unstampable mode — that is the only part of this that a future
    change can inherit.
    """
    import inspect

    from mnemostack import cli, ingest
    from mnemostack.markdown import indexer
    from mnemostack.vector.patch import WRITE_TIME_KEY

    writers = {
        "library Ingestor": inspect.getsource(ingest),
        "mnemostack index": inspect.getsource(cli.cmd_index),
        "index-markdown": inspect.getsource(indexer.collect_markdown),
    }
    for name, source in writers.items():
        assert WRITE_TIME_KEY in source or "WRITE_TIME_KEY" in source, name

    # ...and both merge paths carry the stored value rather than restamping.
    from mnemostack.markdown import sync

    for name, source in (
        ("mnemostack index", inspect.getsource(cli.cmd_index)),
        ("index-markdown", inspect.getsource(sync)),
    ):
        assert "carry_write_time(" in source, name


def test_the_carry_rule_is_one_function_not_a_copy_per_path():
    """Both merge paths call the same helper. Two copies of a rule this
    subtle — carried by presence, never invented — is two rules, and the
    copy that drifts is the one nobody re-reads."""
    from mnemostack.vector.patch import WRITE_TIME_KEY, carry_write_time

    # present, however malformed → carried unchanged
    for stored in ("", "   ", "12345", 12345, None, "not-a-date"):
        assert carry_write_time({WRITE_TIME_KEY: stored}, {WRITE_TIME_KEY: "NEW"}) == {
            WRITE_TIME_KEY: stored
        }, stored
    # absent → never invented
    assert carry_write_time({}, {WRITE_TIME_KEY: "NEW"}) == {}
    # unrelated fields are left alone
    assert carry_write_time({"a": 1}, {"b": 2, WRITE_TIME_KEY: "NEW"}) == {"b": 2}


def test_the_markdown_indexer_stamps_what_it_collects(tmp_path):
    """The behavioural half of the enumeration above, for the path that was
    missing it: assert on the payloads `collect_markdown` produces, and on
    what the freshness stage makes of them."""
    from mnemostack.markdown import collect_markdown
    from mnemostack.vector.patch import WRITE_TIME_KEY

    (tmp_path / "a.md").write_text("# Title\n\n" + "body words " * 200, encoding="utf-8")
    (tmp_path / "b.md").write_text("# Other\n\n" + "more words " * 200, encoding="utf-8")

    collection = collect_markdown(tmp_path, chunk_size=200)
    payloads = [chunk.payload for chunk in collection.chunks]
    assert len(payloads) > 3, len(payloads)

    stamps = [p.get(WRITE_TIME_KEY) for p in payloads]
    assert all(isinstance(v, str) and v for v in stamps), stamps
    # One instant for the whole walk, not one per chunk or per file.
    assert len(set(stamps)) == 1, sorted(set(stamps))

    parsed = datetime.fromisoformat(stamps[0])
    assert parsed.tzinfo is not None and parsed.utcoffset() == timedelta(0), stamps[0]
    # ...and it is readable by the stage that needs it.
    assert _fresh({"indexed_at": stamps[0]}) > 0.99
