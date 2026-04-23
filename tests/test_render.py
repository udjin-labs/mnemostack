"""Tests for mnemostack.recall.render — compact/full rendering helpers."""
from __future__ import annotations

from mnemostack.recall import RecallResult, compact_format, full_format


def _make(id_: str, text: str, score: float, sources=None, source_path="") -> RecallResult:
    return RecallResult(
        id=id_,
        text=text,
        score=score,
        sources=list(sources or []),
        payload={"source": source_path} if source_path else {},
    )


def test_compact_format_basic():
    r = _make("abc-123", "Hello world this is a snippet that goes on and on", 0.42,
              sources=["vector"], source_path="memory/test.md")
    out = compact_format([r], snippet_len=20, include_hint=False)
    assert "[id:abc-123]" in out
    assert "score=0.4200" in out
    assert "(vector)" in out
    assert "memory/test.md" in out
    # snippet truncated
    assert "Hello world this is " + "\u2026" in out


def test_compact_format_sources_multiple():
    r = _make("x", "t", 0.1, sources=["bm25", "vector"], source_path="s.md")
    out = compact_format([r], include_hint=False)
    assert "(bm25,vector)" in out


def test_compact_format_empty():
    assert compact_format([]) == "(no results)"


def test_compact_format_hint_present_by_default():
    r = _make("x", "t", 0.1, sources=["vector"], source_path="s.md")
    out = compact_format([r])
    assert "hint" in out.lower()


def test_compact_format_newlines_flattened():
    r = _make("x", "line one\nline two\nline three", 0.1, sources=["vector"])
    out = compact_format([r], snippet_len=200, include_hint=False)
    assert "line one line two line three" in out
    # no raw newlines in the snippet line (they are replaced with spaces)
    snippet_lines = [ln for ln in out.split("\n") if ln.strip().startswith("line one")]
    assert len(snippet_lines) == 1


def test_compact_format_start_index():
    r = _make("x", "t", 0.1, sources=["vector"])
    out = compact_format([r], start_index=5, include_hint=False)
    assert out.startswith("[5]")


def test_compact_format_source_file_fallback():
    r = RecallResult(id="y", text="t", score=0.2, sources=["vector"],
                     payload={"source_file": "transcripts/foo.jsonl"})
    out = compact_format([r], include_hint=False)
    assert "transcripts/foo.jsonl" in out


def test_full_format_wider_snippet():
    long_text = "a" * 300
    r = _make("z", long_text, 0.5, sources=["vector"])
    out = full_format([r])
    # default full snippet_len is 200, so we expect 200 chars + ellipsis
    assert "a" * 200 + "\u2026" in out
    # full_format does not include hint
    assert "hint" not in out.lower()
