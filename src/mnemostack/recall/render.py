"""Rendering helpers for RecallResult lists.

Provides compact and full text renderings — useful for 3-layer search UX
(first get a cheap index, then fetch full details for selected ids only).
"""
from __future__ import annotations

from collections.abc import Iterable

from .recaller import RecallResult

__all__ = ["compact_format", "full_format"]


def _snippet(text: str, limit: int) -> str:
    if not text:
        return ""
    flat = text.replace("\n", " ").strip()
    if len(flat) > limit:
        return flat[:limit] + "…"
    return flat


def _source_label(r: RecallResult) -> str:
    src = ",".join(r.sources) if r.sources else "?"
    source_path = r.payload.get("source") or r.payload.get("source_file") or ""
    return f"({src})  {source_path}"


def compact_format(
    results: Iterable[RecallResult],
    *,
    snippet_len: int = 60,
    include_hint: bool = True,
    start_index: int = 1,
) -> str:
    """Render results as a compact index (~50-100 tokens per result).

    Layout per row:
        [N] [id:<stable_id>] score=0.1234 (vector,bm25)  <source>
            <snippet…>

    The `id` field is the stable citable identifier of the RecallResult
    (content-derived UUID for most sources). Callers can use it to fetch
    full text later (e.g. via storage-specific helpers).
    """
    lines: list[str] = []
    any_result = False
    for i, r in enumerate(results, start=start_index):
        any_result = True
        lines.append(f"[{i}] [id:{r.id}] score={r.score:.4f} {_source_label(r)}")
        lines.append(f"    {_snippet(r.text, snippet_len)}")
    if not any_result:
        return "(no results)"
    if include_hint:
        lines.append("")
        lines.append("  hint: results carry stable ids; resolve full text via your storage's fetch-by-id helper")
    return "\n".join(lines)


def full_format(
    results: Iterable[RecallResult],
    *,
    snippet_len: int = 200,
    start_index: int = 1,
) -> str:
    """Render results with a wider preview (default ~200 chars).

    Equivalent to compact_format with a larger snippet_len and no hint.
    """
    return compact_format(
        results,
        snippet_len=snippet_len,
        include_hint=False,
        start_index=start_index,
    )
