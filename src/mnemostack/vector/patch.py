"""Payload patching primitives — owned-field diffing for refresh paths.

A warm re-index visits every already-indexed point to refresh its payload.
Writing every point unconditionally turns an unchanged run into O(points)
mutation requests (plus the backend's WAL/replication/optimizer work for no
semantic change). The refresh callers already hold both the old payload
snapshot and the new payload, so they can compute the MINIMAL patch — and
skip the point entirely when nothing effectively changed.

The diff is deliberately scoped to fields the caller OWNS: every key of the
new payload (that is exactly what the historical merge-write would set) plus
the caller-supplied stale keys (formerly-owned fields the new payload no
longer produces). Foreign keys — enrichment written by other pipelines,
validity markers, anything this indexer never produced — are outside the
comparison and can never be set or deleted by a patch.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["PayloadPatch", "apply_patches_via", "carry_snapshot_capture_time", "diff_payload"]


@dataclass(frozen=True)
class PayloadPatch:
    """A write-or-skip decision: the full owned payload to (re)write."""

    id: str | int
    set_values: Mapping[str, Any] = field(default_factory=dict)
    delete_keys: tuple[str, ...] = ()


def _instant(value: _dt.date | _dt.datetime | _dt.time) -> str:
    """One canonical rendering per instant, matching the backend's ISO form.

    A YAML frontmatter timestamp arrives as a ``datetime``/``date`` OBJECT,
    while the stored payload comes back as the ISO STRING the backend
    serialized it to (``T`` separator; UTC may render as ``Z`` or
    ``+00:00``; offsets are kept) — ``str(datetime)`` uses a SPACE
    separator, so comparing through ``str()`` marked every timestamped
    point changed on every warm run. Aware datetimes are reduced to UTC so
    every spelling of the same instant compares equal.
    """
    if isinstance(value, _dt.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(_dt.timezone.utc)
        return value.isoformat()
    return value.isoformat()


#: Calendar-date timestamps only, in exactly the shapes the backend (and
#: ``datetime.isoformat``) emits: T or space separator, optional seconds,
#: optional 3- or 6-digit fraction, optional Z / ±HH:MM offset. Every match
#: parses identically on all supported interpreters — delegating the guard
#: to ``fromisoformat``'s full acceptance surface would make normalization
#: Python-version-dependent (3.11 grew week dates, ordinal dates and
#: arbitrary fraction lengths), i.e. two runs of the same corpus could
#: disagree on what "unchanged" means.
_TEMPORAL_STRING = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{3}|\.\d{6})?)?"
    r"(?:Z|[+-]\d{2}:\d{2})?"
)


def _normalize_temporal_string(value: str) -> str:
    """Map an ISO datetime STRING onto the same canonical instant form.

    The stored side of a comparison is always a string; only strings
    matching the exact backend-emitted shapes are rewritten, and a
    date-only string stays as-is — mirroring the object branch, where a
    ``date`` never grows a time part. Both sides pass through here, so a
    plain string field that merely looks like a timestamp still compares
    consistently against another string.
    """
    if not _TEMPORAL_STRING.fullmatch(value):
        return value
    try:
        parsed = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:  # right shape, impossible date (month 13, hour 25, …)
        return value
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(_dt.timezone.utc)
    return parsed.isoformat()


def _normalize(value: Any) -> Any:
    """Recursively coerce to the shape a JSON backend round-trip produces.

    Mapping keys become strings (mixed int/str keys are VALID YAML and must
    not crash the sort), tuples become lists, and timestamps — objects and
    ISO strings alike — collapse onto one canonical instant form.
    """
    if isinstance(value, Mapping):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return _instant(value)
    if isinstance(value, str):
        return _normalize_temporal_string(value)
    return value


def _canonical(value: Any) -> str:
    """Normalized comparison form for JSON-compatible payload values.

    Stored payloads come back from the backend as JSON types (tuples turn
    into lists, mapping order is arbitrary, mapping keys are strings) —
    equality must reflect the EFFECTIVE value, never object identity,
    container flavor or key order. A value that cannot be canonicalized at
    all compares UNEQUAL to everything (no ``default=`` hook: a lossy
    string fallback would let a ``Decimal("1.0")`` compare EQUAL to the
    string ``"1.0"`` and skip a real change): the safe direction is a
    rewrite, never a skipped one.
    """
    try:
        return json.dumps(_normalize(value), sort_keys=True)
    except (TypeError, ValueError):
        return f"__uncanonical__:{id(value)}"


def diff_payload(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    *,
    point_id: str | int,
    stale_keys: Iterable[str] = (),
) -> PayloadPatch | None:
    """The write-or-skip decision for a point's owned payload, or None.

    The diff decides WHETHER to write, not what: a returned patch carries
    the FULL new payload (plus the caller's stale keys to delete), exactly
    the historical delete-stale + merge-write pair — so any single writer
    leaves the point coherent under the owned-fields last-writer-wins
    contract, even when two refreshes with different snapshots overlap
    (a minimal per-key patch would interleave them into a payload matching
    neither snapshot). None means every owned field already has its
    effective value and the point must not be touched at all.

    ``stale_keys`` come from the caller's ownership record
    (``_md_keys``/``_enrich_keys``); values compare by normalized JSON form.
    """
    changed = any(
        key not in old or _canonical(old[key]) != _canonical(value)
        for key, value in new.items()
    )
    delete = tuple(stale_keys)
    if not changed and not delete:
        return None
    return PayloadPatch(
        id=point_id,
        set_values=dict(new),
        delete_keys=delete,
    )


def carry_snapshot_capture_time(
    old: Mapping[str, Any],
    new: dict[str, Any],
    *,
    hash_key: str = "source_content_hash",
    captured_key: str = "source_captured_at",
) -> dict[str, Any]:
    """Keep the stored capture time when the content snapshot is unchanged.

    ``source_captured_at`` records when the CURRENT content was captured —
    every run stamps a fresh timestamp, so without this parity rule an
    otherwise-unchanged warm re-index would see every point as "changed"
    and the zero-mutation guarantee would be fiction on any real corpus.
    When the content hash is identical, the stored capture time is still
    the truth (the content has not been re-captured as something new), so
    it is carried into the new payload — both for the comparison and for
    any write that happens for other reasons.
    """
    if (
        hash_key in old
        and old.get(hash_key) == new.get(hash_key)
        and captured_key in old
    ):
        new[captured_key] = old[captured_key]
    return new


def apply_patches_via(
    store: Any,
    patches: list[PayloadPatch],
    *,
    tenant: str | None = None,
    batch_size: int = 100,
) -> int:
    """Apply patches through the store's batch hook when it has one.

    Stores without ``apply_payload_patches`` (custom/legacy implementations)
    get the scalar delete-then-set pair per patch — byte-for-byte the
    historical refresh behavior, including the tenant kwarg being passed
    only when scoped so tenant-unaware stores keep working.
    """
    if not patches:
        return 0
    method = getattr(store, "apply_payload_patches", None)
    if method is not None:
        return int(method(patches, tenant=tenant, batch_size=batch_size))
    tkw: dict[str, Any] = {"tenant": tenant} if tenant is not None else {}
    for patch in patches:
        if patch.delete_keys:
            store.delete_payload_keys(patch.id, list(patch.delete_keys), **tkw)
        if patch.set_values:
            store.set_payload(patch.id, dict(patch.set_values), **tkw)
    return len(patches)
