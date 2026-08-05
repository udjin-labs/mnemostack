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

import json
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


def _normalize(value: Any) -> Any:
    """Recursively coerce to the shape a JSON backend round-trip produces.

    Mapping keys become strings (mixed int/str keys are VALID YAML and must
    not crash the sort), tuples become lists.
    """
    if isinstance(value, Mapping):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value


def _canonical(value: Any) -> str:
    """Normalized comparison form for JSON-compatible payload values.

    Stored payloads come back from the backend as JSON types (tuples turn
    into lists, mapping order is arbitrary, mapping keys are strings) —
    equality must reflect the EFFECTIVE value, never object identity,
    container flavor or key order. A value that cannot be canonicalized at
    all compares UNEQUAL to everything: the safe direction is a rewrite,
    never a skipped real change.
    """
    try:
        return json.dumps(_normalize(value), sort_keys=True, default=str)
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
