"""Per-call recall trace — observability on top of fail-open recall.

Recall deliberately degrades instead of failing: a broken retriever
contributes nothing, a broken reranker leaves the original order. This
module makes those degradations visible without changing the behavior.

Usage:

    trace = RecallTrace()
    results = recaller.recall(query, trace=trace)
    results = apply_rerank_safe(reranker, query, results, trace)
    trace.degraded      # e.g. ["retriever:bm25:failed", "reranker:fallback"]
    trace.notes         # routine signals, e.g. ["temporal:no_parse"]
    trace.to_dict()     # JSON-friendly dump for debug responses

A trace object is per-call: never share one between concurrent requests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..observability import counter

if TYPE_CHECKING:
    from .recaller import RecallResult
    from .reranker import Reranker

logger = logging.getLogger(__name__)

#: Degradation counter mirrored from per-call traces so process-wide surfaces
#: (`/status`, `/metrics`) can see degradations that live only on the trace —
#: reranker unavailable/fallback and retriever failures have no other counter.
DEGRADED_COUNTER = "mnemostack.recall.degraded"

#: Trace tags that are routine signals, not service degradations, and so must
#: NOT count toward the operator's degraded-events total. `temporal:no_parse`
#: fires on any non-temporal query (a parallel vector retriever still answers).
_NON_DEGRADED_TAGS = frozenset({"temporal:no_parse"})

#: Same classification for tags with a DYNAMIC prefix — each multi-field
#: lexical arm reports its own gate verdict ("qdrant_text:title:no_tokens"),
#: so an exact set can't name them; the shape is what's routine.
_NON_DEGRADED_SUFFIXES = (":no_tokens",)


def _is_routine(tag: str) -> bool:
    """A stage-did-not-apply signal, not a fault."""
    return tag in _NON_DEGRADED_TAGS or tag.endswith(_NON_DEGRADED_SUFFIXES)


@dataclass
class RetrieverTrace:
    """One retriever's contribution to a recall call, pre-fusion."""

    name: str
    ranked: list[tuple[str, float]] = field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0.0
    query: str | None = None  # set when query expansion produced variants

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "ranked": [[rid, round(score, 6)] for rid, score in self.ranked],
            "latency_ms": round(self.latency_ms, 2),
        }
        if self.error:
            d["error"] = self.error
        if self.query is not None:
            d["query"] = self.query
        return d


@dataclass
class RecallTrace:
    """Trace of one recall call: per-retriever inputs, fused output, degradations.

    `fused` is the order recall returned (post-fusion, post-vector-floor);
    `post_rerank` is the reranker's order when a reranker ran. The final
    response list may still differ if vector-floor re-appends items after
    rerank. Tags are stable strings. `notes` is the AUTHORITATIVE list of
    routine stage-did-not-apply signals ("temporal:no_parse" — any query
    without a parseable date; "<arm>:no_tokens" — a lexical arm whose gate
    found no usable tokens). `degraded` carries components that actually
    fell back ("retriever:<name>:failed", "reranker:fallback",
    "reranker:unavailable") — plus, DEPRECATED until the next major, a
    back-compat duplicate of the routine tags, because its stable-strings
    contract predates the split and existing matchers must survive a minor
    upgrade. New consumers: read `notes` for routine signals; a `degraded`
    entry absent from `notes` is a real fault.
    """

    retrievers: list[RetrieverTrace] = field(default_factory=list)
    fused: list[tuple[str, float]] = field(default_factory=list)
    post_rerank: list[tuple[str, float]] | None = None
    degraded: list[str] = field(default_factory=list)
    #: Routine signals — a stage that did not apply, not a fault. Same stable
    #: strings as `degraded`; classified by `_NON_DEGRADED_TAGS`.
    notes: list[str] = field(default_factory=list)

    def restrict_to_ids(self, allowed: Any) -> None:
        """Drop every trace entry whose id is outside ``allowed`` (tenant scrub).

        A tenant-scoped, ``include_trace`` recall must not expose another tenant's
        ids/scores through the trace even if a retriever's tenant filter had a bug —
        so keep only ids that survived the tenant backstop. Ids are compared as
        strings (result ids may be int, trace ids are str).
        """
        allow = {str(a) for a in allowed}
        for rt in self.retrievers:
            rt.ranked = [(rid, s) for rid, s in rt.ranked if str(rid) in allow]
        self.fused = [(rid, s) for rid, s in self.fused if str(rid) in allow]
        if self.post_rerank is not None:
            self.post_rerank = [(rid, s) for rid, s in self.post_rerank if str(rid) in allow]

    def mark(self, tag: str) -> None:
        # One classification: `notes` is AUTHORITATIVE for routine signals
        # and they never reach the process-wide counter. DEPRECATED
        # back-compat: the same tags are still duplicated into `degraded` —
        # its stable-strings contract predates the split, and clients or
        # alerts matching e.g. `temporal:no_parse` there must not change
        # behavior on a MINOR upgrade. The duplication is removed at the
        # next MAJOR; new consumers should read `notes`.
        if _is_routine(tag):
            if tag not in self.notes:
                self.notes.append(tag)
            if tag not in self.degraded:
                self.degraded.append(tag)
            return
        if tag not in self.degraded:
            self.degraded.append(tag)
            # Mirror the (deduped-per-call) degradation into a process-wide
            # counter so /status and /metrics see trace-only degradations.
            # No-op under the default NullRecorder.
            counter(DEGRADED_COUNTER, 1, labels={"reason": tag})

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "retrievers": [rt.to_dict() for rt in self.retrievers],
            "fused": [[rid, round(score, 6)] for rid, score in self.fused],
            "degraded": list(self.degraded),
            "notes": list(self.notes),
        }
        if self.post_rerank is not None:
            d["post_rerank"] = [[rid, round(score, 6)] for rid, score in self.post_rerank]
        return d


def apply_rerank_safe(
    reranker: Reranker | None,
    query: str,
    results: list[RecallResult],
    trace: RecallTrace | None = None,
) -> list[RecallResult]:
    """Rerank with the fail-open contract, but leave a trace of the fallback."""
    if reranker is None:
        if trace is not None:
            trace.mark("reranker:unavailable")
        return results
    try:
        out = reranker.rerank(query, results)
    except Exception as exc:  # noqa: BLE001 — fail-open by design
        logger.warning("reranker failed (%s) — returning pre-rerank order", exc)
        if trace is not None:
            trace.mark("reranker:fallback")
        return results
    # Stateless fallback detection by identity — correct under concurrency (no
    # per-instance state). Only for rerankers that advertise the contract
    # (`fallback_keeps_input_object`): they return the exact input list object
    # on a kept-order fallback and a new list on success. Rerankers that sort
    # in place and return the input don't set the marker, so a successful
    # reorder is never misread as a fallback.
    kept_input = out is results and getattr(reranker, "fallback_keeps_input_object", False)
    if kept_input:
        if trace is not None:
            trace.mark("reranker:fallback")
        return out
    if trace is not None:
        trace.post_rerank = [(str(r.id), r.score) for r in out]
    return out
