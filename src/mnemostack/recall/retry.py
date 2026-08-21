"""Reformulate-and-retry for a recall that came back nearly empty.

A question phrased one way can miss memories phrased another — "what did
we decide about auth" against a note that says "we're going with OAuth
after all". The stack has always had the paraphrasing machinery
(:class:`~mnemostack.recall.expansion.QueryExpander`) and the answer path
already retries its own sub-recalls, but `/recall` itself had no policy
for "this returned nothing, try saying it differently".

Opt-in, and deliberately so: a second pass costs an LLM call plus another
round of retrieval, and that is a bill the operator has to agree to. It
is off by default and per-tenant metering makes it visible when on.

**What counts as weak is a COUNT, not a score.** Fused scores are RRF
values — `1/(k+rank)` — so they encode position in a list, not
confidence: a query whose only hit is irrelevant scores exactly as high
as one whose only hit is perfect. Thresholding them would look like a
quality signal while measuring nothing of the sort. The number of results
is a real signal ("we found almost nothing"), so that is the trigger, and
the default is the least ambiguous case of all: zero.

Bounded on purpose:

- ONE extra round, never a ladder. A retry that fails is an answer.
- at most :data:`MAX_VARIANTS` paraphrases, so the extra retrieval cost
  is fixed rather than proportional to how badly the query did.
- every recall in the retry carries the caller's tenant, filters,
  validity view and budget — the retry is the SAME call asked again in
  other words, not a wider one. (A retry that quietly widened scope would
  be a tenant leak wearing a feature's clothes.)
"""

from __future__ import annotations

from typing import Any

from ..observability.recorder import counter
from .flow import recall_flow

#: Paraphrases per retry. Two is enough to change the wording materially;
#: more multiplies retrieval cost for a query that has already failed.
MAX_VARIANTS = 2

#: Default weakness threshold: retry only a recall that returned NOTHING.
DEFAULT_WEAK_BELOW = 1


def is_weak(results: list[Any], below: int = DEFAULT_WEAK_BELOW) -> bool:
    """Whether this recall returned too little to leave alone."""
    return len(results) < max(1, below)


def has_room(
    results: list[Any],
    limit: int,
    token_budget: int | None = None,
    token_counter: Any = None,
) -> bool:
    """Whether the caller's page still has space a retry could fill.

    ONE question asked in both of the response's dimensions, because the
    response is cut to both: a page already holding `limit` results, or
    already spending the whole token budget, is full, and this feature
    exists to FIND memories a phrasing missed — not to spend an LLM call
    and a second retrieval reshuffling a page that is already as long as
    the caller asked for. Asking in one dimension and forgetting the other
    is the bug this consolidates (it was missed in the item dimension
    first, then in the budget dimension), so both live here, in the single
    gate the retry consults.
    """
    if len(results) >= limit:
        return False
    if token_budget:
        from .tokens import sum_tokens

        if sum_tokens(results, token_counter) >= token_budget:
            return False
    return True


def retry_weak_recall(
    recaller: Any,
    query: str,
    limit: int,
    *,
    llm: Any,
    results: list[Any],
    below: int = DEFAULT_WEAK_BELOW,
    **flow_kwargs: Any,
) -> tuple[list[Any], bool]:
    """(results, retried) — the original results, or better ones.

    Returns the input untouched when the recall was not weak, when no LLM
    is available to paraphrase with, or when no paraphrase came back with
    anything at all. A paraphrase that returns only memories the caller
    already had is NOT nothing: finding the same memory twice is evidence
    about its rank, and the fusion is where that evidence is spent.
    Never raises: a failed retry is a recall that did not improve, not a
    failed request.
    """
    budget = flow_kwargs.get("token_budget")
    counter_fn = flow_kwargs.get("token_counter")
    if not is_weak(results, below) or not has_room(results, limit, budget, counter_fn):
        return results, False
    if llm is None:
        counter("mnemostack.recall.weak_retry_unavailable", 1)
        return results, False

    counter("mnemostack.recall.weak_retry", 1)
    try:
        from .expansion import QueryExpander

        expander = QueryExpander(recaller, llm, n_variants=MAX_VARIANTS)
        variants = expander.generate_variants(query)[:MAX_VARIANTS]
    except Exception:  # noqa: BLE001 — a retry must not fail the recall
        counter("mnemostack.recall.weak_retry_failed", 1)
        return results, False
    if not variants:
        return results, False

    # Each variant recalls into its OWN trace. Sharing the caller's would
    # let every pass overwrite `fused`/`post_rerank`, so a client asking
    # for a trace would get one describing the LAST paraphrase while the
    # response carried the merged results — and under a tenant scope the
    # per-pass `restrict_to_ids` would additionally drop entries earlier
    # passes had recorded.
    caller_trace = flow_kwargs.pop("trace", None)
    # One ranked list per pass, fused at the end. Appending each pass's
    # hits to the previous ones would make the response arrival-ordered:
    # a worse hit from the first paraphrase would hold a slot ahead of a
    # better hit from the second, the budget trim would cut "the ranked
    # prefix" of a list that was never ranked, and the trace would report
    # that order as the one recall returned. RRF is what the stack
    # already uses to combine rankings of the SAME items from different
    # queries, and it is what makes a second phrasing able to win.
    #
    # ONE notion of identity, `str(id)`, decides whether two hits are the
    # same memory, because a vector store may hand back `1` where another
    # pass got `"1"` (Qdrant point ids are `str | int`). Two notions
    # disagreeing is not a cosmetic difference: RRF dedupes on whatever
    # object it is given, so the same memory would occupy two slots of the
    # caller's page. `_canonical` therefore returns the object already
    # standing for that id, and every list handed to the fusion is built
    # from those, so the fusion's own key cannot split what this dict
    # joined.
    # The caller's own objects as they arrived. The merge mutates them in
    # place — score, and now the metadata a later pass contributed — so any
    # path that decides to hand them back UNCHANGED has to hand back what
    # was on them, or "we kept your results" quietly means "we kept your
    # results with someone else's numbers and arms on them".
    original_state = [(r, r.score, list(r.sources), dict(r.payload or {})) for r in results]
    by_id: dict[str, Any] = {}
    ranked: list[list[tuple[Any, float]]] = [_one_pass(by_id, results, limit)]
    #: What the caller already had, counted the way the merge counts —
    #: `len(results)` would double-count a memory the original pass listed
    #: under two id types, and the recovery counter is supposed to report
    #: memories gained, not rows.
    original_count = len(by_id)
    #: Whether any paraphrase came back with hits at all. This, and NOT
    #: "did a paraphrase find an id we lacked", is what decides whether
    #: there is anything to fuse: a memory that TWO phrasings found should
    #: outrank one only the original found, and that is precisely what RRF
    #: expresses. Gating on new ids instead would silently skip the fusion
    #: this feature advertises whenever the paraphrases merely corroborate
    #: — reachable as soon as `--retry-weak-below` is above 1, where the
    #: original list is not empty.
    retrieved = False
    for variant in variants:
        variant_trace = _fresh_trace(caller_trace)
        if variant_trace is not None:
            flow_kwargs["trace"] = variant_trace
        try:
            # The SAME call in other words: every scoping keyword the
            # caller gave is forwarded unchanged. `QueryExpander.recall`
            # is deliberately not used here — it takes no tenant, so on an
            # authenticated deployment it would read across the boundary
            # (tracked as udjin-labs/mnemostack#166).
            extra = recall_flow(recaller, variant, limit, **flow_kwargs)
        except Exception:  # noqa: BLE001 — same rule, per variant
            counter("mnemostack.recall.weak_retry_failed", 1)
            # Fold in what this pass managed to record BEFORE it raised.
            # Arms that answered, latencies, and any degradation tag it
            # marked are exactly the diagnostics an operator needs to see
            # WHY the retry failed, and skipping this left the request
            # returning success with a trace and a `degraded` field that
            # mentioned nothing — a process-wide counter as the only
            # evidence a whole pass had collapsed.
            _absorb(caller_trace, variant_trace, variant)
            continue
        if extra:
            retrieved = True
        ranked.append(_one_pass(by_id, extra, limit))
        _absorb(caller_trace, variant_trace, variant)
    flow_kwargs.pop("trace", None)
    if caller_trace is not None:
        flow_kwargs["trace"] = caller_trace

    if not retrieved:
        return results, True  # asked again, still nothing: an answer too

    from .fusion import reciprocal_rank_fusion

    merged = []
    for item, fused_score in reciprocal_rank_fusion(ranked, limit=limit):
        # Carry the FUSED score, the way the query-expansion path does.
        # Leaving each object's pre-fusion score on it would publish
        # numbers that do not describe the order they are printed in —
        # not even descending — so a client that sorts by score would
        # undo the ranking this function just computed.
        item.score = fused_score
        merged.append(item)

    # The floor, re-applied to the merge, in the order `recall_flow` uses:
    # cut to `limit`, THEN guarantee the vector candidates, THEN the budget.
    # `vector_floor` is a promise that a configured number of raw vector
    # hits reach the caller even when the ranking stages would not have
    # kept them, and each pass keeps that promise by returning MORE than
    # `limit` — extras appended past the cut. Fusing those lists back down
    # to `limit` silently revoked the guarantee, so a recall that was
    # retried honoured a weaker contract than the same recall left alone,
    # for a floor the operator configured elsewhere and for a reason this
    # feature knows nothing about.
    apply_floor = getattr(recaller, "apply_vector_floor_after_rerank", None)
    if apply_floor is not None:
        # Every result of a pass carries that pass's candidate set in its
        # payload, so the merged list is a pool covering each pass that
        # contributed a survivor; the original pass, when it returned
        # anything, is first and its candidates are the ones that win.
        merged = apply_floor(merged, results + merged)

    if budget:
        # Re-applied to the MERGED list: each variant's own flow capped
        # its own results, and concatenating two lists that each fit the
        # budget produces one that does not. The budget is documented as a
        # hard cap on the response, so it has to hold after the merge, not
        # before it.
        from .tokens import apply_token_budget

        merged, _tokens = apply_token_budget(merged, budget, counter_fn)
    if len(merged) < original_count:
        # A RETRY NEVER RETURNS LESS THAN IT WAS GIVEN. The budget is a
        # hard cap on the response and the fusion reorders by rank, so the
        # two together can leave the caller holding FEWER memories than
        # they arrived with: a corroborated newcomer takes the front, and
        # the greedy trim — which stops at the first item that would
        # overflow — then evicts the smaller memories that fitted
        # perfectly well in the original order. The caller asked a
        # question and got two answers; a feature whose entire premise is
        # "that was too little" must not hand back one. So when the merge
        # cannot carry at least what the caller arrived with, the retry is
        # a no-op: original list, original scores.
        #
        # This is a rule about the SIZE of the response, and deliberately
        # not about its membership. At a fixed `limit` a better-ranked
        # memory does take a weaker one's slot, and it takes less than
        # corroboration to do it: RRF scores a rank-1 hit the same
        # wherever it was found, so ONE paraphrase's single hit already
        # ties the original's best and outranks whatever the original pass
        # put last. That is the entire point of fusing the rounds — and
        # demanding that every original survive would be no kinder, since
        # it protects the caller's weakest hit purely for being incumbent
        # and then drops one of the newcomers on an arbitrary tie-break
        # instead. What the caller is protected from is ending up with less
        # than they had, not from a re-ranking they opted into.
        for result, score, sources, payload in original_state:
            result.score = score
            result.sources = sources
            result.payload = payload
        return results, True
    gained = len(merged) - original_count
    if gained <= 0:
        final = results if not merged else merged
        _retrace(caller_trace, final)
        return final, True
    counter("mnemostack.recall.weak_retry_recovered", gained)
    _retrace(caller_trace, merged)
    return merged, True


def _one_pass(by_id: dict[str, Any], hits: list[Any], limit: int) -> list[tuple[Any, float]]:
    """One pass's RANKING: canonical objects, one entry per memory.

    RRF adds `1/(k+rank)` once per list an item appears in, so a pass that
    listed the same memory twice — `1` here and `"1"` there — would hand it
    two contributions and let a single pass out-vote genuine corroboration
    from two. Canonicalising the objects alone does not prevent that: the
    fusion sees the SAME object twice and scores it twice. One pass, one
    vote, so the duplicate is dropped here where the ranking is built.

    Only the first `limit` are a ranking at all. `recall_flow` returns the
    ranked page AND, past it, the vector floor's guaranteed candidates —
    items placed there precisely BECAUSE the ranking did not choose them.
    Voting them would invert the floor: a candidate appended to two passes
    collects two contributions, out-votes each pass's actual rank-one
    winner, and at `limit=1` becomes the sole survivor, after which the
    final floor step has nothing left to add. Every hit is still
    canonicalised, so the caller's metadata, the "found something new"
    test and the recovery count all see the full pass — it is only the
    VOTE that is limited to what the pass actually ranked.

    Identity is `str(id)` throughout, and a collision needs two distinct
    memories whose ids are `1` and `"1"`. Qdrant does not permit it: a
    string point id must be a UUID (`Point id 1 is not a valid UUID`),
    ingest mints ids as `str(uuid.UUID(...))`, and graph hits are
    namespaced by `graph_result_id()`. The types can differ for ONE
    memory; they cannot coincide for two.
    """
    ranking: list[tuple[Any, float]] = []
    seen: set[str] = set()
    for position, hit in enumerate(hits):
        key = str(hit.id)
        if key in seen:
            continue
        seen.add(key)
        item = _canonical(by_id, hit)
        if position < limit:
            ranking.append((item, hit.score))
    return ranking


#: Payload key under which a pass carries its vector-floor candidate pool.
_FLOOR_CANDIDATES = "_vector_floor_candidates"


def _merged_candidates(first: Any, later: Any) -> Any:
    """Both passes' floor pools, one entry per memory, strongest kept.

    `Recaller._apply_vector_floor` picks the strongest candidates it is
    shown, so shipping it the union is what lets a later paraphrase's
    better vector hit win a floor slot. Malformed input is left alone
    rather than guessed at: this runs inside a retry that must not fail
    the recall.
    """
    if not isinstance(first, list) or not isinstance(later, list):
        return first
    merged: dict[Any, Any] = {}
    for candidate in [*first, *later]:
        if not isinstance(candidate, dict) or "id" not in candidate:
            continue
        seated = merged.get(candidate["id"])
        if seated is None or _candidate_score(candidate) > _candidate_score(seated):
            merged[candidate["id"]] = candidate
    return list(merged.values())


def _candidate_score(candidate: dict[str, Any]) -> float:
    try:
        return float(candidate.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _canonical(by_id: dict[str, Any], result: Any) -> Any:
    """The one object standing for this memory, richer for each pass.

    Passes disagree about the TYPE of an id, never about the id: a store
    that returns `1` here and `"1"` there is describing one memory both
    times. Everything downstream — the "found nothing new" test and the
    fusion's own deduplication — is fed the object this returns, so both
    see the same memory as one thing. The first arrival is the one kept,
    which keeps the caller's own objects and the scores they already saw.

    But keeping the first object must not mean discarding what the later
    passes learned about that memory. A paraphrase may reach it through a
    different arm, and it may be the pass whose results carry the vector
    floor's candidate list. Dropping that would under-credit the
    retrievers in the documented `sources` field — feedback would then
    reinforce the wrong arms — and could lose the floor's pool entirely
    when the later pass was the only one holding it. So the arms are
    unioned and payload keys the incumbent lacks are filled in. Nothing is
    overwritten: where both passes have an opinion, the first one stands,
    and both are describing the same memory under the same tenant and
    filters anyway, because the retry forwards the caller's scope
    unchanged.
    """
    key = str(result.id)
    incumbent = by_id.get(key)
    if incumbent is None:
        by_id[key] = result
        return result
    if incumbent is not result:
        for source in getattr(result, "sources", []) or []:
            if source not in incumbent.sources:
                incumbent.sources.append(source)
        for field, value in (getattr(result, "payload", None) or {}).items():
            if field == _FLOOR_CANDIDATES and field in incumbent.payload:
                # The one payload key where BOTH passes having an opinion
                # is the normal case, not a collision to resolve by
                # seniority: with the floor on, every result of a pass
                # carries that pass's whole candidate pool. Keeping the
                # incumbent's alone would strand a stronger vector hit
                # that only a later paraphrase saw — it cannot vote (a
                # floor extra is not a ranking) and it would no longer be
                # re-addable either, which is the one route it had left.
                incumbent.payload[field] = _merged_candidates(incumbent.payload[field], value)
                continue
            incumbent.payload.setdefault(field, value)
    return incumbent


def _fresh_trace(caller_trace: Any) -> Any:
    """A trace of the same type as the caller's, or None when it wants none."""
    if caller_trace is None:
        return None
    return type(caller_trace)()


def _absorb(caller_trace: Any, variant_trace: Any, variant: str) -> None:
    """Fold one variant's work into the caller's trace.

    Retriever entries are kept — the extra retrieval is exactly the cost
    an operator reading a trace wants to see — marked `:retry` in the name
    and carrying the paraphrase that produced them in `query`, so two
    paraphrases in one trace stay tellable apart. Degradations and notes
    carry over
    verbatim: an arm that failed during the retry failed, and hiding that
    because it happened in the second pass would be the opposite of what
    a trace is for.
    """
    if caller_trace is None or variant_trace is None:
        return
    # Over a SNAPSHOT: if the two traces were ever the same object this
    # would otherwise append to the list it is walking and never stop.
    # They cannot be today — `_fresh_trace` always builds a new one — but
    # an unbounded loop is a bad thing to leave one refactor away.
    for entry in list(getattr(variant_trace, "retrievers", [])):
        entry.name = f"{entry.name}:retry"
        # `RetrieverTrace.query` is the text this retriever was actually
        # given, and with two paraphrases in the same trace the name alone
        # cannot say which one produced these hits, this latency, this
        # error. An inner expansion may already have recorded its own
        # variant there — that is the text that reached the retriever, so
        # it is kept; the paraphrase fills the field only when nothing
        # more specific claimed it.
        if getattr(entry, "query", None) is None:
            entry.query = variant
        caller_trace.retrievers.append(entry)
    # Copied, not re-`mark`ed: the variant's own trace already emitted the
    # process-wide degradation counter for these tags, and marking them
    # again would count one retry-time failure twice in
    # `/status.degraded_events` and `/metrics`. Copying both lists also
    # preserves the classification the variant made — routine signals stay
    # notes — without this module re-deciding what is routine.
    for tag in getattr(variant_trace, "degraded", []):
        if tag not in caller_trace.degraded:
            caller_trace.degraded.append(tag)
    for tag in getattr(variant_trace, "notes", []):
        if tag not in caller_trace.notes:
            caller_trace.notes.append(tag)


def _retrace(caller_trace: Any, final: list[Any]) -> None:
    """Make the trace's order describe the order actually returned.

    `fused` is documented as the order recall returned, and after a merge
    no single pass produced it, so it is rewritten here to keep that
    invariant true.

    `post_rerank` is NOT, and must not be. It means "the reranker's order
    when a reranker ran", a claim about one component's output on one set
    of candidates — and the field's own contract already allows it to
    differ from the response ("the final response list may still differ if
    vector-floor re-appends items after rerank"). No reranker was ever
    shown the cross-query merge, so copying the fused order into it would
    attribute to the reranker an ordering it never produced and make its
    diagnostics lie exactly when a retry succeeded. It keeps whatever the
    caller's own pass recorded; each paraphrase's reranker reports in its
    own trace.
    """
    if caller_trace is None:
        return
    caller_trace.fused = [(str(r.id), float(getattr(r, "score", 0.0))) for r in final]
