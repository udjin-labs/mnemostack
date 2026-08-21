"""Reformulate-and-retry for a recall that came back nearly empty.

A question phrased one way can miss memories phrased another — "what did
we decide about auth" against a note that says "we're going with OAuth
after all". The stack has always had the paraphrasing machinery
(:class:`~mnemostack.recall.expansion.QueryExpander`) and the answer path
already retries its own sub-recalls, but `/recall` itself had no policy
for "this returned nothing, try saying it differently".

Opt-in, and deliberately so, because the bill is bigger than it first
looks. One LLM call paraphrases the query, and then EACH variant is the
caller's own recall repeated — the same pipeline, the same reranker — so
with a reranker configured (the `serve` default, when an LLM is present)
a weak recall costs up to THREE LLM calls and two extra retrieval rounds:
one paraphrase, plus a rerank per variant. That is the price of the retry
being the same question rather than a cheaper one, and it is a bill the
operator has to agree to. Off by default, and per-tenant metering makes
it visible when on.

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

from ..llm.base import LLMProvider
from ..observability.recorder import counter
from .flow import recall_flow

#: Paraphrases per retry. Two is enough to change the wording materially;
#: more multiplies retrieval cost for a query that has already failed.
MAX_VARIANTS = 2

#: Default weakness threshold: retry only a recall that returned NOTHING.
DEFAULT_WEAK_BELOW = 1


def is_weak(results: list[Any], below: int = DEFAULT_WEAK_BELOW) -> bool:
    """Whether this recall returned too little to leave alone.

    Counted in MEMORIES, not rows — the same question `_memory_key`
    answers everywhere else in this module. A recall that returned one
    memory under two id representations found one thing, and reading that
    as two would call a genuinely weak recall healthy and skip the retry
    entirely: the gate that decides whether to ask again was the last
    place still counting rows.

    `has_room` below deliberately still counts rows, and that is not the
    same oversight: it asks whether the RESPONSE has space left, and a
    response is made of rows.
    """
    return len({_memory_key(r.id) for r in results}) < max(1, below)


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
    exists to FIND memories a phrasing missed — not to spend a paraphrase
    call, two retrievals and a rerank per variant reshuffling a page that
    is already as long as the caller asked for. Asking in one dimension and forgetting the other
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
    # The paraphrase call is watched rather than guessed at. `ok=False` is
    # how a provider in this stack reports an outage — it does not raise —
    # and `generate_variants` turns that into an empty list, identical from
    # the outside to a healthy model that simply echoed the query (the
    # expander drops a paraphrase equal to the original). Classifying on
    # "did it raise" or "is the list empty" therefore cannot tell an outage
    # from a shrug, and this module spent two rounds proving it: first
    # filing outages as routine, then filing healthy shrugs as faults.
    # Reading the response is the only answer that is true in both cases.
    watched = _WatchedLLM(llm)
    try:
        from .expansion import QueryExpander

        expander = QueryExpander(recaller, watched, n_variants=MAX_VARIANTS)
        variants = expander.generate_variants(query)[:MAX_VARIANTS]
    except Exception:  # noqa: BLE001 — a retry must not fail the recall
        counter("mnemostack.recall.weak_retry_failed", 1)
        _note(flow_kwargs.get("trace"), "weak_retry:paraphrase_failed")
        return results, False
    if not variants:
        # Counted either way, and visible in /metrics — but only a provider
        # that actually reported failure reaches `/status.degraded_events`,
        # through the expander's own `query_expansion.errors`. A healthy
        # shrug is a routine signal, marked so that `notes` (the
        # authoritative list) says what it is.
        counter("mnemostack.recall.weak_retry_no_variants", 1)
        trace = flow_kwargs.get("trace")
        if watched.failed:
            _note(trace, "weak_retry:paraphrase_failed")
        elif trace is not None:
            trace.mark("weak_retry:no_variants")
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
    original_state = [
        (r, r.score, list(r.sources), dict(r.payload or {}), getattr(r, "from_vector_floor", False))
        for r in results
    ]
    by_id: dict[str, Any] = {}
    ranked: list[list[tuple[Any, float]]] = [_one_pass(by_id, results)]
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
    #: Every pass's vector-floor candidates, unioned as the passes finish.
    #: Gathered from the PASS, not from the survivors: fusion cuts to
    #: `limit`, and a hit that loses the cut takes its pool with it — so
    #: the strongest raw vector candidate of the whole retry could be
    #: discarded because the paraphrase that found it lost a tie. What the
    #: floor is owed is the union of what the retry SAW.
    floor_pool = _pass_pool(results)
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
        floor_pool = _merged_candidates(floor_pool, _pass_pool(extra))
        ranked.append(_one_pass(by_id, extra))
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
        # The pool goes in FRONT: `_vector_floor_candidates_from_results`
        # takes the first carrier it meets and stops, so leading with the
        # union is what makes it the pool the floor actually weighs. When
        # no pass carried one, nothing is prepended and the helper keeps
        # its own fallback (derive candidates from the results themselves).
        pool: list[Any] = [_FloorPool(_as_canonical_ids(by_id, floor_pool))] if floor_pool else []
        merged = apply_floor(merged, pool + results + merged)

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
        for result, score, sources, payload, from_floor in original_state:
            result.score = score
            result.sources = sources
            result.payload = payload
            # Every field the merge can touch, including the one round 6
            # added: a restore that covers all but the newest channel is
            # how "we kept your results" starts quietly meaning something
            # else again.
            result.from_vector_floor = from_floor
        return results, True
    gained = len(merged) - original_count
    if gained <= 0:
        final = results if not merged else merged
        _retrace(caller_trace, final)
        return final, True
    counter("mnemostack.recall.weak_retry_recovered", gained)
    _retrace(caller_trace, merged)
    return merged, True


def _one_pass(by_id: dict[str, Any], hits: list[Any]) -> list[tuple[Any, float]]:
    """One pass's RANKING: canonical objects, one entry per memory.

    RRF adds `1/(k+rank)` once per list an item appears in, so a pass that
    listed the same memory twice — `1` here and `"1"` there — would hand it
    two contributions and let a single pass out-vote genuine corroboration
    from two. Canonicalising the objects alone does not prevent that: the
    fusion sees the SAME object twice and scores it twice. One pass, one
    vote, so the duplicate is dropped here where the ranking is built.

    Not every hit is a ranking. `recall_flow` returns the ranked page AND
    the vector floor's guaranteed candidates, items placed there precisely
    BECAUSE the ranking did not choose them. Voting them inverts the
    floor: a candidate appended to two passes collects two contributions,
    out-votes each pass's actual rank-one winner, and can take the page
    outright — after which the final floor step has nothing left to add
    and the real winners are simply gone.

    They are identified by the marker the floor sets when it appends one,
    NOT by position. Position was the first answer here and it was wrong:
    it only holds while the ranked page is full, and a weak recall — the
    only kind this module ever sees — is exactly when it is not, so the
    extras sit at ordinary indices and vote anyway.

    Every hit is still canonicalised, so the caller's metadata, the "found
    something new" test and the recovery count all see the full pass. It
    is only the VOTE that is limited to what the pass actually ranked.

    Identity is `str(id)` throughout, and a collision needs two distinct
    memories whose ids are `1` and `"1"`. Qdrant does not permit it: a
    string point id must be a UUID (`Point id 1 is not a valid UUID`),
    ingest mints ids as `str(uuid.UUID(...))`, and graph hits are
    namespaced by `graph_result_id()`. The types can differ for ONE
    memory; they cannot coincide for two.
    """
    ranking: list[tuple[Any, float]] = []
    seen: set[str] = set()
    for hit in hits:
        key = _memory_key(hit.id)
        if key in seen:
            continue
        seen.add(key)
        item = _canonical(by_id, hit)
        if not getattr(hit, "from_vector_floor", False):
            ranking.append((item, hit.score))
    return ranking


#: Payload key under which a pass carries its vector-floor candidate pool.
_FLOOR_CANDIDATES = "_vector_floor_candidates"


def _memory_key(value: Any) -> str:
    """THE identity rule of this module. Every id it keys on comes here.

    A memory can come back as `1` from one pass and `"1"` from another, so
    anything that asks "are these the same memory" has to ask it the same
    way. Three separate places learned that lesson separately — the
    "found something new" dict, a pass's own ranking, and the floor's
    candidate pool — each after shipping a bug where one memory occupied
    two slots of the caller's page. This function exists so a fourth place
    cannot be added with a fresh answer: key through here, or you are
    inventing a second notion of identity.

    Two DISTINCT memories cannot collide under it. A string point id must
    be a UUID in Qdrant (`Point id 1 is not a valid UUID`), ingest mints
    ids as `str(uuid.UUID(...))`, and graph hits are namespaced by
    `graph_result_id()`.
    """
    return str(value)


def _as_canonical_ids(by_id: dict[str, Any], candidates: Any) -> Any:
    """The pool, speaking the survivors' id representation.

    `Recaller._apply_vector_floor` dedupes against the results by NATIVE
    id, so a candidate carrying `"1"` for a survivor carrying `1` is a
    memory the floor cannot recognise as already present — it appends it,
    and the caller's page shows one memory twice. Agreement cannot be
    asked of the floor here (that blind spot is its own, tracked as
    udjin-labs/mnemostack#168); it can be established BEFORE it runs, by
    handing it the representation this retry settled on. Copied, not
    mutated: these dicts live in the callers' payloads.
    """
    if not isinstance(candidates, list):
        return candidates
    spoken = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or "id" not in candidate:
            continue
        canonical = by_id.get(_memory_key(candidate["id"]))
        if canonical is not None and canonical.id != candidate["id"]:
            candidate = {**candidate, "id": canonical.id}
        spoken.append(candidate)
    return spoken


class _FloorPool:
    """A carrier for the retry's unioned floor candidates.

    `Recaller.apply_vector_floor_after_rerank` reads its pool out of a
    result's payload, so handing it the union means handing it something
    payload-shaped. Deliberately not a `RecallResult`: it is never ranked,
    never returned, and never counted — it exists only to be read.
    """

    __slots__ = ("payload", "sources")

    def __init__(self, candidates: Any) -> None:
        self.payload = {_FLOOR_CANDIDATES: candidates}
        self.sources: list[str] = []


def _pass_pool(hits: list[Any]) -> Any:
    """One pass's floor candidates: the first carrier wins, as upstream.

    Every result of a pass carries the same pool, so the first one holding
    it speaks for the pass.
    """
    for hit in hits:
        candidates = (getattr(hit, "payload", None) or {}).get(_FLOOR_CANDIDATES)
        if isinstance(candidates, list):
            return candidates
    return None


def _merged_candidates(first: Any, later: Any) -> Any:
    """Both passes' floor pools, one entry per memory, strongest kept.

    `Recaller._apply_vector_floor` picks the strongest candidates it is
    shown, so shipping it the union is what lets a later paraphrase's
    better vector hit win a floor slot. Malformed input is left alone
    rather than guessed at: this runs inside a retry that must not fail
    the recall.
    """
    if not isinstance(later, list):
        return first
    if not isinstance(first, list):
        return later
    merged: dict[Any, Any] = {}
    for candidate in [*first, *later]:
        if not isinstance(candidate, dict) or "id" not in candidate:
            continue
        key = _memory_key(candidate["id"])
        seated = merged.get(key)
        if seated is None or _candidate_score(candidate) > _candidate_score(seated):
            merged[key] = candidate
    return list(merged.values())


def _candidate_score(candidate: dict[str, Any]) -> float:
    """A candidate's vector similarity, read the way the floor reads it.

    `Recaller._apply_vector_floor` OVERWRITES `.score` on the candidates
    it appends (`floor_score * 0.999`) — a synthetic value seeded by that
    pass's own ranked page, unrelated to similarity — and orders by
    `payload["raw_vector_score"]` for exactly that reason. Deduping this
    pool by the top-level score therefore compared a real similarity in
    one pass against a rank artefact in another, and could drop the
    stronger observation of a memory while promising "strongest kept".
    Same field, same fallback, same answer as `Recaller._raw_score`.
    """
    payload = candidate.get("payload")
    if isinstance(payload, dict) and "raw_vector_score" in payload:
        raw = payload["raw_vector_score"]
    else:
        raw = candidate.get("score", 0.0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        # The same fallback `Recaller._raw_score` takes, for the same
        # reason: a custom retriever's unparseable `raw_vector_score` must
        # not read as zero here while the floor reads it as the candidate's
        # own score — that disagreement picks a different memory than the
        # floor would have.
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
    key = _memory_key(result.id)
    incumbent = by_id.get(key)
    if incumbent is None:
        by_id[key] = result
        return result
    if incumbent is not result:
        if not getattr(result, "from_vector_floor", False):
            # This pass RANKED it. The incumbent may have reached the page
            # only because the floor guaranteed it, and that is exactly the
            # evidence a paraphrase can overturn — a memory the ranking
            # picks is not a floor extra, whichever pass picked it.
            incumbent.from_vector_floor = False
        for source in getattr(result, "sources", []) or []:
            if source not in incumbent.sources:
                incumbent.sources.append(source)
        for field, value in (getattr(result, "payload", None) or {}).items():
            incumbent.payload.setdefault(field, value)
    return incumbent


class _WatchedLLM(LLMProvider):
    """The caller's LLM, remembering whether it reported a failure.

    `LLMProvider`'s own contract is that providers "handle their own errors
    gracefully — set `error` field in LLMResponse rather than raising", so
    the verdict this records is the one the interface says to look at.

    Per call, never shared: `retry_weak_recall` builds one and drops it, so
    there is no cross-request state to race — the mistake this repo already
    made once with a reranker's `last_fallback_reason`.
    """

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.failed = False

    @property
    def name(self) -> str:
        return str(getattr(self.llm, "name", "unknown"))

    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> Any:
        response = self.llm.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        # Read `ok` directly, like every other consumer in this codebase. A
        # `getattr` hedge was tried and removed: a response that cannot
        # report `ok` raises HERE, on this line, before any default could
        # be consulted — and even if this line were gone, `generate_variants`
        # reads `resp.ok` itself a moment later and raises identically. The
        # failure path is reached either way, so the hedge could not change
        # an outcome. A guard that cannot fire is not defence, it is a claim
        # the tests cannot check.
        if not response.ok:
            self.failed = True
        return response


def _note(trace: Any, tag: str) -> None:
    """Record a degradation on the caller's trace, WITHOUT the counter.

    `RecallTrace.mark` emits the process-wide degradation counter, and
    this module's failures already have their own — a raise emits
    `weak_retry_failed`, and a provider reporting `ok=False` emits
    `query_expansion.errors` from inside the expander. Both are in
    `/status`'s allowlist, so marking here would count one failure twice,
    the same double-count an earlier round removed from the variant path.
    The tag is therefore appended directly: the trace gains the evidence,
    the metric keeps its arithmetic. Which counter belongs to which
    failure is decided at the call sites, not here — see the comment on
    the empty-variants branch for why one of the three cases is counted
    but deliberately not as a degradation.

    Degradations only. `mark()` guarantees that a routine tag written to
    `notes` is ALSO mirrored into `degraded` for back-compat until the
    next major, and this helper cannot honour that contract, so it does
    not offer the choice — anything routine belongs in `mark()`.
    """
    if trace is None:
        return
    if tag not in trace.degraded:
        trace.degraded.append(tag)


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
    caller_trace.fused = [(_memory_key(r.id), float(getattr(r, "score", 0.0))) for r in final]
