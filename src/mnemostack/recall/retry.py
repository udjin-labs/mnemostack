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


def is_weak(results: list[Any], below: int = DEFAULT_WEAK_BELOW, limit: int | None = None) -> bool:
    """Whether this recall is weak enough to be worth asking again.

    Weak means BOTH too few results and room to add some: with a
    threshold above the caller's limit, a full page still counts as
    "below the threshold", and every hit the retry found would then be
    appended past the limit and cut away again — real spend, no change to
    the answer, and a recovery counter that lied about it.
    """
    if limit is not None and len(results) >= limit:
        return False
    return len(results) < max(1, below)


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
    is available to paraphrase with, or when the retry found nothing new.
    Never raises: a failed retry is a recall that did not improve, not a
    failed request.
    """
    if not is_weak(results, below, limit):
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

    seen = {str(r.id) for r in results}
    merged = list(results)
    for variant in variants:
        try:
            # The SAME call in other words: every scoping keyword the
            # caller gave is forwarded unchanged. `QueryExpander.recall`
            # is deliberately not used here — it takes no tenant, so on an
            # authenticated deployment it would read across the boundary
            # (tracked as udjin-labs/mnemostack#166).
            extra = recall_flow(recaller, variant, limit, **flow_kwargs)
        except Exception:  # noqa: BLE001 — same rule, per variant
            counter("mnemostack.recall.weak_retry_failed", 1)
            continue
        for result in extra:
            if str(result.id) not in seen:
                seen.add(str(result.id))
                merged.append(result)

    if len(merged) == len(results):
        return results, True  # asked again, still nothing: an answer too

    merged = merged[:limit]
    budget = flow_kwargs.get("token_budget")
    if budget:
        # Re-applied to the MERGED list: each variant's own flow capped
        # its own results, and concatenating two lists that each fit the
        # budget produces one that does not. The budget is documented as a
        # hard cap on the response, so it has to hold after the merge, not
        # before it.
        from .tokens import apply_token_budget

        merged, _tokens = apply_token_budget(
            merged, budget, flow_kwargs.get("token_counter")
        )
    gained = len(merged) - len(results)
    if gained <= 0:
        return results if not merged else merged, True
    counter("mnemostack.recall.weak_retry_recovered", gained)
    return merged, True
