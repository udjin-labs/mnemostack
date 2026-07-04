# Proposal: stale-fact invalidation (validity timestamps)

**Status:** proposed — awaiting decision before implementation. Payload-schema
choices here are expensive to change later (they become an on-disk contract),
so this doc fixes them first.

## Problem

mnemostack accumulates memories but never marks them stale. When a fact is
superseded — a preference changes, a project moves, a person's role updates —
the old chunk stays in the index at full weight, and the only thing standing
between it and the answer is a prompt instruction ("prefer the most recent",
`answer.py`). That is:

- **Not filterable.** Recall/BM25/temporal all return the stale chunk; nothing
  can scope it out, because there is no payload field that says "no longer
  true". The `filters=` path has no negation (`must_not`) today either.
- **Not point-in-time.** You cannot ask "what did we believe as of March?" on
  the vector side. The graph side already can (`query_triples(as_of=...)`).
- **Prompt-only and lossy.** "Prefer the most recent" fails when both facts are
  recalled but the model can't tell which superseded which, and it silently
  breaks on corpora where recency ≠ correctness.

This is the single strongest "table-stakes" signal from the mid-2026 landscape
research: Mem0 ("forgetting and update — overwriting stale memories rather than
accumulation"), Letta (reflect/consolidate), and Zep (`valid_at`/`invalid_at`/
`expired_at` bi-temporal invalidation) all treat it as core.

**Crucially, mnemostack already has half of this — on the graph side.**
`GraphStore.invalidate()` closes a fact non-destructively by setting its edge
`valid_until` to the end timestamp; `MemgraphRetriever` recall already returns
only `valid_until = 'current'` edges; `query_triples(as_of=...)` already does
point-in-time. The vector side has no equivalent. **This proposal aligns the
vector side with the model the graph already ships** — same vocabulary, same
non-destructive close, same "current by default" recall.

## Scope

**In scope (this proposal / stage 1):** a validity model on vector-store
payloads, the API to set it, and default recall behavior that hides invalidated
facts. **Out of scope (stage 2, separate effort):** automatic detection of
*which* fact supersedes which (background consolidation / LLM reflection). Stage
1 gives the caller the mechanism; stage 2 later automates the policy.

## Model

Bi-temporal-lite, matching the graph's existing `valid_from`/`valid_until`
vocabulary rather than inventing a third scheme. Three OPTIONAL payload keys,
all ISO-8601 strings, all absent by default (so existing chunks and existing
behavior are unchanged):

| Key | Meaning | Analogue |
| --- | --- | --- |
| `valid_from` | event-time: fact became true at | graph `valid_from`; Zep `valid_at` |
| `valid_until` | event-time: fact stopped being true at (its successor's start) | graph `valid_until`; Zep `invalid_at` |
| `invalidated_at` | system-time: when *we recorded* it as stale | Zep `expired_at` |

Two distinct axes on purpose (this is the "bi-temporal" part, and the reason to
decide now):

- **`valid_until`** is about the *world*: "this was true until March." Set it
  when you know the fact's real-world end. Drives point-in-time (`as_of`).
- **`invalidated_at`** is about *our knowledge*: "we learned on 2026-07-04 this
  was wrong." Set it when superseding a fact regardless of when it actually
  became false. Drives default recall exclusion.

A fact is **current** iff `invalidated_at` is absent. Default recall returns
only current facts. This mirrors the graph's `coalesce(valid_until,'current') =
'current'` recall filter, but on the system-time axis so that "we found out it's
stale" and "it expired in the world" stay separable (Zep keeps them separate for
exactly this reason — you can be wrong about world-time and still want the fact
gone from default recall).

`indexed_at` (already written at ingest) stays as-is: it is neither of these —
it's "when the bytes landed", not "when the fact was true/known".

## API

### Library — new `VectorStore` method

```python
def invalidate(
    self,
    ids: str | list[str],
    *,
    invalidated_at: str | None = None,   # default: now(UTC), ISO-8601
    valid_until: str | None = None,       # optional world-time end
    index_root: str | None = None,        # owner guard, like --refresh-payloads
) -> int:
    """Mark chunks stale without deleting or re-embedding them.

    Sets ``invalidated_at`` (and optionally ``valid_until``) on each point's
    payload via ``set_payload`` (merge; vector untouched). Non-destructive:
    the chunk stays searchable via ``include_invalidated=True`` and for
    point-in-time recall. Returns the number of points updated.
    """
```

Built entirely on primitives that already exist and are already exercised by
`--refresh-payloads`: `set_payload` (merge), the `index_root` owner-guard
pattern, `scroll`/`iter_ids`. No new storage concept.

### Library — recall gains a default exclusion

```python
recall_flow(recaller, query, ...,
            include_invalidated: bool = False,   # default: hide stale facts
            as_of: str | None = None)            # point-in-time (world-time)
```

- `include_invalidated=False` (default): drop any hit whose payload has
  `invalidated_at`. **This is a behavior change** — see Back-compat.
- `as_of="2026-03-01"`: return facts valid at that instant — `valid_from <=
  as_of AND (valid_until absent OR valid_until > as_of)` — the same predicate
  `query_triples(as_of=...)` already uses on the graph.

### Filter primitive — add negation

The exclusion needs `must_not` on the Qdrant side, which does not exist today
(`_build_filter` is AND-only, `must=[...]`). Add an `IsEmpty`/`must_not`
condition for "`invalidated_at` is absent". This is the one genuinely new
low-level primitive; it is also independently useful (any future "exclude
tag X" filter wants it).

### Surfaces (thin pass-throughs, mirroring the token-budget rollout)

- **HTTP:** `include_invalidated` / `as_of` on `POST /recall` and `/answer`; a
  new `POST /invalidate` (`{ids, invalidated_at?, valid_until?}`) — the first
  vector-payload write endpoint the server exposes, so it stays behind the same
  "front it with your own auth" note as the rest of the write-capable surface.
- **MCP:** same two recall params on `mnemostack_search`/`mnemostack_answer`; a
  new `mnemostack_invalidate` tool (parallel to the existing
  `mnemostack_graph_add_triple` write tool).
- **CLI:** `mnemostack invalidate <id>... [--valid-until] [--index-root]`, and
  `--include-invalidated` / `--as-of` on `search`/`answer`.
- **Async:** `AsyncVectorStore` currently lacks `set_payload`/`scroll`/
  `iter_ids` — add them so `invalidate` has an async mirror, consistent with the
  async-API work just landed.

## Recall default: exclude, but keep discoverable

Invalidated facts are **hidden from default recall, never deleted**. They remain
reachable three ways: `include_invalidated=True`, point-in-time `as_of`, and
(unchanged) direct id fetch. Rationale: deletion loses audit history and makes
"what did we believe in March" impossible; the graph already chose
non-destructive close, and the two sides should agree.

## Back-compat

- All three keys are optional and absent on every existing chunk. Chunks
  without `invalidated_at` are current — so **existing indexes need no
  migration** (same stance the graph took: legacy `NULL` = current).
- The **one** behavior change is `include_invalidated` defaulting to `False`.
  For any index that has never called `invalidate`, this is a no-op (no chunk
  has the key). It only changes results once you start invalidating — which is
  the point. Documented in CHANGELOG under a behavior-change note.
- `stable_chunk_id` is untouched: invalidation is a payload write, not an id
  change, so re-ingesting the same content does not resurrect staleness and does
  not require a hash-scheme change (which remains a major-version-only event per
  the fixed design principle).
- Ingest interaction: re-ingesting identical content upserts onto the same id
  and **would carry the old payload forward via merge** — but `upsert` writes a
  fresh payload, so a re-ingest of a fact clears `invalidated_at` (the fact is
  asserted true again). This is the correct semantics (you re-stated it) and is
  called out so it isn't a surprise.

## Open decisions (the reason this is a proposal, not a PR)

1. **Two keys or three?** Full bi-temporal (`invalidated_at` system-time +
   `valid_until` world-time) vs. collapsing to just `invalidated_at`. Recommend
   **both**, because the graph already exposes world-time (`valid_until`) and
   dropping it on the vector side would make the two halves disagree — but it is
   more surface. Cutting to `invalidated_at`-only is a viable smaller stage 1.
2. **Default `include_invalidated=False`** — accept the (opt-in-triggered)
   behavior change, or make exclusion opt-*in* (`exclude_invalidated=True`)?
   Recommend default-exclude: a stale fact resurfacing is the bug users are
   asking us to fix; an index that never invalidates sees no change anyway.
3. **`POST /invalidate` on the HTTP server at all?** The server is read+feedback
   today and never writes vector payloads. Adding a write endpoint is a real
   surface-area step. Alternative: keep invalidation CLI/library-only for stage
   1 (matching how `GraphStore.invalidate` is library-only today) and add the
   endpoint later if asked. Recommend **library + CLI + MCP, defer HTTP** unless
   you want it now.

## Relationship to existing prior art

This is deliberately the vector-side twin of `GraphStore.invalidate`: same
`valid_from`/`valid_until` vocabulary, same non-destructive close, same
"current-by-default" recall, plus the system-time `invalidated_at` axis that Zep
validated. It reuses the `--refresh-payloads` write path (`set_payload` merge +
`index_root` guard) wholesale. The only new low-level capability is `must_not`
negation in the Qdrant filter builder.