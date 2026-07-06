# Migration notes: 0.8.x → 1.0

mnemostack is on the road to 1.0. This page tracks the on-disk and interface
contracts that a 0.8.x deployment relies on, what 1.0 will preserve, and the one
migration that exists today. It's kept current as 1.0 work lands.

**TL;DR** — 0.8.x data (Qdrant collections and Memgraph graphs) is expected to
carry forward to 1.0 **without re-indexing**. The only active migration is a
graph marker normalization (`graph-migrate-current`), and it's optional and
back-compatible. Run `mnemostack doctor` after any upgrade as the go/no-go gate.

> Verify any upgrade with:
> ```bash
> mnemostack doctor              # 0 healthy / 1 dependency down / 2 config invalid
> pytest tests/test_smoke.py -q  # end-to-end smoke (source checkout; in-memory)
> ```

---

## Qdrant collections & embedding dimension

The vector **dimension is fixed when a collection is created** and is set by the
embedding provider/model. Changing the embedding model (or provider) to one with
a different dimension makes the existing collection unreadable — recall would
return noise.

- **1.0 preserves** existing collections created with the same embedding
  model. No re-index needed.
- **If you change the embedding model**: create a **new** collection
  (`--collection <new>` or a fresh `--recreate`) and re-index. `mnemostack
  doctor` detects a stored-vs-provider **dimension mismatch** and reports it as
  `down` with a remediation hint, so a misconfigured swap fails the deploy gate
  instead of silently returning garbage.
- **Rollback**: point `vector.collection` back at the old collection; it's
  untouched by a new-collection re-index.

## Chunk ids & `index_root`

Chunk ids are deterministic so re-indexing is idempotent:

- `mnemostack index` derives ids from `(source, offset, text)`.
- `mnemostack index-markdown` derives ids that **include `index_root`** (the
  corpus root), so the same file under a different root is a different chunk.

Implication: **keep `index_root` stable across re-indexes of the same corpus.**
Changing it (or moving the corpus so relative `source` paths change) produces a
new id set; the old chunks become orphans.

- **1.0 preserves** the current id derivation — ids computed on 0.8.x remain
  valid.
- **If you must change roots/paths**: re-index into a **fresh collection**
  (`--collection <new>`) and cut over, then drop the old one — the cleanest path,
  with no orphans left behind. Re-indexing **in place does not remove the old
  root's chunks**: `--prune` only walks and prunes the root you pass, so chunks
  still carrying the previous `index_root` fall outside its scope and stay as
  orphans (and generic `mnemostack index` keys ids by `(source, offset, text)`, so
  unchanged chunks keep their ids regardless). To clean in place, delete the old
  root's chunks explicitly before relying on recall. For markdown, `--index-root`
  pins the root so a nested/single-file refresh keeps the parent index's ids.

## Graph markers: `valid_until = 'current'` vs legacy `NULL`

Open-ended graph edges/facts use a sentinel `valid_until = 'current'`. Graphs
written by early versions used `NULL` instead. **Reads treat `NULL` as current**,
so a legacy graph keeps working — this is non-breaking. The one-off migration
normalizes the marker (on both nodes and relationships) so all downstream
tooling sees a single form:

```bash
# Preview what would change (no writes):
mnemostack graph-migrate-current --memgraph-uri bolt://localhost:7687 --dry-run

# Apply:
mnemostack graph-migrate-current --memgraph-uri bolt://localhost:7687
```

- Optional on 0.8.x (reads already treat `NULL` as current). Recommended before
  1.0 so the `'current'` marker is the only open-ended form on disk.
- **Rollback**: the migration only sets `valid_until = 'current'` where it was
  `NULL` on open edges; it does not delete or re-time anything. There's no
  reverse command, but the change is read-equivalent to the pre-migration state.
- `graph-migrate-current` is a **transitional** command and will be removed once
  no pre-marker graphs remain (see [api-stability.md](api-stability.md)).

## `:File` node keying (multi-root graphs)

Markdown `:File` graph nodes are keyed by `(name, index_root)`, so two corpora
that share a filename (both have `index.md`) stay distinct nodes and only surface
their own root's `LINKS_TO` edges. `:Entity` nodes (from triples) have no
`index_root`. This is a 🟢 stable contract; no migration needed for graphs first
written by 0.8.x.

If you indexed multiple markdown roots into one graph on an **older** version that
keyed `:File` link nodes by `name` alone, re-indexing is **not** enough on its own:
the new writes `MERGE` fresh `(name, index_root)` nodes but leave the legacy
name-only link nodes — and their `LINKS_TO` edges — in place, where recall still
surfaces them under the empty-root key.

The safe fix is to **rebuild the link graph**: point at a fresh graph database (or
clear it) and re-index all roots. The graph is derived data, so re-indexing
regenerates it — no surgical deletes to get wrong.

> ⚠️ Do **not** blanket-delete `:File` nodes that lack `index_root`. Current ingest
> writes tag/file nodes without one — `:File {path}` with `TAGGED` edges, and
> `File`-labelled triple subjects — so `WHERE index_root IS NULL DETACH DELETE`
> would drop **live** data. If you must clean in place instead of rebuilding, scope
> to the legacy *link* nodes only (a `LINKS_TO` edge and no `index_root`); the
> current `:File {path}`/triple nodes carry `TAGGED`, not `LINKS_TO`, so they're
> excluded:
>
> ```cypher
> // legacy markdown link nodes only — leaves :File {path}/triple nodes untouched
> MATCH (f:File)-[:LINKS_TO]-() WHERE f.index_root IS NULL DETACH DELETE f;
> ```

## Payload validity keys

The stale-fact model uses three payload keys — `invalidated_at` (system-time),
`valid_from` / `valid_until` (world-time). These are 🟢 stable and carry forward.
A collection that never called `invalidate` carries none of them and is
unaffected by the default "hide invalidated" view. Graph validity timestamps are
canonicalized on write (a time-of-day is normalized to a UTC instant; a bare
date stays a date) — values written by 0.8.x are already in the expected shape.

## Config & CLI

No config keys or CLI commands are being **removed** on the way to 1.0. Env
aliases (`MNEMOSTACK_QDRANT_URL`, `MNEMOSTACK_MEMGRAPH_URI`, …) are stable. The
only planned interface change is promoting a few de-facto-public names into their
package `__all__` (`make_graph_store`, `InMemoryRecorder`, `synthesize_async`) —
additive, not breaking. See [api-stability.md](api-stability.md) for the full
stable/experimental split; anything marked 🟡 experimental there (pipeline
internals, `synthesize`, alternate rerankers) may change in a minor release
before 1.0.

## Upgrade checklist

1. Read the [CHANGELOG](../CHANGELOG.md) `[Unreleased]` / release section for the
   target version.
2. Upgrade the package; keep your existing config file.
3. `mnemostack doctor` — must exit `0`. A `disabled` (unconfigured) or `warn`
   (configured-but-unreachable) graph line is expected and does **not** affect
   the exit code; exit `1` means embedding or Qdrant is down, exit `2` means the
   config is invalid.
4. If the graph is configured: run `graph-migrate-current --dry-run`, then apply
   if it reports changes.
5. From a **source checkout**, `pytest tests/test_smoke.py -q` against a staging
   copy; on an installed host without the test tree, rely on the step-3
   `mnemostack doctor` diagnostics plus the step-6 spot-check.
6. Spot-check a representative `search` / `answer`.
