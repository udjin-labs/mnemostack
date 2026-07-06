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
  (`--collection <new>`) and re-index. `mnemostack doctor` detects a
  stored-vs-provider **dimension mismatch** and reports it as `down` with a
  remediation hint, so a misconfigured swap fails the deploy gate instead of
  silently returning garbage. Avoid `--recreate` for a reversible swap — it
  **drops and rebuilds the same collection in place**, leaving nothing to roll
  back to.
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
  pins the root so a nested/single-file refresh keeps the parent index's ids. **If
  the graph is enabled**, a fresh Qdrant collection doesn't fix graph recall —
  `MemgraphRetriever` searches the whole graph and can still surface the old root's
  `:File` link nodes; clean the old graph root too (see
  [`:File` node keying](#file-node-keying-multi-root-graphs)) or use a fresh graph
  database.

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

> ⚠️ This normalizes markers on **every** node and relationship in the target
> database, unscoped. Point it at a **dedicated** mnemostack Memgraph database (or
> back up first) — on a graph shared with non-mnemostack data it will set
> `valid_until = 'current'` on records mnemostack doesn't own.

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

The safe, targeted fix is to delete only the legacy markdown **link** nodes — those
with a `LINKS_TO` edge and no `index_root` — then re-index each root. Current
`:File {path}` and `File`-labelled triple nodes carry `TAGGED`, not `LINKS_TO`, so
they're left untouched:

```cypher
// legacy markdown link nodes only — leaves :File {path}/triple nodes untouched
MATCH (f:File)-[:LINKS_TO]-() WHERE f.index_root IS NULL DETACH DELETE f;
```

> ⚠️ Only **clear/rebuild the whole graph** if it holds nothing but markdown link
> data. `index-markdown` regenerates the link graph, but **not** manually added
> temporal facts (`mnemostack_graph_add_triple` / `GraphStore.add_triple`) or
> ingest tag/file nodes (`Ingestor(graph=...)`, written by `mnemostack index`) —
> clearing drops those for good. Back them up first, or stick to the scoped delete
> above. And never blanket-delete every `index_root`-less `:File` node: that hits
> the live `:File {path}`/triple nodes too.

## Payload validity keys

The stale-fact model uses three payload keys — `invalidated_at` (system-time),
`valid_from` / `valid_until` (world-time). These are 🟢 stable and carry forward.
A collection that never called `invalidate` carries none of them and is
unaffected by the default "hide invalidated" view. Graph validity timestamps are
canonicalized on write (a time-of-day is normalized to a UTC instant; a bare
date stays a date) — values written by 0.8.x are already in the expected shape.

## Multi-tenancy & authentication (new, opt-in)

1.0 adds a multi-tenant boundary and service-key auth. Both are **off by default
and purely additive** — a 0.8.x deployment that ignores them is unchanged, and no
existing data needs migrating.

- **`tenant_id` payload key** — the partition key. It's absent on collections
  written by 0.8.x (they read as single-tenant), and it's **server-owned**: only a
  write that passes `tenant=` sets it, and no client can assert or overwrite it (a
  cross-tenant `upsert` raises `TenantConflictError`; the id-targeted owner-guards
  `set_payload` / `invalidate` / `delete_*` silently skip a foreign-owned id). To
  adopt tenancy on an existing collection, back-fill with
  `mnemostack tenant-migrate --tenant <id>` (operator path; `--dry-run` to preview,
  `--all` to force-relabel) or its library twin
  `VectorStore.stamp_tenant(tenant, only_missing=True)` — both stamp only points
  that lack the key by default, so they're safe to re-run.
- **Stamping needs no re-index — but a later tenant-aware re-ingest does create new
  ids.** `tenant-migrate` / `stamp_tenant` is a payload-only merge: it scopes
  existing points at query time and changes no chunk id or embedding, so un-stamped
  points simply stay outside every tenant's view until stamped. However, chunk ids
  are **tenant-scoped** (`stable_chunk_id(..., tenant=)`), so once you re-ingest the
  same source *under a tenant* the new points get different ids and land **beside**
  the stamped legacy points — recall then sees duplicates. Note that tenant-aware
  ingest is a **library** operation (`Ingestor(tenant=...)`): the `mnemostack index`
  / `index-markdown` CLI has no `--tenant`, and its `--prune` is `index_root`-scoped
  (not tenant-scoped), so it computes the *legacy* ids and would prune the wrong
  set. Reconcile the first tenant re-ingest with the library
  `prune_stale_chunks(store, fresh, tenant=<t>)` (tenant-scoped), or — simplest —
  ingest each tenant into a **fresh collection** and cut over. Stamping alone (no
  re-ingest) has no such duplication.
- **Service keys** — auth on the HTTP/MCP surfaces resolves the tenant from a
  service key (never from the request). Issue one with
  `mnemostack keys add --tenant <t> --scopes read,write` (plaintext printed once;
  the store holds only its SHA-256 hash). Turn the surfaces on with
  `serve --auth` / `mcp-serve --auth` (`--keys-file` / `MNEMOSTACK_KEYS_FILE`
  picks the store; `MNEMOSTACK_AUTH_ENABLED` flips auth on). HTTP `serve` takes a
  per-request bearer / `X-API-Key` header; `mcp-serve` binds one process key via
  `--api-key` / `MNEMOSTACK_API_KEY`. See
  [api-stability.md](api-stability.md#multi-tenancy--authentication).
- **Graph is not tenant-scoped yet.** Under auth no tenant reads another's graph
  nodes, but the two surfaces differ: `mcp-serve --auth` fails the graph tools
  closed and builds recall with no graph, while `serve --auth` still queries the
  configured Memgraph and relies on the `filter_by_tenant` backstop to drop the
  (tenant-less) graph hits — so an authenticated HTTP recall still contacts the
  shared graph. If you need per-tenant graph recall, keep each tenant's graph in a
  separate Memgraph database (or run HTTP with `--memgraph-uri ""`). Full graph
  tenant scoping is planned follow-up work.

## Config & CLI

No config keys or CLI commands are being **removed** on the way to 1.0. Env
aliases (`MNEMOSTACK_QDRANT_URL`, `MNEMOSTACK_MEMGRAPH_URI`, …) are stable. New in
1.0 and additive: the `keys` command, the `--auth` mode on `serve` / `mcp-serve`,
and the `MNEMOSTACK_AUTH_ENABLED` / `MNEMOSTACK_API_KEY` / `MNEMOSTACK_KEYS_FILE`
env vars (all off by default — see the multi-tenancy section above). The only
planned interface change is promoting a few de-facto-public names into their
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
