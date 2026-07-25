# API stability: stable vs experimental

This page tells you which mnemostack surfaces are **safe to build on** and which
may change. It's the contract for the road to 1.0.

**Stability levels**

- 🟢 **Stable** — supported for production. Breaking changes only with a major
  version bump, and only with migration notes. Names, defaults, and documented
  error behavior won't change under you within a major line.
- 🟡 **Experimental** — useful but may change (signature, behavior, or removal)
  in any minor release. Pin a version if you depend on it.
- 🔴 **Internal** — not a public API even if importable. No stability promise;
  don't build on it.

> Scope: this covers the surfaces an integrator touches — the Python library,
> CLI, config, MCP tools, HTTP API, and the on-disk payload/graph contracts.
> Anything not listed here is 🔴 internal by default.

---

## HTTP API (`mnemostack serve`)

🟢 **Stable** — this is the primary integration surface and is versioned
carefully.

| Endpoint | Level | Notes |
| --- | --- | --- |
| `POST /recall` | 🟢 | `RecallRequest` → `RecallResponse`. Fields are additive-only within a major line. |
| `POST /answer` | 🟢 | `AnswerRequest` → `AnswerResponse`. |
| `POST /feedback` | 🟢 | `FeedbackRequest` → `FeedbackResponse`. |
| `GET /health` | 🟢 | Existing shape preserved. |
| `GET /healthz` | 🟢 | Liveness; `{status, version}`. |
| `GET /readyz` | 🟢 | Readiness; 503 when Qdrant/embedding down. Graph is deliberately not gated. |
| `GET /status` | 🟢 | Operator snapshot. Counter set may grow (additive). |
| `GET /metrics` | 🟢 | Prometheus text. Individual metric names: see Observability below. |

Contract guarantees: request fields are only ever added (never removed or
retyped) within a major line; new response fields may appear, so parse
leniently. `build_app(config)` and `ServerConfig` are 🟢 stable.

**Authentication (opt-in, 🟢 stable).** `serve --auth` (or `MNEMOSTACK_AUTH_ENABLED`)
turns the server into a default-deny, multi-tenant surface: `/recall`, `/answer`,
and `/feedback` require a service key (bearer token), which resolves to a
`Principal(tenant, scopes)` — `read` gates recall/answer, `write` gates feedback.
The tenant is derived from the key, never asserted by the caller, and scopes every
result. `/health`, `/healthz`, `/readyz`, `/status`, `/metrics` stay public. With
`--auth` off the server is single-tenant and unauthenticated exactly as before —
this is purely additive. See [Multi-tenancy & authentication](#multi-tenancy--authentication).

## CLI

🟢 **Stable** commands and their documented flags / exit codes:
`health`, `doctor`, `search`, `answer`, `index`, `index-markdown`, `invalidate`,
`feedback`, `serve`, `mcp-serve`, `keys`, `quota`, `tenant-migrate`, `tenant-export`,
`tenant-rm`, `init`, `config`.

- `tenant-migrate --tenant <id>` stamps existing points into a tenant (idempotent;
  `--all` to force-relabel, `--dry-run` to preview) — the operator path to adopting
  tenancy on a pre-tenant collection; `VectorStore.stamp_tenant` is its library twin.
  Add `--memgraph-uri <bolt>` to also stamp the graph (nodes + edges), the twin of
  `GraphStore.stamp_tenant`.
- `tenant-export --tenant <id>` dumps the tenant's vector points as JSONL (vectors +
  payloads; `--no-vectors` for payloads only; `-o` file or stdout). Keys, quotas,
  learning state, and graph records are deliberately not exported — the points are
  the portable source of truth.
- `tenant-rm --tenant <id>` offboards a tenant across every store (vector points,
  graph with `--memgraph-uri`, service keys, quota, learning-state partitions):
  counts first, `--dry-run` previews, the deletion requires `--yes`, and a per-store
  failure exits nonzero listing what remains. Library twins:
  `VectorStore.delete_tenant`, `GraphStore.delete_tenant`, `StateStore.delete(key)`
  (on the shipped stores).
- `keys add` / `keys list` / `keys revoke` manage the service-key store for
  multi-tenant auth. `keys add --tenant <t> --scopes read,write` prints the
  plaintext key **once** (only its SHA-256 hash is stored); `--keys-file` (or
  `MNEMOSTACK_KEYS_FILE`) picks the store, defaulting to
  `~/.config/mnemostack/keys.json`.
- `serve --auth` and `mcp-serve --auth` gate their surfaces behind a service key
  (`--keys-file` / `MNEMOSTACK_KEYS_FILE` picks the store; `MNEMOSTACK_AUTH_ENABLED`
  flips auth on). HTTP `serve` reads a per-request key from the `Authorization:
  Bearer` / `X-API-Key` header (many keys/tenants); `mcp-serve` binds **one**
  process principal from `--api-key` / `MNEMOSTACK_API_KEY`. Default-off; see
  [Multi-tenancy & authentication](#multi-tenancy--authentication).
- `quota set` / `quota list` / `quota rm` manage per-tenant resource limits — a
  `max_points` storage cap and a `max_rps` request-rate cap (with optional
  `--burst`) — in a per-tenant quota store (`--quotas-file` /
  `MNEMOSTACK_QUOTAS_FILE`, default `~/.config/mnemostack/quotas.json`). `quota set`
  is a **partial update** (only the fields you pass change; `none` clears one). The
  storage cap is enforced at ingest — `index-markdown --tenant <id>` (and the
  library `Ingestor(max_points=)`) refuse a write that would exceed it. The rate
  cap is enforced at the authenticated HTTP surface — `serve --auth` returns **429**
  (with `Retry-After`) when a tenant exceeds its `max_rps` (a per-process token
  bucket; see [Multi-tenancy & authentication](#multi-tenancy--authentication)). A
  quota is a resource guardrail, not a security boundary, so a corrupt quota store
  **fails open** (no limit, logged) rather than blocking traffic.

- `inspect` runs an operator web console (default `127.0.0.1:8100`): read-only
  data browsing by default, or a tenant-administration console (issue/revoke keys,
  manage quotas) with `inspect --auth` (admin-scoped key required). The **CLI flags**
  are stable; the console's **HTML/JSON `/api` surface is 🟡 experimental** — it's an
  operator tool, not a programmatic API, so don't build automation against it.
- `doctor` exit codes (`0` healthy / `1` core dependency down / `2` config
  invalid) are a 🟢 stable contract — safe to gate CI/deploys on. Exit `2` covers a
  config that **loads** but fails `doctor`'s validation; a value that fails to
  **parse** (e.g. a non-numeric `MNEMOSTACK_TOKEN_BUDGET`) is rejected at CLI
  startup before any command runs, so treat any nonzero startup failure as
  "config unusable" in your gate.
- `--json` output shapes on `search` / `answer` / `invalidate` / `feedback` /
  `doctor` are 🟢 stable (additive fields only).

🟡 **Experimental**:
- `synthesize` — the entity-synthesis command and its `--format json` shape may
  change.
- `--tier` (on `search` / `answer`) — the tiered result-shaping convenience.

🔴 **Transitional (do not depend on)**:
- `graph-migrate-current` — a one-off migration to backfill graph `valid_until`
  markers (NULL → `'current'`). It will be removed once no pre-marker graphs
  remain. See [migration notes](migration-0.8-to-1.0.md).

## Config (file + `MNEMOSTACK_*` env)

🟢 **Stable** — all documented keys (and their `MNEMOSTACK_*` env alias where one exists):
`embedding.{provider,model}`,
`vector.{host,collection,chunk_size,window_size,health_timeout}`,
`llm.{provider,model}`,
`graph.{uri,user,password,database,timeout}`,
`recall.{bm25_paths,vector_floor,rerank_mode,token_budget}`,
`recall.{text_key,timestamp_key,timestamp_format}` (payload schema of the
collection recall reads — mounts a pre-existing collection with its own field
names / numeric epoch timestamps; `timestamp_format` ∈ `iso|epoch|epoch_ms`),
`recall.text_search` (lexical-arm selector ∈
`auto|off|bm25|qdrant_bm25|lexical|sparse` — the mode names and their
semantics are the stable contract; `QdrantTextRetriever` /
`QdrantSparseRetriever` / `SparseTextEncoder` class shapes are 🟡
experimental).

🟡 **Accepted but not yet wired** — present in the config schema, but the runtime
does not consume them today, so don't depend on them until they are:
`embedding.api_key_env` (only drives a CLI "set `$KEY`" hint, not provider key
resolution — the provider reads its own fixed env var), `embedding.ollama_host`,
`vector.overlap` (the chunker's overlap is currently fixed), `recall.rrf_k`
(the RRF constant is currently fixed at 60), and `graph.health_timeout` (honored
by the env / ASGI-factory server via `ServerConfig.from_env`, but the `mnemostack
serve` CLI has no matching flag and uses the 1s default). These will be wired
through or removed; either way it won't silently change a working deployment.

🟢 **CLI-only defaults**: `recall.top_k` and `recall.confidence_threshold` are
stable as defaults for the `search` / `answer` CLI (`--limit` / `--min-confidence`),
but the HTTP and MCP surfaces don't read them — `/recall` and the MCP tools take a
per-request `limit` (default 10) and apply no server-side confidence threshold. Set
the result count per request there.

`graph.timeout` bounds graph connections for `serve`, `mcp-serve`,
`index-markdown`, and `graph-migrate-current`; the CLI recall commands (`search` /
`answer` / `synthesize`) don't thread it into their graph retriever, which uses the
built-in default.

`vector.health_timeout` bounds the Qdrant readiness probe on the **HTTP server**
(`/readyz`, `/status`, `/health` — via `serve --qdrant-health-timeout`); `/healthz`
is liveness-only and never checks Qdrant, and the `mnemostack health` CLI and MCP
health tool build the store without the timeout (client default).

🟢 **Auth env (no YAML key)** — the multi-tenant auth surface is driven by env /
flags only, not the config file: `MNEMOSTACK_AUTH_ENABLED` (truthy →
`serve`/`mcp-serve` require a key), `MNEMOSTACK_API_KEY` (the process's service
key — **`mcp-serve` only**; HTTP `serve` reads a per-request bearer/`X-API-Key`
header instead), `MNEMOSTACK_KEYS_FILE` (service-key store path). All three are
🟢 stable and off by default.

Env aliases (e.g. `MNEMOSTACK_QDRANT_URL` for `vector.host`,
`MNEMOSTACK_MEMGRAPH_URI` for `graph.uri`) are 🟢 stable **where they exist** — not
every stable key has one. `vector.chunk_size` and `vector.window_size`, for
example, are set via the YAML file or CLI flags only (no `MNEMOSTACK_*` override),
and they tune the generic `index` command — `index-markdown` uses a fixed chunk
size (1200) and no windowing, so it ignores both.
Defaults won't change
without a major bump. `graph.uri = null` (disabled) is the documented off state for
the **library** recall path — but the **HTTP server defaults it on**: both
`mnemostack serve` and the `ServerConfig.from_env()` ASGI factory expand an unset
`graph.uri` to `bolt://localhost:7687`. To keep graph off there, pass
`--memgraph-uri ""` (CLI) or construct `ServerConfig(graph_uri=None)`
programmatically — an empty `MNEMOSTACK_MEMGRAPH_URI` does **not** work, since the
env override ignores empty values. `token_budget <= 0` normalizing to "no budget"
is a 🟢 documented behavior.

## MCP tools (`mnemostack mcp-serve`)

🟢 **Stable**: `mnemostack_search`, `mnemostack_answer`, `mnemostack_health`,
`mnemostack_invalidate`, `mnemostack_feedback` — parameters are additive-only.

🟡 **Conditional**: `mnemostack_graph_query`, `mnemostack_graph_add_triple` are
only registered when a graph URI is configured — a client must not assume they
exist. Their shapes are otherwise stable. Under `--auth` they are tenant-scoped
(the query is confined to the key's tenant; the write is stamped with it), not
fail-closed — see below.

**Under `mcp-serve --auth` (🟢 stable).** The stdio transport binds one client per
process, so the server runs *as* a single principal: the key (`--api-key` /
`MNEMOSTACK_API_KEY`) resolves to a tenant + scopes at boot (and boot-fails loud
without a valid key). Every tool call re-verifies the key, so revocation takes
effect immediately mid-session. `search`/`answer` require `read`,
`invalidate`/`feedback` require `write`, and the tenant is threaded into recall,
answer, and invalidate. The graph is **tenant-scoped** (see
[Multi-tenancy & authentication](#multi-tenancy--authentication)): the recall
pipeline keeps its graph-resurrection stage (confined to the caller's tenant), and
the graph tools are tenant-scoped rather than fail-closed — `mnemostack_graph_query`
is confined to the key's tenant, `mnemostack_graph_add_triple` (still `write`-scoped)
is stamped with it. `mnemostack_health` stays public. Without `--auth` the tools
behave exactly as listed above.

## Python library

### 🟢 Stable

- **Ingest**: `Ingestor` (constructor + `ingest` / `ingest_one` / `ingest_async`
  / `ingest_one_async` / `stream`), `IngestItem`, `IngestStats`.
- **Vector**: `VectorStore`, `AsyncVectorStore`, `Hit`, `DimensionMismatchError`,
  `TenantConflictError`, `TENANT_ID_KEY` — the documented methods (`upsert`,
  `search`, `count`, `invalidate`, `ensure_collection`, `collection_exists`,
  `set_payload`, `scroll`). Note the async surface is a **subset** of sync (no
  `scroll` / `iter_ids` / `delete_points` / `delete_payload_keys` /
  `index_payload_field` / `stamp_tenant`) — parity is planned but not guaranteed
  yet, so treat async-only-missing methods as 🟡. In particular the tenant helpers
  below (`delete_payload_keys`, `stamp_tenant`) are **`VectorStore`-only** today. The
  read/write/delete methods take an **optional keyword `tenant=`** (additive,
  default `None` = single-tenant, unchanged behavior); when set, the server stamps
  and filters on the `tenant_id` payload key and refuses to touch another tenant's
  points. Two enforcement shapes: `upsert` / `upsert_batch` **raise
  `TenantConflictError`** if an id is already owned by a different tenant (a guessed
  id can't overwrite another tenant's point), while the id-targeted owner-guarded
  writes — `set_payload`, `invalidate`, `delete_payload_keys`, `delete_points` —
  **silently skip** foreign-owned ids (they leave them untouched and return without
  error, so don't write `except TenantConflictError` around them). `stamp_tenant(
  tenant, only_missing=True)` backfills the key on a pre-tenant collection.
- **Embeddings**: `EmbeddingProvider` (incl. `health_check`), `get_provider`,
  `list_providers`, `register_provider`.
- **LLM**: `LLMProvider` (incl. `health_check`), `LLMResponse`, `get_llm`,
  `list_llms`, `register_llm`.
- **Recall (core)**: `Recaller` (`recall` / `recall_async`), `recall_flow`,
  `recall_flow_async`, `RecallResult`, `AnswerGenerator`, `Answer`,
  `Reranker`, `RERANK_MODES`, `BM25`, `BM25Doc`, `reciprocal_rank_fusion`,
  `build_bm25_docs`, the validity helpers (`filter_by_validity`, `is_current`,
  `valid_at`), the tenant backstop `filter_by_tenant`, and token helpers
  (`estimate_tokens`, `apply_token_budget`, `sum_tokens`, `TokenCounter`), and the
  trace helpers (`RecallTrace`, `apply_rerank_safe`). `recall_flow` /
  `recall_flow_async` and `Recaller.recall` / `recall_async` take an **optional
  keyword `tenant=`** (additive, default `None`) that scopes retrieval to that
  tenant and applies `filter_by_tenant` as a backstop. `AnswerGenerator.generate`
  (+ async) also takes `tenant=`, but it only scopes the method's **own** internal
  retry sub-recalls (expansion / inference) — it does **not** re-filter the
  `memories` you pass in. Pre-scope those yourself: feed `generate` the output of a
  `tenant`-scoped `recall_flow`, don't hand it mixed-tenant memories expecting the
  `tenant=` argument to clean them.
- **Auth (multi-tenant service keys)**: `mnemostack.auth` — `Principal`
  (`tenant`, `scopes`, `.can(scope)`, plus an optional `key_id` — the public id
  of the key that authenticated, for audit attribution; `None` on backends whose
  records carry no id), `KeyStore` (Protocol: `.verify(key) ->
  Principal | None`), `FileKeyStore` (`.verify` / `.issue` / `.revoke` /
  `.list_keys`; SHA-256-hashed JSON store), `SCOPES` (`{"read", "write",
  "admin"}`; `admin` implies the others), `KeyStoreError`, `hash_key`,
  `default_keys_path`, `make_key_store` (the backend factory the servers use —
  selected by `MNEMOSTACK_KEYSTORE`: `file` default, `openbao` for
  `mnemostack.openbao.OpenBaoKeyStore`, a **verify-only** KV-v2 adapter; a
  selected-but-misconfigured backend fails the server at boot). This is the
  credential surface behind `serve --auth` / `mcp-serve --auth`; keys are stored
  hashed and the plaintext is shown once. The env selection contract (`file` /
  `openbao` names, fail-loud) is 🟢 stable; `OpenBaoKeyStore`'s constructor shape
  is 🟡 experimental.
- **Quotas (per-tenant resource limits)**: `mnemostack.quotas` — `TenantQuota`
  (`max_points`, `max_rps`, `burst`, `.effective_burst()`), `QuotaStore` (Protocol:
  `.get(tenant) -> TenantQuota | None`), `FileQuotaStore` (`.get` / `.set` /
  `.remove` / `.list_quotas`; `.set` is a partial update), `QuotaExceededError`,
  `RateLimitExceededError`, `QuotaStoreError`, `enforce_points_quota`,
  `default_quotas_path`. The storage-cap surface behind `quota` /
  `Ingestor(max_points=)`, and the rate-cap config behind `serve --auth`; fails open
  on a broken store. Rate-limit mechanics live in `mnemostack.ratelimit`
  (`RateLimiter`, `TokenBucket`) — 🟡 experimental (the enforcement is the stable
  contract, not the class shapes).
- **Audit (control-plane trail)**: `mnemostack.audit` — the *behavior contract*
  is 🟢 stable: opt-in via `MNEMOSTACK_AUDIT_FILE` (unset = nothing written),
  append-only JSONL events (`ts`/`action`/`actor`/`surface`/`outcome`/`tenant`/
  `details`), best-effort writes that never raise into the audited operation,
  and **no key material ever** (public key ids only). The class shapes
  (`FileAuditLog` / `NullAuditLog` / `AuditLogError` / `audit_log_from_env`) are
  🟡 experimental.
- **Markdown**: `collect_markdown`, `MarkdownChunk`, `LinkEdge`,
  `MarkdownCollection`, `parse_frontmatter`, `extract_links`, `MarkdownSyncer`.
- **Graph**: `GraphStore` (documented methods), `make_graph_store` (the
  constructor used everywhere — currently only importable as
  `mnemostack.graph.factory.make_graph_store`; **to be re-exported from
  `mnemostack.graph` and added to its `__all__` for 1.0**), `Triple`,
  `TripleExtractor`. The write/read methods (`add_triple`, `invalidate`,
  `sync_file_links`, `query_triples`, `neighbors`) take an **optional keyword
  `tenant=`** that folds a server-owned `tenant` property into the node key and
  confines the query to it; `stamp_tenant(tenant, only_missing=True)` backfills it
  on a pre-tenant graph. With `tenant=None` the Cypher is the legacy single-tenant
  form (no-op). `mnemostack.graph.store.TENANT_KEY` (`"tenant"`) is the property name.
- **Observability**: `counter`, `histogram`, `timed`, `get_recorder`,
  `set_recorder`, `MetricsRecorder`, `NullRecorder`. `InMemoryRecorder` is used
  by the HTTP server (imported from `mnemostack.observability.recorder`
  directly) and is 🟢 for that purpose; **to be re-exported from
  `mnemostack.observability` (`__all__`) for 1.0**.
- **Config**: `Config` (+ `load` / `save` / `to_dict`), `model_kwargs`.

### 🟡 Experimental

- **Recall pipeline internals**: `Pipeline`, `Stage`, `PipelineContext`,
  `build_full_pipeline`, and the individual stages — especially the
  cognitive-science stages (`GravityDampen`, `HubDampen`, `CuriosityBoost`,
  `InhibitionOfReturn`, `QLearningReranker`, `FreshnessBlend`). Behavior and
  ordering may change as ranking is tuned. Use the `Recaller` / `recall_flow`
  entry points, which are stable, rather than assembling stages yourself.
- **`HyDERetriever`** — exported but not wired into any serving surface.
- **`ScoringReranker` / `RelevanceScorer`** — an alternate reranker path not used
  by the CLI/server/MCP (which use `Reranker`).
- **`synthesize` / `synthesize_async` / `SynthesisResult` / `SynthesisFact`** —
  the entity-synthesis API. Note `synthesize_async` is exported at the top level
  but not from `synthesis.__all__` (an asymmetry to be reconciled for 1.0).
- **State stores** (`StateStore`, `FileStateStore`, `InMemoryStateStore`) — used
  by the stateful pipeline (IoR / Q-learning); the on-disk state format is not
  yet a stability contract.
- **`stable_chunk_id`** — exported (top-level `__all__` and `ingest.__all__`), so
  it's a sanctioned import, but the **id-derivation semantics** it encodes are an
  internal contract (see [migration notes](migration-0.8-to-1.0.md)): the
  function stays, but don't hard-code assumptions about the id format.
- **`dumps`** (`synthesis.__all__`) — the synthesis serialization helper; tied to
  the experimental `synthesize` API above.

### 🔴 Internal

`apply_enrichment`, `prune_stale_chunks`, `_vector_floor_candidates` and any
`_`-prefixed payload key, and anything imported from a submodule but absent from
that submodule's `__all__`.

## Multi-tenancy & authentication

🟢 **Stable, opt-in, additive.** Multi-tenancy is off until you turn it on; a
deployment that never passes `tenant=` or `--auth` behaves exactly as it did
pre-1.0.

**The boundary is server-enforced.** A tenant is a string carried in the
server-owned `tenant_id` payload key. Writes stamp it, reads filter on it, and the
store refuses to cross tenants — a client can never assert a tenant it wasn't
issued. Enforcement has two shapes (see the Vector entry above): `upsert` **raises
`TenantConflictError`** on a conflicting id; the id-targeted owner-guards
(`set_payload`, `invalidate`, `delete_payload_keys`, `delete_points`) **silently
skip** foreign-owned ids. `filter_by_tenant` is a defense-in-depth backstop on the
recall path.

**Who sets the tenant.** In the library you pass `tenant=` explicitly. On the HTTP
and MCP surfaces the tenant is *derived from a service key*, never taken from the
request body:

1. Issue a key: `mnemostack keys add --tenant acme --scopes read,write`
   (plaintext shown once; stored SHA-256-hashed in the keys file).
2. Scopes gate operations: `read` → recall/answer, `write` → invalidate/feedback,
   `admin` implies all. **`feedback` is tenant-partitioned:** its Q-learning /
   inhibition-of-return learning state is keyed by the caller's tenant in the
   shared state store (the Q-table, the IoR log, and its cap are all per-tenant),
   so a `write`-scoped tenant's feedback — and auto-recorded IoR from its
   recalls — only ever move its own ranking, never another tenant's. Recall reads
   the tenant's own partition too. (`tenant=None` uses the unscoped, single-tenant
   state, so a non-auth deployment is unchanged.)
3. Run the surface with auth on:
   - HTTP: `mnemostack serve --auth` — clients present the key as a bearer token;
     `/recall` `/answer` `/feedback` are default-deny, health/status/metrics stay
     public.
   - MCP: `mnemostack mcp-serve --auth --api-key <key>` — one principal per
     process, re-verified per call (revocation is immediate).
4. `--keys-file` / `MNEMOSTACK_KEYS_FILE` selects the store;
   `MNEMOSTACK_AUTH_ENABLED` / `MNEMOSTACK_API_KEY` are the env equivalents of
   `--auth` / `--api-key`.

**Graph tenant-scoping (🟢 stable).** The graph (Memgraph) is now tenant-scoped,
so graph recall works under auth on **both** surfaces — one shared graph, isolated
by a server-owned `tenant` property on every node and edge (the same shape as the
vector store's `tenant_id`):

- **Writes** stamp `tenant` and fold it into the node MERGE key, so two tenants
  writing the same fact get distinct, isolated nodes. A scoped `add_triple`/
  `invalidate`/`sync_file_links` can only ever touch its own tenant's records.
- **Reads** (`GraphStore.query_triples`, `MemgraphRetriever`, and the pipeline's
  `GraphResurrection` stage) confine every match to `tenant` and stamp `tenant_id`
  into the result payload, so graph hits survive the `filter_by_tenant` backstop
  instead of being dropped. Under `mcp-serve --auth` / `serve --auth` a tenant only
  ever resurrects or queries its own subgraph.
- **MCP graph tools** are no longer fail-closed under auth: `mnemostack_graph_query`
  (a structured SPO query) and `mnemostack_graph_add_triple` are tenant-scoped —
  the query is confined to the key's tenant and the write is stamped with it
  (`add_triple` still requires the `write` scope).
- **`tenant=None` stays single-tenant**: on a graph with no `tenant` property,
  every unscoped write/read behaves exactly as pre-1.0 (an existing single-tenant
  graph is untouched and needs no migration). Unscoped writes are additionally
  hardened so they can't corrupt a *migrated* graph: an unscoped `add_triple`
  won't overwrite a tenant-owned edge, and an unscoped `sync_file_links` re-index
  won't delete a tenant file's links (gated on the source node's tenant).
- **The markdown indexer is tenant-aware**: `index-markdown --tenant <id>` (and
  `MarkdownSyncer(tenant=...)`) scope the chunk ids, payloads, and `:File` link
  nodes/edges to the tenant, so a multi-tenant deployment has **no unscoped graph
  write path** — index each tenant's corpus with its own `--tenant`.

Adopt tenancy on an existing graph with `mnemostack tenant-migrate --tenant <id>
--memgraph-uri <bolt>` (stamps nodes + edges; `--dry-run` / `--all` parity with the
vector stamp), or `GraphStore.stamp_tenant(tenant, only_missing=True)` in the
library.

## On-disk / payload contracts

Covered in detail in [migration notes](migration-0.8-to-1.0.md). Summary:

- 🟢 **Stable payload keys**: `text`, `source`, `offset`, `timestamp`,
  `indexed_at`, `index_root`, `invalidated_at`, `valid_from`, `valid_until`,
  `tags`. Frontmatter keys from markdown are your own namespace.
- 🟢 **`tenant_id`** — the multi-tenant partition key. It is **server-owned**:
  set only when a write passes `tenant=`, and the client can never assert or
  change it — a caller `tenant_id` in an `upsert` is overridden (or rejected with
  `TenantConflictError` if the id is another tenant's), and a `set_payload` under a
  tenant restores the server-owned value rather than let the merge change it.
  Absent on single-tenant collections; back-fillable via `VectorStore.stamp_tenant`.
- 🟡 The sliding-window keys (`chunk_window`, `chunk_kind`,
  `chunk_start_offset`, `chunk_end_offset`) and `heading_path`.
- 🔴 Ownership records (`_enrich_keys`, `_md_keys`) and `_`-prefixed keys.
- **Graph schema**: `:Entity` / `:File` node labels, `LINKS_TO` edges, and the
  `valid_until = 'current'` open-ended marker are 🟢 stable contracts. Note two
  `:File` shapes: markdown **link** nodes are keyed by `(name, index_root)` (stable
  — see migration notes for the multi-root implications), while ingest **tag/file**
  nodes are keyed by `{path}` (with `TAGGED` edges) and carry no `index_root`.
  Graph queries and cleanups must handle both. The server-owned **`tenant`**
  property (🟢) partitions the graph: when set it's folded into the node key and
  present on nodes and edges, exactly like the vector `tenant_id`; absent means
  single-tenant (unscoped). A scoped read confines to it and can never see another
  tenant's — or an unscoped — node.

---

## What "1.0" will lock

At 1.0 the 🟢 surfaces above become the compatibility contract. Between now and
then, the remaining churn is expected only in the 🟡 items (pipeline internals,
synthesis, alternate rerankers) and the `__all__` reconciliations noted
above (`make_graph_store`, `InMemoryRecorder`, `synthesize_async`).
