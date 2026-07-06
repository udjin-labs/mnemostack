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
`feedback`, `serve`, `mcp-serve`, `keys`, `tenant-migrate`, `init`, `config`.

- `tenant-migrate --tenant <id>` stamps existing points into a tenant (idempotent;
  `--all` to force-relabel, `--dry-run` to preview) — the operator path to adopting
  tenancy on a pre-tenant collection; `VectorStore.stamp_tenant` is its library twin.
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
`recall.{bm25_paths,vector_floor,rerank_mode,token_budget}`.

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
exist. Their shapes are otherwise stable.

**Under `mcp-serve --auth` (🟢 stable).** The stdio transport binds one client per
process, so the server runs *as* a single principal: the key (`--api-key` /
`MNEMOSTACK_API_KEY`) resolves to a tenant + scopes at boot (and boot-fails loud
without a valid key). Every tool call re-verifies the key, so revocation takes
effect immediately mid-session. `search`/`answer` require `read`,
`invalidate`/`feedback` require `write`, and the tenant is threaded into recall,
answer, and invalidate. The graph tools **fail closed** under an authenticated
tenant (the shared graph isn't tenant-scoped yet), and the recall pipeline drops
its graph-resurrection stage for the same reason. `mnemostack_health` stays
public. Without `--auth` the tools behave exactly as listed above.

## Python library

### 🟢 Stable

- **Ingest**: `Ingestor` (constructor + `ingest` / `ingest_one` / `ingest_async`
  / `ingest_one_async` / `stream`), `IngestItem`, `IngestStats`.
- **Vector**: `VectorStore`, `AsyncVectorStore`, `Hit`, `DimensionMismatchError`,
  `TenantConflictError`, `TENANT_ID_KEY` — the documented methods (`upsert`,
  `search`, `count`, `invalidate`, `ensure_collection`, `collection_exists`,
  `set_payload`, `scroll`). Note the async surface is a **subset** of sync (no
  `scroll` / `iter_ids` / `delete_points` / `index_payload_field`) — parity is
  planned but not guaranteed yet, so treat async-only-missing methods as 🟡. The
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
  `recall_flow_async`, `Recaller.recall` / `recall_async`, and
  `AnswerGenerator.generate` (+ async) all take an **optional keyword `tenant=`**
  (additive, default `None`) that scopes retrieval to that tenant and applies
  `filter_by_tenant` as a backstop.
- **Auth (multi-tenant service keys)**: `mnemostack.auth` — `Principal`
  (`tenant`, `scopes`, `.can(scope)`), `KeyStore` (Protocol: `.verify(key) ->
  Principal | None`), `FileKeyStore` (`.verify` / `.issue` / `.revoke` /
  `.list_keys`; SHA-256-hashed JSON store), `SCOPES` (`{"read", "write",
  "admin"}`; `admin` implies the others), `KeyStoreError`, `hash_key`,
  `default_keys_path`. This is the credential surface behind `serve --auth` /
  `mcp-serve --auth`; keys are stored hashed and the plaintext is shown once.
- **Markdown**: `collect_markdown`, `MarkdownChunk`, `LinkEdge`,
  `MarkdownCollection`, `parse_frontmatter`, `extract_links`, `MarkdownSyncer`.
- **Graph**: `GraphStore` (documented methods), `make_graph_store` (the
  constructor used everywhere — currently only importable as
  `mnemostack.graph.factory.make_graph_store`; **to be re-exported from
  `mnemostack.graph` and added to its `__all__` for 1.0**), `Triple`,
  `TripleExtractor`.
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
   `admin` implies all. **Caveat — `feedback` is access-gated but not
   tenant-partitioned:** its Q-learning / inhibition-of-return learning state is
   process-global, so a `write`-scoped tenant's feedback can shift ranking state
   shared with other tenants on the same process. Until per-tenant state lands
   (planned follow-up), don't expose `feedback` across mutually-distrusting tenants
   on one process — run a process per trust boundary, or leave feedback to a
   trusted operator.
3. Run the surface with auth on:
   - HTTP: `mnemostack serve --auth` — clients present the key as a bearer token;
     `/recall` `/answer` `/feedback` are default-deny, health/status/metrics stay
     public.
   - MCP: `mnemostack mcp-serve --auth --api-key <key>` — one principal per
     process, re-verified per call (revocation is immediate).
4. `--keys-file` / `MNEMOSTACK_KEYS_FILE` selects the store;
   `MNEMOSTACK_AUTH_ENABLED` / `MNEMOSTACK_API_KEY` are the env equivalents of
   `--auth` / `--api-key`.

**Current limitation (graph).** The graph (Memgraph) is **not tenant-scoped yet**,
and the two surfaces guard it differently:

- **MCP** (`mcp-serve --auth`): the graph tools fail closed, and the recall
  pipeline is built with **no graph** under auth, so its graph-resurrection stage
  never runs.
- **HTTP** (`serve --auth`): the pipeline is built with the configured graph
  **regardless of auth**, so graph resurrection still opens and queries the shared
  Memgraph. No tenant reads another's graph nodes — the `filter_by_tenant` backstop
  drops the (tenant-less) graph hits before they reach the caller — but the graph
  is still contacted per recall. So under `serve --auth`, a slow or blackholed
  Memgraph still adds latency; it is **not** skipped the way MCP skips it.

Either way no tenant can read another's graph nodes. Graph tenant scoping is
planned follow-up work; until it lands, keep per-tenant graphs in separate
databases (or, for HTTP, run with `--memgraph-uri ""`) if you need isolation
without the shared-graph query.

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
  Graph queries and cleanups must handle both.

---

## What "1.0" will lock

At 1.0 the 🟢 surfaces above become the compatibility contract. Between now and
then, the remaining churn is expected only in the 🟡 items (pipeline internals,
synthesis, alternate rerankers) and the `__all__` reconciliations noted
above (`make_graph_store`, `InMemoryRecorder`, `synthesize_async`).
