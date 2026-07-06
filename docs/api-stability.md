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

## CLI

🟢 **Stable** commands and their documented flags / exit codes:
`health`, `doctor`, `search`, `answer`, `index`, `index-markdown`, `invalidate`,
`feedback`, `serve`, `mcp-serve`, `init`, `config`.

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

## Python library

### 🟢 Stable

- **Ingest**: `Ingestor` (constructor + `ingest` / `ingest_one` / `ingest_async`
  / `ingest_one_async` / `stream`), `IngestItem`, `IngestStats`.
- **Vector**: `VectorStore`, `AsyncVectorStore`, `Hit`, `DimensionMismatchError`
  — the documented methods (`upsert`, `search`, `count`, `invalidate`,
  `ensure_collection`, `collection_exists`, `set_payload`, `scroll`). Note the
  async surface is a **subset** of sync (no `scroll` / `iter_ids` /
  `delete_points` / `index_payload_field`) — parity is planned but not
  guaranteed yet, so treat async-only-missing methods as 🟡.
- **Embeddings**: `EmbeddingProvider` (incl. `health_check`), `get_provider`,
  `list_providers`, `register_provider`.
- **LLM**: `LLMProvider` (incl. `health_check`), `LLMResponse`, `get_llm`,
  `list_llms`, `register_llm`.
- **Recall (core)**: `Recaller` (`recall` / `recall_async`), `recall_flow`,
  `recall_flow_async`, `RecallResult`, `AnswerGenerator`, `Answer`,
  `Reranker`, `RERANK_MODES`, `BM25`, `BM25Doc`, `reciprocal_rank_fusion`,
  `build_bm25_docs`, the validity helpers (`filter_by_validity`, `is_current`,
  `valid_at`) and token helpers (`estimate_tokens`, `apply_token_budget`,
  `sum_tokens`, `TokenCounter`), and the trace helpers (`RecallTrace`,
  `apply_rerank_safe`).
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

## On-disk / payload contracts

Covered in detail in [migration notes](migration-0.8-to-1.0.md). Summary:

- 🟢 **Stable payload keys**: `text`, `source`, `offset`, `timestamp`,
  `indexed_at`, `index_root`, `invalidated_at`, `valid_from`, `valid_until`,
  `tags`. Frontmatter keys from markdown are your own namespace.
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
