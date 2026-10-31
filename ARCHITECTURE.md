# mnemostack Architecture

## Overview

**mnemostack** is an independent memory and retrieval layer for agents and applications: ingestion, configurable hybrid recall, temporal validity and source evidence, plus service controls for separate consumers. Python SDK, CLI, HTTP, and MCP expose these capabilities at different levels of composition. See the [memory model](docs/memory-model.md) for their roles and boundaries.

Built around proven patterns: hybrid retrieval (BM25 + vector + graph), RRF fusion, reranker, inference layer. Pluggable embedding providers so you can run local or cloud.

## Module map

```
mnemostack/
├── embeddings/       # Pluggable providers: Gemini, Ollama, HuggingFace
├── vector/           # Qdrant operations (indexing, search)
├── graph/            # Memgraph/Neo4j operations (temporal knowledge graph)
├── recall/           # Unified recall pipeline (BM25 + vector + RRF + rerank + answer)
├── consolidation/    # Phase orchestrator; health checks and custom phases
├── observability/    # Metrics recorder + Prometheus exposition helpers
├── mcp/              # Model Context Protocol server
├── server.py         # HTTP memory service and tenant controls
└── cli.py            # `mnemostack` command-line tool
```

## Core modules

### Embeddings (pluggable)

Abstract base class + provider registry:

```python
from mnemostack.embeddings import get_provider

provider = get_provider("gemini", api_key="...")        # Cloud (best quality)
provider = get_provider("ollama", host="http://localhost:11434")  # Local
provider = get_provider("huggingface", model="BAAI/bge-large-en-v1.5")  # Local GPU

query_vec = provider.embed_query("some question")   # retrieval query role
doc_vec = provider.embed_document("some text")      # document/chunk role
vec = provider.embed("some text")                   # neutral primitive
assert provider.dimension == len(vec)
```

`embed`/`embed_batch` are the neutral primitives. The role methods
(`embed_query`, `embed_document` and their batch forms) apply the model
family's *embedding profile* — declarative query/document input conventions
such as E5's `query: `/`passage: ` prefixes or Qwen3-Embedding's query
instruction — exactly once before delegating to the primitives. For
symmetric models the profile is the identity and the role methods are
bit-identical to the primitives. Transforms touch only inference input:
stored text, chunk ids, BM25 input and citations always keep the original
document text.

Critical constraint: **use the same provider/model for indexing and searching**. Mixing dimensions or models breaks similarity scores silently. Indexed points are stamped with a *document-space fingerprint* (provider + model + document transform + dimension + provider inference knobs; profile name/version are metadata and never hashed), and the index commands refuse a collection whose stored fingerprint differs — dimension alone can't catch two different models sharing a size.

### Vector store

Qdrant wrapper. Collection holds chunked documents with payload:

```json
{
  "text": "...",
  "source": "notes/2024-01-15.md",
  "offset": 120,
  "timestamp": "2024-01-15T10:30:00Z",
  "index_root": "/abs/indexed/root",
  "indexed_at": "2026-06-12T10:30:00Z"
}
```

Arbitrary additional fields arrive via `IngestItem.metadata` or the ingest-time enrichment hook (`Ingestor(enrich=callable)` / `mnemostack index --enrich pkg.mod:func`); they are filterable in recall (`filters=`) and projectable into the answer prompt (`context_fields=`). Chunk identity is the deterministic `stable_chunk_id(source, offset, text)`; lifecycle is handled by opt-in pruning (`index --prune`, scoped to the indexing root) and payload-only refresh (`index --refresh-payloads`, no re-embedding).

Chunking depends on the ingest surface. The CLI uses deterministic character windows by default; the Python chunking package also provides fixed-size, paragraph, markdown, and message-pair chunkers for applications that need richer boundaries. The `vector.window_size` config (default `1`) carries adjacent-turn context inside each message chunk — `window_size=3` was worth **+5.8pp strict / +4.1pp combined** on LoCoMo in v0.4.0. Timestamp payloads are indexed as datetime for temporal queries.

### Knowledge graph

Memgraph (Cypher-compatible) with temporal validity:

- By default, `Entity` nodes hold structured subject/object facts; `File` nodes support file/tag and markdown-link relationships.
- Triple predicates become validated relationship types (for example `WORKS_ON`); markdown links use `LINKS_TO`. Authenticated operations scope graph data by tenant.
- Each edge has `valid_from` and `valid_until`; current facts use the explicit `valid_until="current"` marker, and older `NULL` markers are still read as current for compatibility.

### Recall pipeline (hybrid)

Retrieval and reranking are fail-open: a broken retriever contributes nothing, a failing LLM reranker leaves the pre-rerank order. Query expansion is the exception — it runs before retrieval, and a misconfigured expansion step (`query_expansion=True` without an `expansion_llm`, or a provider error inside `expand_query`) surfaces as an error rather than degrading. Since v0.4.5 those degradations are observable instead of silent: pass a per-call `RecallTrace` to `Recaller.recall(trace=...)` to capture per-retriever ranked lists (with errors and latency), the fused order, the post-rerank order, and stable tags split by meaning: `notes` is the authoritative list of routine stage-did-not-apply signals (`temporal:no_parse` on any query without a parseable date, `<arm>:no_tokens` when a lexical arm's gate finds no usable tokens), while `degraded` carries components that actually fell back (`retriever:<name>:failed`, `reranker:fallback`, `reranker:unavailable`) plus — deprecated until the next major — a back-compat duplicate of the routine tags; a `degraded` entry absent from `notes` is a real fault. The HTTP and MCP surfaces always expose `degraded` and `notes` in responses and offer the full trace opt-in via `include_trace`; the shared `apply_rerank_safe` helper implements the reranker's fail-open contract for all entry points.

Since v0.5.0 every entry point (CLI, HTTP, MCP) ranks through one canonical chain, `recall_flow()`: recall with a 3x candidate pool → ranking pipeline → fail-open rerank → top-K cut → vector-floor → optional token-budget cut (keep the ranked prefix whose total text tokens fit `token_budget`; a hard cap, never overshot). The serving paths build the pipeline by default; `--raw` (CLI) or `full_pipeline=false` (HTTP) opt back out to plain fused recall. SDK calls compose these parts explicitly: `Recaller.recall()` does not build the full pipeline, and `recall_flow()` defaults to no pipeline or reranker unless supplied.

Query goes through that staged chain:

1. Optional **query expansion** (`Recaller(expansion_llm=...)`) — reformulate the query into multiple variants before retrieval (v0.4.0). Conversational follow-ups can be resolved into standalone questions first with the `rewrite_followup()` helper (the caller supplies the dialog history).
2. **Parallel retrieval**: Qdrant vector, Temporal vector, optional BM25, optional Memgraph. Optional payload `filters=` apply *inside every retriever* (exact match and `gte`/`lte` ranges, native in Qdrant, mirrored in-process for BM25/MCA via `payload_matches`) — the retrieval-level isolation contract for multi-tenant memory: a source that cannot attribute its results to the filtered scope contributes nothing rather than leak. (Filters are caller-supplied, so they are an isolation *mechanism*, not an authorization boundary; for a server-enforced tenant boundary run `serve --auth` / `mcp-serve --auth`, which resolves the tenant from a service key.)
3. **Reciprocal Rank Fusion (RRF)** — equal-weight by default; optionally weighted (`reciprocal_rank_fusion(weights=...)`) or per-query-shape adaptive (`Recaller(adaptive_weights=True)`).
4. **8-stage ranking pipeline** (default on CLI search/answer and HTTP/MCP serving paths; explicit SDK composition): ClassifyQuery → ExactTokenRescue → GravityDampen → HubDampen → FreshnessBlend → InhibitionOfReturn → CuriosityBoost → QLearningReranker. The final stage and IoR pull from a `StateStore`; `/feedback` (HTTP, CLI, MCP) drives Q-learning updates.
5. **Graph spreading activation** — optional `GraphResurrection` stage walks 1 hop in Memgraph from RRF seeds (its unattributable additions are dropped under `filters=`).
6. **LLM reranker** for top-K (`rerank_mode`: `relevant_only` subset or `full_reorder`).
7. **Inference layer** (Answer mode) — LLM synthesizes a concise factual answer from top memories, with category-aware prompts, specificity resolver, and `cat_3` inference retry on by default. With `retry_with_expansion=True`, low-confidence answers are retried with an expanded query and a HyDE-style hypothetical before giving up. Localization hooks: `prompt_overrides=`, `question_classifier=`, `abstention_text=`; count/list questions get the two-pass batched extract path (`list_extract_mode`, optional `list_finalize="verbatim"`).

Output:

- Raw mode: top-K snippets with scores and sources.
- Answer mode: short factual answer + confidence + source citations.

### Consolidation / decay

`consolidation.Runtime` is a phase orchestrator: it executes registered `Phase`
objects in order, records timing and outcomes, and supports `dry_run` and
`stop_on_error`. Built-in phases are `EmbeddingHealthPhase`, `VectorHealthPhase`,
and `GraphHealthPhase`; `CallablePhase` wraps application-provided work.

A scheduled sequence that ingests files, extracts facts, resolves contradictions,
summarizes, promotes, or forgets memories requires your own phases and scheduler.
Those tasks are not implemented as a turnkey nightly workflow. The separate
experimental synthesis API can build entity profiles; it is not automatically
run by consolidation.

Memory lifecycle operations already exist outside this orchestrator: idempotent
ingest, root-scoped chunk pruning, payload refresh, stale-fact invalidation,
validity views, hard deletion, and tenant export/offboarding. See the
[memory model](docs/memory-model.md#maintain-facts-and-their-evidence).

### MCP server

Expose mnemostack tools over Model Context Protocol so LLM clients (Claude Desktop, Cursor, etc.) can call:

- `mnemostack_health()` — config + reachability summary
- `mnemostack_search(query, limit, include_trace, filters)` — hybrid recall through the full ranking chain
- `mnemostack_answer(query, limit, filters)` — concise answer with confidence + source citations
- `mnemostack_remember(text, source, ...)` — explicit ingestion with server-side embedding
- `mnemostack_invalidate(ids, ...)` — mark memories stale without re-embedding
- `mnemostack_resolve(chunk_id)` — verify a citation against an allowed local source root
- `mnemostack_feedback(hit_id, signal, query, ...)` — explicit feedback into the stateful pipeline (Q-learning + inhibition-of-return)
- `mnemostack_graph_query(subject, predicate, obj, as_of)` — point-in-time graph query *(only registered when `--memgraph-uri` is set)*
- `mnemostack_graph_add_triple(subject, predicate, obj, valid_from, valid_until)` *(only registered when `--memgraph-uri` is set)*

Tool signatures above are abbreviated; optional arguments and auth requirements
are defined by the server schema. Graph tools are conditional. MCP is one
interface to the backend, alongside CLI, HTTP, and Python. See [MCP](docs/mcp.md)
and [source verification](docs/recipes.md#auditing-answer-citations-against-their-sources).

## Configuration

```yaml
embedding:
  provider: gemini        # or ollama, huggingface
  model: gemini-embedding-001

vector:
  host: http://localhost:6333
  collection: mnemostack
  chunk_size: 600
  overlap: 100
  window_size: 1            # v0.4.0+: sliding-window context across neighbouring turns

graph:
  uri: bolt://localhost:7687
  timeout: 5.0
  health_timeout: 1.0

recall:
  rrf_k: 60
  top_k: 10
  confidence_threshold: 0.5
  bm25_paths: []
  vector_floor: 0             # append missing top-N raw-vector candidates after fusion/rerank
  rerank_mode: relevant_only  # or full_reorder

llm:
  provider: gemini
  model: null
```

## Deployment

```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:v1.20.0  # match your qdrant-client: same major, minor within 1
    ports: ["6333:6333"]
  memgraph:
    image: memgraph/memgraph:latest
    ports: ["7687:7687"]
  mnemostack:
    build: .
    depends_on: [qdrant, memgraph]
```

## Design principles

- **Hybrid over pure vector.** BM25 finds exact tokens (IPs, versions, error strings) that vector search misses.
- **Temporal validity.** Facts have `valid_from`/`valid_until` — memory is not just what happened but when it was true. On both sides now: the graph closes a fact via `GraphStore.invalidate` (`valid_until`), and the vector store mirrors it via `VectorStore.invalidate` (bi-temporal `invalidated_at` system-time + `valid_until` world-time). Recall hides invalidated facts by default; `include_invalidated` / `as_of` opt back in or reconstruct a past instant.
- **Pluggable embeddings.** No vendor lock-in. User picks provider; system validates dimension consistency.
- **Same provider for write and read.** Indexing and querying must use identical embedding model.
- **Graceful degradation.** If Memgraph is down, retrieval still works with BM25+vector only.
- **Evidence and observable retrieval.** Recall returns source-bearing results and optional traces. Response order is authoritative; scores are not calibrated confidence. Citation resolution checks current source support separately.

## Status

**Actively developed.** Public API is stable; new functionality lands additively in minor releases. Breaking changes are rare and called out in `CHANGELOG.md`.

- [x] Embedding provider registry (Gemini, Ollama, HuggingFace)
- [x] Vector store wrapper (sync + async)
- [x] Hybrid recall pipeline (Vector + BM25 + Memgraph + Temporal + RRF)
- [x] 8-stage pipeline with stateful learning (Q-learning + inhibition-of-return)
- [x] LLM reranker
- [x] Inference layer (answer mode) with category-aware prompts and inference retry
- [x] Graph client with temporal validity
- [x] Consolidation phase orchestrator (health phases and application-supplied work)
- [x] CLI
- [x] MCP server (search, answer, remember, invalidate, resolve, feedback, health; graph tools when configured)
- [x] HTTP server with `/metrics` (Prometheus)
- [x] Streaming `Ingestor` with lazy iterator
- [x] Sliding-window message chunking (`window_size`) — v0.4.0
- [x] Query expansion + smart retry on low-confidence answers — v0.4.0
- [x] Knowledge synthesis (`synthesize()`) — v0.3.0
- [x] Canonical `recall_flow()` shared by CLI/HTTP/MCP — v0.5.0
- [x] Chunk lifecycle: `index --prune` (root-scoped) + `--refresh-payloads` — v0.5.0+
- [x] Recall `filters=` with adversarially-tested tenant isolation — post-0.5.0
- [x] Ingest-time payload enrichment (`Ingestor(enrich=...)`) + `context_fields` projection — post-0.5.0
- [x] Ollama `think` control (off by default) + generation options passthrough — post-0.5.0
- [x] Follow-up question rewriting (`rewrite_followup`) — post-0.5.0
