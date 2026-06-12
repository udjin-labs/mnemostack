# mnemostack Architecture

## Overview

**mnemostack** is a memory stack for AI agents — durable, structured, semantically searchable memory with knowledge graph and consolidation lifecycle.

Built around proven patterns: hybrid retrieval (BM25 + vector + graph), RRF fusion, reranker, inference layer. Pluggable embedding providers so you can run local or cloud.

## Target architecture

```
mnemostack/
├── embeddings/       # Pluggable providers: Gemini, Ollama, HuggingFace
├── vector/           # Qdrant operations (indexing, search)
├── graph/            # Memgraph/Neo4j operations (temporal knowledge graph)
├── recall/           # Unified recall pipeline (BM25 + vector + RRF + rerank + answer)
├── consolidation/    # Memory lifecycle (decay, promote, summarize)
├── observability/    # Metrics recorder + Prometheus exposition helpers
├── mcp/              # Model Context Protocol server
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

vec = provider.embed("some text")
assert provider.dimension == len(vec)
```

Critical constraint: **use the same provider/model for indexing and searching**. Mixing dimensions or models breaks similarity scores silently.

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

- Nodes: `Person`, `Project`, `Decision`, `Event`, `Fact`
- Edges: `WORKS_ON`, `DECIDED`, `RELATES_TO`, `CAUSES`, `ENABLES`, `PREVENTS`
- Each edge has `valid_from` and `valid_until`; current facts use the explicit `valid_until="current"` marker, and older `NULL` markers are still read as current for compatibility.

### Recall pipeline (hybrid)

Retrieval and reranking are fail-open: a broken retriever contributes nothing, a failing LLM reranker leaves the pre-rerank order. Query expansion is the exception — it runs before retrieval, and a misconfigured expansion step (`query_expansion=True` without an `expansion_llm`, or a provider error inside `expand_query`) surfaces as an error rather than degrading. Since v0.4.5 those degradations are observable instead of silent: pass a per-call `RecallTrace` to `Recaller.recall(trace=...)` to capture per-retriever ranked lists (with errors and latency), the fused order, the post-rerank order, and stable `degraded` tags (`retriever:<name>:failed`, `reranker:fallback`, `reranker:unavailable`, `temporal:no_parse`). The HTTP and MCP surfaces always expose `degraded` in responses and offer the full trace opt-in via `include_trace`; the shared `apply_rerank_safe` helper implements the reranker's fail-open contract for all entry points.

Since v0.5.0 every entry point (CLI, HTTP, MCP) ranks through one canonical chain, `recall_flow()`: recall with a 3x candidate pool → ranking pipeline → fail-open rerank → top-K cut → vector-floor. The pipeline is the default everywhere; `--raw` (CLI) or `full_pipeline=false` (HTTP) opt back out to plain fused recall.

Query goes through that staged chain:

1. Optional **query expansion** (`Recaller(expansion_llm=...)`) — reformulate the query into multiple variants before retrieval (v0.4.0). Conversational follow-ups can be resolved into standalone questions first with the `rewrite_followup()` helper (the caller supplies the dialog history).
2. **Parallel retrieval**: Qdrant vector, Temporal vector, optional BM25, optional Memgraph. Optional payload `filters=` apply *inside every retriever* (exact match and `gte`/`lte` ranges, native in Qdrant, mirrored in-process for BM25/MCA via `payload_matches`) — the isolation contract for multi-tenant memory: a source that cannot attribute its results to the filtered scope contributes nothing rather than leak.
3. **Reciprocal Rank Fusion (RRF)** — equal-weight by default; optionally weighted (`reciprocal_rank_fusion(weights=...)`) or per-query-shape adaptive (`Recaller(adaptive_weights=True)`).
4. **8-stage ranking pipeline** (default on all surfaces since v0.5.0): ClassifyQuery → ExactTokenRescue → GravityDampen → HubDampen → FreshnessBlend → InhibitionOfReturn → CuriosityBoost → QLearningReranker. The final stage and IoR pull from a `StateStore`; `/feedback` (HTTP, CLI, MCP) drives Q-learning updates.
5. **Graph spreading activation** — optional `GraphResurrection` stage walks 1 hop in Memgraph from RRF seeds (its unattributable additions are dropped under `filters=`).
6. **LLM reranker** for top-K (`rerank_mode`: `relevant_only` subset or `full_reorder`).
7. **Inference layer** (Answer mode) — LLM synthesizes a concise factual answer from top memories, with category-aware prompts, specificity resolver, and `cat_3` inference retry on by default. With `retry_with_expansion=True`, low-confidence answers are retried with an expanded query and a HyDE-style hypothetical before giving up. Localization hooks: `prompt_overrides=`, `question_classifier=`, `abstention_text=`; count/list questions get the two-pass batched extract path (`list_extract_mode`, optional `list_finalize="verbatim"`).

Output:

- Raw mode: top-K snippets with scores and sources.
- Answer mode: short factual answer + confidence + source citations.

### Consolidation / decay

Nightly lifecycle runtime:

1. Ingest new files into vector store
2. Extract facts from recent text → update graph
3. Detect consolidation candidates (repeated patterns, aging items) → summarize into long-term memory
4. Decay sweep — reduce activation scores for unused memories
5. Prune orphan graph nodes
6. Health check

### MCP server

Expose mnemostack tools over Model Context Protocol so LLM clients (Claude Desktop, Cursor, etc.) can call:

- `mnemostack_health()` — config + reachability summary
- `mnemostack_search(query, limit, include_trace, filters)` — hybrid recall through the full ranking chain
- `mnemostack_answer(query, limit, filters)` — concise answer with confidence + source citations
- `mnemostack_feedback(hit_id, signal, query, ...)` — explicit feedback into the stateful pipeline (Q-learning + inhibition-of-return)
- `mnemostack_graph_query(subject, predicate, obj, as_of)` — point-in-time graph query *(only registered when `--memgraph-uri` is set)*
- `mnemostack_graph_add_triple(subject, predicate, obj, valid_from, valid_until)` *(only registered when `--memgraph-uri` is set)*

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
    image: qdrant/qdrant:latest
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
- **Temporal validity.** Facts have `valid_from`/`valid_until` — memory is not just what happened but when it was true.
- **Pluggable embeddings.** No vendor lock-in. User picks provider; system validates dimension consistency.
- **Same provider for write and read.** Indexing and querying must use identical embedding model.
- **Graceful degradation.** If Memgraph is down, retrieval still works with BM25+vector only.
- **Confidence over certainty.** Recall returns scores and sources; caller decides what to trust.

## Status

**Actively developed.** Public API is stable; new functionality lands additively in minor releases. Breaking changes are rare and called out in `CHANGELOG.md`.

- [x] Embedding provider registry (Gemini, Ollama, HuggingFace)
- [x] Vector store wrapper (sync + async)
- [x] Hybrid recall pipeline (Vector + BM25 + Memgraph + Temporal + RRF)
- [x] 8-stage pipeline with stateful learning (Q-learning + inhibition-of-return)
- [x] LLM reranker
- [x] Inference layer (answer mode) with category-aware prompts and inference retry
- [x] Graph client with temporal validity
- [x] Consolidation runtime
- [x] CLI
- [x] MCP server (6 tools)
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
