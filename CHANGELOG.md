# Changelog

All notable changes to mnemostack will be documented here. Format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0a8] - 2026-04-17

### Fixed

- **Critical indent bug in `Recaller.recall()`**: `return results` was inside the fusion `for` loop, causing recall to return only the first result regardless of `limit`. LoCoMo benchmark correctness on a 60 QA sample went from 28.3% to 56.7% after the fix (combined 38.3% → 73.3%).

## [0.1.0a7] - 2026-04-17

### Added — full 8-stage pipeline ported from the reference enhanced-recall implementation

- `mnemostack.recall.pipeline` subpackage with composable stage architecture
  - `Pipeline` orchestrator (sequence of Stage instances)
  - `PipelineContext` shared state (query, query_type, tokens, extras)
  - `Stage` base class for custom stages
- `StateStore` abstraction — plug in persistence for stateful stages
  - `InMemoryStateStore` (thread-safe, lost on restart)
  - `FileStateStore` (atomic JSON writes)
- 8 built-in stages:
  - `ClassifyQuery` — categorize queries (person/project/decision/event/technical/general)
  - `ExactTokenRescue` — boost exact-match sources for infra queries (IPs, ports, versions)
  - `GravityDampen` — penalize results where query terms are absent in content
  - `HubDampen` — penalize top-degree graph hubs (prevents domination by overconnected nodes)
  - `FreshnessBlend` — blend similarity with recency + echo penalty for sub-10min results
  - `InhibitionOfReturn` — penalize recently-recalled memories (biological diversity)
  - `CuriosityBoost` — surface old, rarely-recalled memories
  - `QLearningReranker` — learn which retrieval sources perform best per query type
- Presets: `build_full_pipeline()` and `build_stateless_pipeline()` for common configurations

### Tests
- 25 pipeline tests (stages + orchestrator + presets + integration)
- Total: 160 passing

## [0.1.0a6] - 2026-04-17

### Added
- `mnemostack.observability` — metrics infrastructure (counters, histograms, pluggable recorder)
  - `NullRecorder` (default, zero overhead), `LoggingRecorder`, `InMemoryRecorder`
  - Integration with Recaller + AnswerGenerator (latency + throughput metrics)
  - Protocol-based — plug in Prometheus / OpenTelemetry / StatsD via `set_recorder()`
- `VectorStore.scroll()` and `iter_ids()` — memory-efficient iteration over large collections
  - Supports optional payload filters
  - Uses Qdrant native scroll API with cursor-based pagination

### Tests
- 10 observability tests (counters, histograms, decorators, faulty-recorder safety)
- 4 new vector tests (scroll, iter_ids, filters on scroll)
- Total: 135 passing

## [0.1.0a5] - 2026-04-17

### Added
- Batch embedding via Gemini `:batchEmbedContents` endpoint (3-5x speedup on corpora)
- Parallel batch embedding for Ollama (thread pool)
- Smarter chunkers: `CharChunker`, `ParagraphChunker`, `MarkdownChunker` (header-aware, code-block-safe)
- `AsyncVectorStore` — asyncio variant for high-concurrency servers
- Structured logging under `mnemostack.*` namespace with `configure_logging()` helper
- Replaced silent failures with `logger.warning`/`logger.error` in Gemini and Reranker
- CHANGELOG.md, CONTRIBUTING.md

### Tests
- 9 embedding tests (+4 batch)
- 14 chunking tests
- 7 logging tests
- 5 async vector tests
- Total: 128 passing

## [0.1.0a4] - 2026-04-17

### Added
- `mnemostack.config` — YAML config loader with env var overrides and `Config.load()` / `Config.save()`
- `mnemostack.graph.TripleExtractor` — LLM-based extraction of knowledge graph triples from free-form text
- CLI commands: `mnemostack init` (create example config), `mnemostack config` (show resolved config)
- `examples/docker-compose.yml` — minimal stack with Qdrant + Memgraph
- `examples/quickstart.md` — end-to-end user guide

## [0.1.0a3] - 2026-04-17

### Added
- `mnemostack.mcp.server` — FastMCP server exposing 5 tools: health, search, answer, graph_query, graph_add_triple
- CLI command `mnemostack mcp-serve` for running stdio MCP server
- `mnemostack.graph.GraphStore` — Memgraph/Neo4j wrapper with temporal validity (valid_from/valid_until)
- `mnemostack.consolidation` — phase orchestrator for memory lifecycle (runtime + built-in health phases)
- `mnemostack.recall.Reranker` — LLM-based rerank of retrieval results
- Point-in-time graph queries via `as_of` parameter

## [0.1.0a2] - 2026-04-17

### Fixed
- Dependency version upper bounds (`httpx<1`, `qdrant-client<2`, etc.) to prevent pip from pulling pre-release `httpx 1.0.dev` with `--pre` flag

## [0.1.0a1] - 2026-04-17 (yanked)

First published release under the name `mnemostack` (renamed from `memvault` due to PyPI similarity with `mem-vault`). Scaffold + embedding + vector + recall + answer.

### Yanked
Dependency constraints too loose; `pip install --pre mnemostack` pulled broken httpx variant. Fixed in 0.1.0a2.
