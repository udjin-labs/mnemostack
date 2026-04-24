# Changelog

All notable changes to mnemostack will be documented here. Format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0a1] - 2026-04-24

### Fixed

- **`TemporalRetriever` silently returned zero hits** (PR #1 by @perlowja). Two compounding bugs hidden behind a broad `except Exception: return []`: the retriever emitted a nested qdrant-raw filter shape that `VectorStore._build_filter` didn't understand, and `_build_filter` always built a numeric `Range` even for ISO datetime strings, which pydantic rejected. Both paths now use a dispatch-by-type filter builder, and the fallback logs a `WARNING` so future silent-zero cases are observable. 7 new regression tests in `tests/test_temporal_retriever.py`. This was a real bug affecting any install that relied on the temporal recall path — users on `0.1.0a14` should upgrade.

### Added

- **Progressive Tiers API** (`mnemostack search --tier {1,2,3}`, `mnemostack answer --tier {1,2,3}`). Optional output budgets that let agents pay only for the detail they actually need:
  - **Tier 1** (~50 tokens): list view only — score, id, and source labels. Use when the agent just needs to know *whether* anything relevant exists in memory.
  - **Tier 2** (~200 tokens): short (~40 char) snippets around hits. Default-useful recall for most triage flows.
  - **Tier 3** (~500 tokens): fuller 200-char previews and up to 10 results. Use when the agent actually needs to read memory.
  - Omitting `--tier` keeps the existing full-output behavior (backward compatible). Covered by 9 unit tests in `tests/test_cli_tiers.py`.
- **MCP integration guides** for Claude Desktop / Claude Code, Cursor, and OpenClaw under `integrations/`. Each is self-contained (install / verify / uninstall / troubleshooting). The OpenClaw guide explicitly documents coexistence with the host's native memory tools — MCP is an out-of-band channel for sub-agents and external hosts, not a replacement for in-host recall.

### Docs

- Progressive Tiers examples added to `README.md` and `examples/quickstart.md`, including the `answer --tier 1` shortcut that drops the `SOURCES:` block when only the answer text is wanted.

## [0.1.0a14] - 2026-04-19

### Added

- **Weighted RRF fusion** (`reciprocal_rank_fusion(weights=[...])`). Classical equal-weight RRF stays the default; callers can now lift sources they trust more for a given workload (for example BM25 and graph on exact-token queries).
- **Adaptive per-query-shape weights in `Recaller`** (`adaptive_weights=True`, opt-in). Light regex+keyword detection picks a profile per query: exact-token queries (IPs, ports, version strings, API / UUID markers) lift BM25 and the graph retriever; person queries lift the graph retriever; temporal queries (`when`, `когда`, `yesterday` …) lift the temporal retriever; general queries keep classical equal weights. Measured on our real production corpus (10 needle probes): recall@1 went 50% → 60% with recall@5 unchanged at 90%. Static `retriever_weights={...}` always wins over adaptive when both are set.
- **`HyDERetriever`** (opt-in, not in the default `Recaller`). Generates a hypothetical answer via an LLM, embeds that, and searches vectors for memories similar to the synthesised answer. Ships with measured limitations in the docstring — on dialogue-backed memory (our LoCoMo smoke) it gave +1 correct on the hardest cat_3 reasoning sample (14.3% → 21.4%) at the cost of one extra LLM roundtrip per query. Ship it when your query↔answer vocabulary gap is large (structured docs, code, schemas); skip it for general dialogue memory. 5 unit tests included.

## [0.1.0a13] - 2026-04-18

### Added

- **Streaming `Ingestor` API** (`mnemostack.ingest`): batched, idempotent, LRU-cached streaming ingest from arbitrary Python code. `IngestItem` / `IngestStats` / `stable_chunk_id` are public.
- **`mnemostack serve` HTTP API** (opt-in via `pip install 'mnemostack[server]'`): `/recall`, `/answer`, `/health`, `/metrics`, `/docs` FastAPI endpoints with pydantic schemas. Docker-compose example runs the server on port 8000 by default.
- **`/metrics` Prometheus endpoint**: counters and summary histograms in standard text exposition format, no extra dependency.
- **`Recaller.recall_async`** plus parallel retriever dispatch: retrievers run concurrently in a thread pool, the async wrapper lets HTTP endpoints yield the event loop while embedding / Qdrant / Memgraph work happens. Verified with five concurrent recalls completing in roughly one single-recall wall-clock.
- **`MemgraphRetriever` probes by `telegram_id`, handle, and `name_lower`**: numeric queries (>=6 digits) resolve to the canonical Person if `n.telegram_id` is set; non-ASCII names now match correctly via the precomputed `name_lower` property (Memgraph's `toLower()` lower-cases ASCII only).
- **Reproducible LoCoMo harness** in-tree: `benchmarks/download_locomo.sh` + `benchmarks/run_locomo.sh`, results land under `benchmarks/results/`.
- **Synthetic long-horizon benchmark** (`benchmarks/synthetic_longhorizon.py`): generates configurable-size corpora with planted needles and measures `recall@K` and MRR.
- Community health files: `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue and PR templates.
- `Dockerfile` and an updated `examples/docker-compose.yml` that brings up Qdrant + Memgraph + mnemostack as a working stack.

### Changed

- **`mnemostack index` is now idempotent**: deterministic UUID-5 chunk ids derived from `(source, offset, content)`. Re-runs skip unchanged chunks (no duplicates, no wasted embedding calls).
- `/health` endpoint uses a ping-level Qdrant check (`get_collections`) so fresh deployments without any ingested points are reported healthy.
- Server translation layer now reads `RecallResult.payload` / `.sources` correctly — earlier build returned `source=null` in production despite mocks passing.

### Fixed

- `Reranker` composite-id parser (paths with `/` or `:`): ids are preserved intact, lookup falls back to prefix/substring so LLM outputs that truncate ids still resolve.
- Pyproject URLs now point at the `udjin-labs` GitHub org.
- README: full per-section rewrite with Quick start, 30-second Docker try-it-out, honest LoCoMo benchmark table, integration walkthrough for markdown-backed assistants, and environment variable reference.

### Internal

- 207 passing tests (up from 183 at `0.1.0a11.post1`), including new Ingestor, async Recaller, and server contract suites.

## [0.1.0a12] - 2026-04-17

### Documentation

- README: added "Who is this for" and "How it works in one paragraph" sections for new users.
- README: added `Environment` table listing `GEMINI_API_KEY`, `OLLAMA_HOST`, `MNEMOSTACK_COLLECTION`, `MNEMOSTACK_QDRANT_URL`, `MNEMOSTACK_GRAPH_URI`.
- README: documented `RecallResult` fields returned by `Recaller.recall()`.
- README: added full-stack Python API example (4 retrievers + 8-stage pipeline + LLM reranker) — the exact configuration used for the published LoCoMo benchmark.
- README: added "Building a BM25 corpus" walkthrough so the full-stack example is self-contained.
- README: added "Pipeline state" and "Graceful degradation" subsections describing `FileStateStore` and failure behaviour across retrievers and reranker.
- Features list rewritten to reflect the current 4-source retrieval architecture, 8 named pipeline stages, `MessagePairChunker`, and `QueryExpander`.
- Benchmark section on README now ships in the PyPI page (previous `a11` release had the old README).

### Internal

- No behavioural changes. Functionally identical to `0.1.0a11` / `0.1.0a11.post1`; this release exists so the documentation on PyPI matches the code.

## [0.1.0a11.post1] - 2026-04-17

### Documentation

- README: added full LoCoMo benchmark table for this build (1986 QA, 66.4% correct / 79.2% combined), per-category breakdown, and an honest-numbers disclaimer about the difference between full-benchmark aggregate and sub-category cherry-picking reported by some vendors.
- No code changes; functional behaviour is identical to `0.1.0a11`.

## [0.1.0a11] - 2026-04-17

### Parity with legacy workspace stack

mnemostack reached parity with the reference `workspace/scripts/enhanced-recall.py` on a head-to-head benchmark over 12 real workspace queries (clean state for both, top-5 keyword coverage).

Scoreboard:
- Before today: `LEGACY 4 / MNEMOSTACK 0 / 8 ties` (28 vs 21 keywords)
- After a11:    `LEGACY 0 / MNEMOSTACK 2 / 10 ties` (28 vs 30 keywords)

The gap was a packaging miss, not a missing idea. This release ports the remaining pieces from the legacy pipeline and makes them available as first-class modules.

### New

- **Retriever abstraction** (`mnemostack.recall.retrievers`) with four built-in retrievers: `VectorRetriever`, `BM25Retriever`, `MemgraphRetriever`, `TemporalRetriever`. `Recaller` now accepts `retrievers=[...]` and fuses N ranked lists via RRF, matching the legacy architecture where Memgraph and temporal search are first-class sources instead of post-ranking stages.
- **`GraphResurrection` pipeline stage** (`mnemostack.recall.pipeline.resurrection`) — spreading activation via Memgraph 1-hop walk, scored by seed overlap, capped at 0.30.
- **`benchmarks/locomo_single.py`** — single-variant LoCoMo runner (no A/B compare), so full 10-sample runs no longer double the cost.

### Improved

- **Pipeline stage order** now mirrors legacy: `ClassifyQuery → ExactTokenRescue → GravityDampen → HubDampen → QLearningReranker → CuriosityBoost → FreshnessBlend → InhibitionOfReturn → GraphResurrection`. Q-learning runs before Freshness so cold-state boosts land on raw RRF scores.
- **`QLearningReranker`**: adds UCB1 exploration bonus for under-sampled sources (default `min_samples=10`, `exploration_bonus=0.1`) and switches to multiplicative blend (`(1-w)*score + w*avg_q`) so unseen sources get a measurable lift at defaults. `use_blend=False` keeps the old additive path available.
- **`FreshnessBlend`**: always-current files (`MEMORY.md`, `AGENTS.md`, `TOOLS.md`, `IDENTITY.md`, `USER.md`, `SOUL.md`, `RULES.md`, `HEALTHCHECK.md`) now receive a static freshness of `0.8` so they are not out-ranked by today's transcripts. Configurable via `always_current_files` / `always_current_freshness`.
- **`Reranker`**: ID parser now supports composite identifiers (paths, `graph:Kairos`, etc.) and falls back to prefix/substring match when the LLM drops trailing segments (for example emitting `/path/MEMORY.md` instead of `/path/MEMORY.md:45`). The longest full-ID match wins.
- **`Recaller`**: backward-compatible dual constructor. Existing code using `Recaller(embedding_provider=..., vector_store=..., bm25_docs=...)` continues to work; new code can pass `Recaller(retrievers=[VectorRetriever(...), BM25Retriever(...), MemgraphRetriever(), TemporalRetriever(...)])`.

### Fixed

- `Reranker` silently dropped composite IDs because the previous regex (`\b[\w-]+\b`) could not match slashes. The parser now keeps full tokens and strips only trailing punctuation.
- Rerank lookups no longer miss the right result when the LLM truncates a composite ID — fuzzy prefix/substring fallback resolves the longest matching entry in the result set.

### Tests

183 passing (up from 170 in a10) — 7 new tests for `GraphResurrection`, 13 total for the new retrievers + rerank fixes + always-current freshness.

## [0.1.0a10] - 2026-04-17

### Improved

Same set of changes as a9, re-released because a9 on PyPI shipped an intermediate snapshot without the final ablation-tested configuration. See a9 notes below.

## [0.1.0a9] - 2026-04-17

### Improved

- Strengthened temporal answer generation for relative-time memories (for example, converting "yesterday" and "last week" against the memory timestamp instead of returning the session date).
- Added a targeted inference rule for hypothetical and cross-memory questions so the answer layer is less likely to fall back to "Not in memory." when the evidence is present but requires light synthesis.
- Small-batch LoCoMo ablation identified the best-performing configuration as: recaller bugfix from a8 + temporal prompt fix + inference prompt fix, while larger context windows (`max_memories=25`) and more aggressive list prompting did not help.

### Benchmark (LoCoMo, 5 samples / 75 QA, full_pipeline)

- correct 61.3% / combined 80.0%
- By category: cat_2 temporal 88.2%, cat_4 multi-hop 100%, cat_1 list 39%, cat_3 reasoning 27%
- Compared with workspace `enhanced-recall.py` baseline of 51% correct / 68% combined on 1986 QA.

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
