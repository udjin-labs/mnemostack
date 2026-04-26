# Changelog

All notable changes to mnemostack will be documented here. Format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0b3] - 2026-04-26

### Fixed

- feat(answer): multi-hop category protection — fix cat_4 regression from category-aware routing (#17)

## [0.2.0b2] - 2026-04-26

### Added

- feat(answer): category-aware prompt routing (opt-in via `category_aware_prompts=True`) (#16)

## [0.2.0b1] - 2026-04-26

### Changed

- **BREAKING**: `bm25_docs_from_qdrant()` and `BM25Retriever.from_qdrant()` default `limit` changed from `40000` to `None` (unbounded). For users who relied on the implicit cap, pass `limit=40000` explicitly. Rationale: silent data loss for users with active conversations.

### Added

- feat: `newer_than` / `older_than` ISO timestamp filters for rolling-window BM25 corpora loaded from Qdrant.

## [0.2.0a6] - 2026-04-26

### Fixed

- fix(utils): is_heartbeat_poll false positive on mid-message HEARTBEAT_OK mention (#13)

## [0.2.0a5] - 2026-04-26

### Added

- feat: utils.text.strip_metadata_blocks() and is_heartbeat_poll() — clean agent-runtime metadata envelopes from transcripts before chunking/embedding

## [0.2.0a4] - 2026-04-26

### Added

- feat: BM25Retriever.from_qdrant() — build BM25 corpus from existing Qdrant payloads (#11)

### Fixed

- fix: bm25_docs_from_qdrant uses raw Qdrant point ids by default so RRF fuses BM25+vector hits on the same chunk

## [0.2.0a3] - 2026-04-25

### Fixed

- **Temporal year regex extended past 2030.** `extract_temporal()` now matches `20\d{2}` (2000-2099) instead of `20[12]\d` (2010-2029). Queries with years 2030+ no longer silently fall back to current year (PR #7).
- **Month + year matches preferred over bare year matches.** `extract_temporal("april 2035")` now returns the April-only window instead of the entire 2035; bare-year fallback applies only when no month is mentioned (PR #7).
- **`VectorStore.collection_exists()` and `AsyncVectorStore.collection_exists()` no longer mask non-404 errors.** The previous `except Exception: return False` swallowed Qdrant auth, network, and runtime errors, making them look like missing collections to callers. Now only `ValueError("not found")` and `UnexpectedResponse(404)` map to `False`; everything else re-raises (PR #7). Regression tests added.
- **`embedding.model` and `llm.model` from `Config` are now actually applied.** PR #5 added these fields but service surfaces ignored them; CLI / HTTP server / MCP now pass them into `get_provider(...)` and `get_llm(...)` (PR #8).
- **`mnemostack serve --llm` default now reads from `Config`** instead of being hardcoded to `gemini` (PR #8).

### Added

- **Explicit stateful feedback API** (PR #9). The 8-stage pipeline now has a real way to learn from usage:
  - `POST /feedback` HTTP endpoint with `{hit_id, signal, query, query_type, source, sources, reward}`. Signals: `useful` (reward 1.0), `clicked` (0.7), `irrelevant` (0.0); explicit `reward` overrides the map.
  - `mnemostack feedback <hit_id>` CLI command writing into the same state file (PR #10).
  - `mnemostack_feedback` MCP tool with the same contract (PR #10).
  - `--auto-record-ior` / `MNEMOSTACK_AUTO_RECORD_IOR` opt-in flag for the HTTP server. When enabled, `/recall` records returned ids into the inhibition-of-return state automatically. Off by default to avoid silent state drift.
  - `build_full_pipeline(enable_stateful_stages=False)` master toggle for deterministic benchmark runs. When `False`, Q-learning, IoR, and CuriosityBoost are removed from the pipeline so accumulated state cannot affect scores.
  - Shared `mnemostack.feedback` module (`apply_feedback`, `feedback_reward`, `feedback_query_type`, `feedback_sources`, `record_recall_events`, `record_feedback_events`, `FeedbackOutcome`, `FeedbackHit`) used by all three surfaces — single source of truth.
- **CLI overrides for provider models:** `--embedding-model` (in the common parser block) and `--llm-model` (in `answer`/`mcp-serve`/`serve`) (PR #8).
- **`MNEMOSTACK_EMBEDDING_MODEL` and `MNEMOSTACK_LLM_MODEL` environment variables** wired through `Config.load()` (PR #8).
- **`MNEMOSTACK_STATE_PATH` environment variable** for MCP feedback state file location (PR #10).

### Changed

- `ServerConfig.graph_uri` is now `str | None` so HTTP server can run with the graph layer fully disabled (PR #9 internal).
- README and docstrings updated for stateful learning behavior, CLI/MCP feedback usage, and benchmark-mode toggle.

## [0.2.0a2] - 2026-04-25

### Fixed

- **`recall_async` no longer hangs** — replaced the previous `asyncio.to_thread` path with a direct `await asyncio.to_thread(...)` call signature that works reliably under all async test harnesses (PR #2). Health checks gained timeouts so a dead Memgraph cannot stall startup or `/health`.
- **Path validation before destructive index ops** — `mnemostack index --recreate` now verifies the input path exists *before* dropping/recreating the Qdrant collection (PR #3). A typo in `--path` no longer wipes your index.
- **Graph contract is now consistent** — `add_triple` writes `valid_until = "current"` for both nodes and relationships; `MemgraphRetriever`, `GraphResurrection`, `query_triples`, and `end_validity` all read with `coalesce(x.valid_until, 'current') = 'current'`, so legacy `NULL` markers continue to work without migration (PR #3). A `mnemostack graph-migrate-current --dry-run` CLI is provided for hosts that want to normalize storage explicitly.
- **`TemporalRetriever` filter merge** — caller filters (workspace/source/tenant scope) are now preserved instead of being overwritten by the temporal timestamp filter (PR #4).
- **Vector + Temporal dedupe** — temporal hits keep the original Qdrant point id instead of a `temporal:` prefix, so RRF can dedupe the same hit returned by both retrievers (PR #4).
- **Async/sync Qdrant filter parity** — `AsyncVectorStore._build_filter` now dispatches to `DatetimeRange` for ISO-string ranges and `Range` for numeric, matching the sync `VectorStore` behavior (PR #4).
- **Query classification false positives** — `ClassifyQuery` and `is_exact_token_query` now match against tokenized words instead of raw substrings, so markers like `id` or `api` no longer fire inside `idea`/`bridge`/`hidden`/`apiary` (PR #4).

### Changed

- **HTTP server defaults to `127.0.0.1`** instead of `0.0.0.0`, with an explicit warning printed on stderr when `--host 0.0.0.0` is used (PR #3). The HTTP API has no built-in auth or rate limiting; expose it only behind a reverse proxy.
- **Gemini API key is sent via `x-goog-api-key` header** instead of as a `?key=` query string parameter, so it no longer appears in proxy/access logs (PR #3, embeddings + LLM paths).
- **`/recall` and `/answer` no longer leak raw exception text** — backend errors are logged with full stack via `log.exception(...)` and the client receives a generic `"recall failed"` / `"answer failed"` (PR #3).
- **CLI and MCP `search` / `answer` now use the same retriever-mode `Recaller`** as the HTTP server: Vector + Temporal by default, with optional BM25 (`--bm25-path`) and optional Memgraph (`--memgraph-uri`). The legacy `Recaller(embedding_provider=, vector_store=)` path is no longer used for service surfaces (PR #4). Bench numbers from `locomo_single` are now achievable from CLI/MCP, not just HTTP.
- **Single-source `Config`** — CLI, HTTP server (`ServerConfig.from_env`), and MCP (`main()`) all resolve through `Config.load()` with backward-compatible env aliases (`MNEMOSTACK_PROVIDER`/`_EMBEDDING_PROVIDER`/`_EMBEDDING`, `MNEMOSTACK_QDRANT_URL`/`_QDRANT_HOST`/`_VECTOR_HOST`, `MNEMOSTACK_LLM`/`_LLM_PROVIDER`, `MNEMOSTACK_GRAPH_URI`/`_MEMGRAPH_URI`, `MNEMOSTACK_BM25_PATHS`) (PR #5). `Config` itself now carries `graph.timeout`, `graph.health_timeout`, and `recall.bm25_paths`.
- **Docker image installs the `[server]` extra at build time** instead of running `pip install` in the container entrypoint (PR #6). `examples/docker-compose.yml` no longer needs network access at start.

### Added

- `mnemostack graph-migrate-current` CLI to backfill legacy `NULL` graph validity markers to the explicit `"current"` marker (with `--dry-run`).
- `build_bm25_docs(paths)` helper used by HTTP, CLI, and MCP recall setup (PR #4).
- `MNEMOSTACK_GRAPH_TIMEOUT`, `MNEMOSTACK_GRAPH_HEALTH_TIMEOUT`, and `MNEMOSTACK_BM25_PATHS` environment variables (PRs #3, #5).
- Regression coverage: 8 new tests covering env aliases, CLI defaults, server env resolution, async filter parity, graph migration, embedding header, and exception scrub (PRs #3, #4, #5).

### Docs

- README updated for 0.2.x: corrected retriever defaults (Vector + Temporal default, BM25 + Memgraph optional), accurate description of the 8-stage pipeline (stateful stages have APIs but standard CLI/HTTP/MCP do not auto-record feedback), refreshed env table with all new aliases.
- ARCHITECTURE.md aligned with code: `observability/` directory (was `monitoring/`), MCP server marked as shipped with current tool names, graph contract documents the `"current"` marker, config example reflects PR #5 schema.
- SECURITY.md gained explicit notes on HTTP without built-in auth and on LLM-backed features sending memory text to the configured provider.
- Chunking and chunk-id wording corrected (deterministic UUID-shaped content id, deterministic character windows by default).

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
