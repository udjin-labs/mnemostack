# Changelog

All notable changes to mnemostack will be documented here. Format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Ollama think control and options passthrough**: `OllamaLLM(think=..., options=...)`. `options` is merged into the request's generation options (e.g. `num_ctx`, `top_p`; explicit keys override the per-call temperature/num_predict). `think` controls reasoning on models that support it and defaults to **False**: reasoning models (qwen3, deepseek-r1 and similar) otherwise spend the whole `num_predict` budget on thoughts and return empty text, which silently degrades reranking, query expansion and extraction into their fallbacks. `think=False` is accepted by every model and ignored by servers that predate the field — only `think=True` errors on models without thinking support. Pass `think=None` to omit the field and keep the model's own default. **Behavior change** for Ollama deployments with reasoning models: internal generations now produce text instead of silently empty responses.

## [0.5.0] - 2026-06-11

### Added

- **Stale-chunk pruning**: `mnemostack index --prune` deletes chunks that a re-indexed file no longer produces (edits shifted offsets, the document shrank, chunking parameters changed). Only the sources indexed in the current run are touched, and the delete is scoped to the indexing root (recorded as a new additive `index_root` payload field): a same-named file indexed from another root, or chunks indexed by earlier versions that did not record a root, are never pruned — rebuild those with `--recreate` if needed. Sources with failed embeddings are skipped (the old chunk stays until a replacement lands). The same logic is available to library users as `mnemostack.ingest.prune_stale_chunks(store, fresh_ids_by_source, index_root=...)`, backed by the new `VectorStore.delete_points(ids)`. Chunk ids themselves are unchanged — the `stable_chunk_id` scheme is API and will only change in a major release.
- **Localizable abstention text**: `AnswerGenerator(abstention_text=...)` replaces the hardcoded English "Not in memory." for non-English deployments — in the empty-memory answer, in the built-in prompts that instruct the LLM which marker to reply with on low evidence, and in the retry logic that detects abstentions (`should_retry` gained an `abstention_text=` parameter; the English literal is always recognized too). User-supplied `prompt_template`/`prompt_overrides` are left verbatim.
- **Pluggable question classifier**: `AnswerGenerator(question_classifier=...)` overrides the built-in English-regex category classifier; an unknown returned category falls back to `general` (fail-open).
- `--raw` flag on `mnemostack search` / `mnemostack answer` restores plain fused recall without the post-processing pipeline.
- **Benchmark harness metadata**: LoCoMo probe outputs now record the `--only-questions` path and the number of QA rows actually evaluated, so filtered probe artifacts are distinguishable from full runs.

### Changed

- **All entry points now rank identically**: CLI, HTTP server and MCP server share one canonical post-recall chain (`recall_flow`: 3x candidate pool -> pipeline -> rerank fail-open -> limit -> vector floor). MCP search/answer and CLI search/answer now run the full 8-stage recall pipeline that was previously HTTP-only; pass `--raw` (CLI) to get the old plain-recall behavior.
- **Learning state moved out of /tmp**: the pipeline state file now defaults to `$XDG_STATE_HOME/mnemostack/server-state.json` (or `~/.local/state/...`), with a one-time automatic migration from the old `/tmp` path. `FileStateStore` takes a cross-process `flock` around reads and writes, so multiple servers sharing one state file no longer corrupt it.
- **Gemini retry storms tamed**: retry backoff is jittered in both the embedding and LLM clients, and batch-embedding fallback probes a single item first — if the provider is down, the remaining items fail fast instead of issuing two retries each.
- **Documentation follow-ups from the PR #54-#60 audit**: the README vision-ingest example is fail-open for captioning, graceful-degradation docs now name the query-expansion exception and trace/degraded tags, and the benchmark README clarifies degraded semantics for empty-ground-truth rows and probe metadata.

### Fixed

- A finalize-stage LLM failure in list-extract mode no longer discards already-extracted items: the answer is built deterministically from them (count questions get the number, list questions get the joined items).
- LLM echoes of the `REWRITTEN_ANSWER:` prompt label are stripped from specificity-resolver output instead of leaking into answers.
- MCP lazy component initialization is serialized with a lock — concurrent first requests no longer build duplicate recallers/embedders.

## [0.4.8] - 2026-06-11

### Added

- **Count/list aggregation recipe**: `AnswerGenerator` now supports the wide-recall-pool plus `list_extract_mode` recipe for count and "list all X" questions. In this mode, count/list questions use a two-pass extract-and-aggregate flow with one extra LLM call; other question categories are unaffected. The recipe is documented in the README and exposed in the LoCoMo harness with `--list-extract` and `--pool`.
- **Per-prompt overrides and explicit category routing**: `AnswerGenerator` now supports per-prompt override text and explicit question-category routing, so callers can localize or specialize prompts while keeping category-aware answer behavior predictable.

## [0.4.7] - 2026-06-11

### Added

- **Provider-universal image description**: `describe_image()` is now available on the `LLMProvider` base class. `OllamaLLM.describe_image()` adds a real implementation through `/api/generate` with the `images` field for vision-capable models such as llava, llama3.2-vision, and qwen2.5-vl. Providers without vision support return a fail-open `LLMResponse` error instead of breaking ingestion.

### Fixed

- **Gemini thinking-model text extraction**: `GeminiLLM._extract_text()` now joins answer text parts and skips thought parts, so Gemini Pro and Flash models with a thinking budget no longer return empty extraction when a thought part appears first.

## [0.4.6] - 2026-06-11

### Added

- **Multimodal ingest support**: `GeminiLLM.describe_image()` adds opt-in VLM image captioning for PR #58, allowing image descriptions to be appended to ingested text while text-only pipelines remain unaffected. Gemini text generation and image captioning now share the `_generate_content` transport, with 4 new vision tests covering the behavior.

## [0.4.5] - 2026-06-10

### Added

- **Timestamps are first-class in ingest**: `IngestItem` gains an explicit `timestamp` field (ISO-8601) that lands in `payload["timestamp"]`; passing it via `metadata` still works. Every payload now records `indexed_at` (UTC). Sliding-window chunks carry the temporal range of the window (`window_start_ts` / `window_end_ts`) alongside the middle item's timestamp.
- **Recall trace and degraded flags**: new `mnemostack.recall.trace` module (`RecallTrace`, `apply_rerank_safe`). `Recaller.recall()` accepts an optional per-call `trace` that captures per-retriever ranked lists (with errors and latency), the fused order, the post-rerank order, and stable degradation tags (`retriever:<name>:failed`, `reranker:fallback`, `reranker:unavailable`, `temporal:no_parse`). HTTP `/recall` and `/answer` always return `degraded` and accept `include_trace`; MCP `mnemostack_search` / `mnemostack_answer` return `degraded`, search accepts `include_trace`. Fail-open behavior is unchanged: degradations are now visible instead of silent.
- **Temporal parsing**: part-of-month expressions in English and Russian (`early/mid/late April`, `в начале/середине/конце апреля` -> day 1-10 / 11-20 / 21-EOM windows) and `around <date>` qualifiers (`around/about/circa/примерно/около/где-то`) widen the window to +/-3 days while keeping the exact target date.
- **LoCoMo benchmark coverage**: photo captions (`blip_caption`) are now ingested into the benchmark corpus, covering image evidence that was previously dropped silently.
- **LoCoMo harness**: per-QA `degraded` recording and an `--only-questions` probe mode for cheap fix-validation iterations.
- **CI**: mypy typecheck job (src fully clean, config in `pyproject.toml`) and coverage reporting in the test matrix.
- `mnemostack index --recreate` now asks for confirmation (shows point count); `--yes/-y` skips the prompt, non-interactive runs without `--yes` exit with code 2.

### Changed

- New LoCoMo headline, measured with the documented default config (`window_size=3`, query expansion, top-K 25) on the same judge (`gemini-3-flash-preview`): **82.9%** strict / **92.7%** combined (signal-only: 78.0% / 90.6%). Two methodology changes vs the previous 82.5/92.2 measurement, both about data fidelity: photo captions are now ingested into the benchmark corpus (697/1540 signal QA cite image turns as evidence), and answer prompts show the time of day of each memory.
- Answer context no longer truncates timestamps to the date: time of day is kept when meaningful (`[2023-05-08 13:41]`), midnight/date-only values render as date. **This changes the answer prompt, so LoCoMo numbers are not directly comparable to the 82.5/92.2 baseline.**
- `ensure_collection()` (sync and async) raises `DimensionMismatchError` when an existing collection stores vectors of a different size than the embedding provider produces. Previously the mismatch surfaced as garbage search results; deployments that were silently mis-dimensioned will now fail loudly at startup.
- The LoCoMo docs now clarify that the LLM reranker and graph retrieval are runtime-only options, not part of the benchmark methodology.

### Fixed

- `TemporalRetriever` emits a `mnemostack.recall.temporal_no_parse` counter and exposes `explain_empty()` so an unparsed date is distinguishable from an empty corpus.

## [0.4.4] - 2026-06-10

### Fixed

- Restored the default rerank behavior to `relevant_only`, matching the pre-0.4.3 baseline while keeping `full_reorder` available through `rerank_mode` / `MNEMOSTACK_RERANK_MODE`.
- Preserved vector-floor candidates in the MCP search and answer path after reranking.
- Aligned search and answer reranking so both paths apply the same rerank mode semantics.

## [0.4.1] - 2026-05-03

- Documentation restructuring and improvements.

## [0.4.0] - 2026-05-03

### Added

- Sliding-window message chunking on ingest, controlled by a new `vector.window_size` config (default `1`, i.e. legacy behavior). Larger windows widen the context each chunk carries, improving recall on questions whose evidence spans neighboring turns — PR #33.
- Opt-in query expansion for recall via the new `--query-expansion` flag on `mnemostack answer`. When enabled, the recaller widens the search with reformulated queries before answering.
- Smart retry on low-confidence answers: when query expansion is on, `AnswerGenerator` retries with an expanded query and a HyDE-style hypothetical answer before giving up. Exposed as `retry_with_expansion` / `expansion_llm` constructor args on `AnswerGenerator`.
- New `mnemostack.recall.query_expansion` module and `Recaller` constructor support for query expansion (`embedding_provider`, `vector_store`, `expansion_llm`).

### Changed

- `Ingestor.ingest()` no longer materializes the input iterator up-front; it streams items lazily so large corpora ingest with bounded memory.
- `AnswerGenerator.generate()` now resolves the question category before short-circuiting on empty memories, so category-aware behavior runs even when retrieval returns nothing.

### Benchmarks

- LoCoMo, all numbers re-measured with the new `gemini-3-flash-preview` judge (apples-to-apples):
  - v0.3.0 baseline: **76.7%** strict (1524/1986) / **88.1%** combined.
  - v0.4.0 with `window_size=3`, `top_k=25`, `gemini-3-flash-preview` answerer: **82.5%** strict / **92.2%** combined.
  - Delta attributable to the v0.4.0 retrieval improvements: **+5.8pp strict / +4.1pp combined**.
- Earlier `gemini-2.5-flash`-judge numbers (e.g. 67.8% / 80.4% on v0.3.0) are not directly comparable — the new judge is more lenient on partials, which alone moves v0.3.0 from 67.8% to 76.7% strict.

### Docs

- Refreshed LoCoMo benchmark numbers using the `gemini-3-flash-preview` judge.
- Fixed category names and aligned table formatting across benchmark documents.

## [0.3.0] - 2026-05-02

### Added

- Entity-centric knowledge synthesis (`synthesize()`) — issue #30.
- `SynthesisFact` / `SynthesisResult` with markdown/JSON rendering.
- CLI command `mnemostack synthesize <entity>`.
- Graceful degradation — works with any subset of backends.
- Related entities extraction from graph.
- Optional LLM summarization pass.

## [0.2.1] - 2026-05-02

### Added

- MD wrapper generation on document ingestion (`Ingestor(wrapper_dir=...)`) — issue #29.
- `IngestItem.tags` and per-item `wrapper_dir` override.
- `IngestStats.wrappers_created` and `wrappers_updated` counters.
- Optional graph integration for File→Tag linking.

### Changed

- Wrapper failures are warning-only and never break ingest.

## [0.2.0] - 2026-04-26

### Changed

- feat(answer): enable proven P1 answer-generation features by default for the 0.2.0 stable release:
  - `category_aware_prompts=True` to route list, temporal, multi-hop, inference, and adversarial questions through category-specific prompts.
  - `specificity_resolver=True` to rewrite placeholder answers when concrete evidence is available.
  - `inference_retry=True` to retry low-confidence open-domain inference answers with decomposed evidence queries.
  - `list_extract_mode=False` remains off by default because LoCoMo results were a wash.
- Explicit constructor arguments remain backward compatible: pass `category_aware_prompts=False`, `specificity_resolver=False`, or `inference_retry=False` to keep the old 0.2.0b6 behavior.

### Benchmarks

- Full LoCoMo run, 10/10 conversations (1986 QA), same Gemini Flash judge for both rows:
  - Strict accuracy: 66.4% (1319/1986, first full run) → **67.8%** (1346/1986, 0.2.0b6 with P1), **+1.4pp**.
  - Combined (correct + partial): 79.2% → **80.4%**, +1.2pp.
  - Per-category strict: `cat_3` open-domain reasoning 31.2% → **41.7%** (+10.5pp), `cat_2` temporal 64.5% → **69.8%** (+5.3pp), `cat_4` 69.2% → 69.6% (+0.4pp), `cat_1` 34.8% → 34.4% (−0.4pp), `cat_5` 90.1% → 89.7% (−0.4pp, within run-to-run noise).
- Conv-50 was rerun in isolation on 2026-04-27 (after the Gemini RPD reset) and merged with the other 9 conversations from the 2026-04-26 b6+all-P1 full run; both halves use the same pipeline and judge.
- Mechanism for the `cat_3` lift: `inference_retry` decomposed evidence queries, which converts a chunk of previous "not in memory" answers into correct ones; combined with `category_aware_prompts` for `cat_2`/`cat_3` the gains stack.
- These are now the official numbers cited in `README.md`.

## [0.2.0b6] - 2026-04-26

### Added

- feat(answer): cat_3 inference retry with query decomposition (#21)

### Fixed

- fix(answer): specificity resolver uses merged_memories after retry (#23)

## [0.2.0b5] - 2026-04-26

### Added

- feat(answer): specificity resolver — opt-in via `specificity_resolver=True` (#20)

## [0.2.0b4] - 2026-04-26

### Added

- feat(answer): retrieval-then-extract for list/count questions, opt-in via `list_extract_mode` (#19)

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
