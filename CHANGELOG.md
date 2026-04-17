# Changelog

All notable changes to mnemostack will be documented here. Format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
