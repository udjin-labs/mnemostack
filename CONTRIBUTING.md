# Contributing to mnemostack

## Getting started

```bash
git clone <your-fork>
cd mnemostack
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,mcp,huggingface]'
```

## Running tests

```bash
pytest tests/ -v
```

All tests should pass before submitting a PR. Integration tests for Memgraph
require a live server at `bolt://localhost:7687`; they skip automatically if
unreachable. Start Memgraph via `docker compose -f examples/docker-compose.yml up -d`.

## Code style

- Ruff + mypy enforced via `.[dev]` extras.
- Type hints required on public API.
- Docstrings for public classes and functions.
- Line length 100 (configured in `pyproject.toml`).

## Project structure

```
src/mnemostack/
├── embeddings/       # Pluggable embedding providers
├── llm/              # Pluggable LLM providers (for answer + rerank)
├── vector/           # Qdrant wrappers (sync + async)
├── recall/           # BM25, RRF, Recaller, AnswerGenerator, Reranker
├── graph/            # Memgraph wrapper with temporal validity + TripleExtractor
├── consolidation/    # Phase orchestrator for memory lifecycle
├── chunking/         # Text chunking strategies (char / paragraph / markdown)
├── mcp/              # FastMCP server
├── config.py         # YAML config loader
├── cli.py            # argparse CLI entry point
└── logging_config.py # Structured logging setup
```

## Design principles

- **Pluggable.** Embeddings and LLMs go through registries. Users swap providers in config.
- **Graceful degradation.** If graph is down, retrieval works on BM25+vector. If LLM fails, recall returns raw snippets.
- **Same model for index and query.** Mixing embedding dimensions silently breaks similarity — we validate.
- **Temporal truth.** Graph edges have `valid_from`/`valid_until`; facts can expire without deletion.
- **Confidence over certainty.** Answer mode returns a confidence score; callers decide what to trust.

## Adding a new embedding provider

1. Subclass `mnemostack.embeddings.EmbeddingProvider`, implement `embed`, `embed_batch`, `dimension`, `name`.
2. Optionally register lazily in `embeddings/registry.py:_lazy_register_builtins`.
3. Add tests in `tests/test_embeddings.py`.

## Adding a new LLM provider

1. Subclass `mnemostack.llm.LLMProvider`, implement `generate` returning `LLMResponse`.
2. Register in `llm/registry.py`.

## Publishing (maintainers)

```bash
# Bump version in pyproject.toml and src/mnemostack/__init__.py
rm -rf dist build src/mnemostack.egg-info
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

Follow CHANGELOG.md.

## License

By contributing, you agree your contributions are licensed under Apache 2.0.
