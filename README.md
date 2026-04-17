# mnemostack

> Memory stack for AI agents — durable, structured, semantically searchable.

`mnemostack` is a hybrid memory system combining BM25, vector search (Qdrant), and knowledge graph (Memgraph) with a unified recall pipeline, reranker, and optional inference layer.

**Status:** 🚧 alpha — under active development.

## Features

- 🧠 **Hybrid retrieval** — BM25 (exact tokens) + vector (semantic) + graph (relationships), fused via Reciprocal Rank Fusion
- 🔌 **Pluggable embeddings** — Gemini, Ollama, or HuggingFace (local GPU), via provider registry
- 📚 **Temporal knowledge graph** — facts have `valid_from`/`valid_until` so you can query point-in-time state
- 💬 **Answer mode** — inference layer synthesizes concise factual answers with source citations and confidence scores
- 🔄 **Consolidation lifecycle** — nightly decay, extraction, and promotion of memories
- 🛡 **Graceful degradation** — retrieval keeps working if graph is down

## Installation

```bash
# Clone and install in dev mode
git clone https://github.com/YOUR_ORG/mnemostack.git
cd mnemostack
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Optional extras
pip install -e ".[huggingface]"  # local GPU embeddings
pip install -e ".[mcp]"          # MCP server
pip install -e ".[dev]"          # tests + linters
```

## Quick start

### Pick an embedding provider

```python
from mnemostack.embeddings import get_provider

# Option A: Gemini (cloud, best quality)
provider = get_provider("gemini")  # reads GEMINI_API_KEY from env

# Option B: Ollama (local, no API key)
provider = get_provider("ollama", model="nomic-embed-text")

# Option C: HuggingFace (local, GPU)
provider = get_provider("huggingface", model="BAAI/bge-large-en-v1.5")

vec = provider.embed("What did we decide about authentication?")
print(f"{provider.name} — dim {provider.dimension}")
```

### Health check

```python
ok, msg = provider.health_check()
print("provider ok" if ok else f"down: {msg}")
```

### Custom provider

Any class that inherits `EmbeddingProvider` can be registered:

```python
from mnemostack.embeddings import EmbeddingProvider, register_provider

class MyProvider(EmbeddingProvider):
    @property
    def name(self): return "my-provider"
    @property
    def dimension(self): return 512
    def embed(self, text): ...
    def embed_batch(self, texts): ...

register_provider("my-provider", MyProvider)
```

## Design

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design: pipeline stages, Qdrant schema, Memgraph temporal model, consolidation runtime, MCP tools.

## Roadmap

- [x] Embedding provider registry (Gemini / Ollama / HuggingFace)
- [ ] Qdrant wrapper + chunking
- [ ] BM25 + RRF + reranker pipeline
- [ ] Answer mode (Gemini Flash inference)
- [ ] Memgraph wrapper + temporal queries
- [ ] Consolidation runtime (decay, promote, summarize)
- [ ] CLI (`mnemostack search`, `mnemostack index`, `mnemostack runtime`)
- [ ] MCP server (Model Context Protocol for Claude/GPT clients)

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Contributing

Early days. Issues and PRs welcome once API stabilizes.
