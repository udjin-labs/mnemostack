# mnemostack

> Memory stack for AI agents — durable, structured, semantically searchable.

`mnemostack` is a hybrid memory system combining BM25, vector search (Qdrant), and knowledge graph (Memgraph) with a unified recall pipeline, reranker, and optional LLM inference layer.

**Status:** 🚧 alpha — API may change between 0.1.x releases.

## Benchmarks

Full LoCoMo run (official SNAP-Research dataset, 10 samples / **1986 QA**, clean state, judged by Gemini Flash):

| Metric | mnemostack 0.1.0a11 |
| --- | --- |
| **Correct (strict)** | **66.4%** (1319 / 1986) |
| Partial | 12.8% (254) |
| Wrong | 20.8% (413) |
| **Combined (correct + partial)** | **79.2%** |

By question category:

| Category | Correct |
| --- | --- |
| `cat_5` adversarial open-domain | **90.1%** |
| `cat_4` multi-hop reasoning | 69.2% |
| `cat_2` temporal | 64.5% |
| `cat_1` single-hop lists | 34.8% |
| `cat_3` open-domain reasoning | 31.2% |

How that compares with reported numbers from other systems on the same benchmark (caveat: different judges and evaluation protocols):

| System | LoCoMo correct |
| --- | --- |
| Hindsight (leader) | 78–85% |
| Memobase (temporal subset) | 85% |
| Letta filesystem agent | 74% |
| Mem0 graph variant | ~68.5% |
| **mnemostack 0.1.0a11** | **66.4%** |
| Zep (independently replicated) | 58.4% |

Reproduce with `python benchmarks/locomo_single.py --samples 10` from a clone; the runner only needs a `GEMINI_API_KEY`.

## Features

- 🧠 **Hybrid retrieval** — BM25 (exact tokens) + vector (semantic), fused via Reciprocal Rank Fusion
- 🔌 **Pluggable embeddings** — Gemini, Ollama, or HuggingFace (local GPU), via provider registry
- 🤖 **Pluggable LLM** — Gemini Flash / Ollama for answer generation and reranking
- 📚 **Temporal knowledge graph** — facts have `valid_from`/`valid_until`, query point-in-time state
- 💬 **Answer mode** — inference layer synthesizes concise factual answers with source citations and confidence
- 🔁 **Reranker** — LLM-based reordering of top results
- ⚙ **Consolidation runtime** — phase orchestrator for nightly memory lifecycle
- 🔌 **MCP server** — expose memory tools to Claude Desktop, ChatGPT, Cursor, etc.
- 🛡 **Graceful degradation** — retrieval keeps working if graph is down

## Installation

```bash
# From PyPI
pip install mnemostack

# Optional extras
pip install 'mnemostack[huggingface]'  # local GPU embeddings
pip install 'mnemostack[mcp]'          # MCP server
pip install 'mnemostack[dev]'          # tests + linters
```

Run a local Qdrant for the vector store:

```bash
docker run -p 6333:6333 qdrant/qdrant:latest
```

Optionally a Memgraph for the knowledge graph:

```bash
docker run -p 7687:7687 memgraph/memgraph:latest
```

## Quick start

### CLI

```bash
# Health check
mnemostack health --provider ollama

# Index a directory of notes
mnemostack index ./my-notes/ --provider gemini --collection my-memory --recreate

# Hybrid recall
mnemostack search "what did we decide about auth" --provider gemini --collection my-memory

# Synthesize answer
mnemostack answer "what is the capital of France" --provider gemini --collection my-memory

# MCP server (for Claude Desktop, Cursor, etc.)
mnemostack mcp-serve --provider gemini --collection my-memory
```

### Python API

```python
from mnemostack.embeddings import get_provider
from mnemostack.vector import VectorStore
from mnemostack.recall import Recaller, AnswerGenerator
from mnemostack.llm import get_llm

emb = get_provider("gemini")
store = VectorStore(collection="my-memory", dimension=emb.dimension)
store.ensure_collection()

# ... index data here ...

recaller = Recaller(embedding_provider=emb, vector_store=store)
results = recaller.recall("what did we decide", limit=10)

# Optional: synthesize a concise answer
gen = AnswerGenerator(llm=get_llm("gemini"))
answer = gen.generate("what did we decide", results)
print(answer.text, answer.confidence, answer.sources)
```

### Knowledge graph (optional)

```python
from mnemostack.graph import GraphStore

graph = GraphStore(uri="bolt://localhost:7687")
graph.add_triple("alice", "works_on", "project-x", valid_from="2024-01-01")
graph.add_triple("alice", "works_on", "project-y", valid_from="2024-07-01")

# Who was alice working on in March?
march_facts = graph.query_triples(subject="alice", as_of="2024-03-15")
```

### MCP server for Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mnemostack": {
      "command": "mnemostack",
      "args": ["mcp-serve", "--provider", "gemini", "--collection", "my-memory"],
      "env": {
        "GEMINI_API_KEY": "your-key-here"
      }
    }
  }
}
```

Claude will then be able to call `mnemostack_search`, `mnemostack_answer`, and graph tools.

### Custom embedding provider

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
- [x] LLM provider registry (Gemini Flash / Ollama)
- [x] Qdrant wrapper
- [x] BM25 + RRF recall pipeline
- [x] Answer mode with confidence + citations
- [x] LLM-based reranker
- [x] Memgraph wrapper with temporal validity
- [x] Consolidation runtime (phase orchestrator)
- [x] CLI (`mnemostack health/search/answer/index/mcp-serve`)
- [x] MCP server (Model Context Protocol)
- [ ] Text → graph triple extractor helpers
- [ ] Config file support (YAML/JSON)
- [ ] Async variants for high-throughput servers
- [ ] Docker compose examples

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Contributing

Early days. Issues and PRs welcome once API stabilizes.
