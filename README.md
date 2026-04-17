# mnemostack

> Memory stack for AI agents — durable, structured, semantically searchable.

`mnemostack` is a hybrid memory system combining BM25, vector search (Qdrant), and knowledge graph (Memgraph) with a unified recall pipeline, reranker, and optional LLM inference layer.

**Status:** 🚧 alpha — API may change between 0.1.x releases.

### Who is this for?

Build it in if you need:

- Long-lived agent memory that survives session restarts and doesn't drift into irrelevance as the corpus grows.
- Recall quality on **mixed workloads** — exact-token lookups (IDs, tickers, error strings), semantic queries, temporal questions, multi-hop reasoning — not just one of them.
- A stack you can **plug into your own infrastructure**: bring your own embedding model, LLM, vector store, or graph DB.

Not the best fit if you only need a single call to `text-embedding-3-small` + cosine similarity — something simpler will do. mnemostack earns its complexity on mixed, long-horizon workloads.

### How it works, in one paragraph

On each `recall(query)`: the four retrievers (Vector, BM25, Memgraph, Temporal) run in parallel and return ranked lists. Reciprocal Rank Fusion merges them. The 8-stage pipeline reweights results using query classification, exact-token rescue, gravity/hub dampening (to avoid always-winning popular chunks), freshness, inhibition-of-return (to not return the exact same thing twice in a row), curiosity boosts, a Q-learning reranker learned from usage, and graph resurrection (pull in related facts that weren't in top-K). An optional LLM reranker does a final ordering pass. You get a list of `RecallResult` with source, score, and provenance — ready to hand to a model.

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

> **Honest numbers disclaimer.** The table above is our full-benchmark number across **all 1986 questions and all 5 categories**. Some vendors report their strongest sub-category only; if we did the same we could honestly claim **90.1% on adversarial open-domain** or **69.2% on multi-hop reasoning**. We publish the full aggregate because that's what actually predicts how the system behaves on mixed workloads.

How that compares with reported numbers from other systems on the same benchmark (caveat: different judges, evaluation protocols, and in some cases category cherry-picking):

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

- 🧠 **4-source hybrid retrieval** — Vector (Qdrant) + BM25 (exact tokens) + Memgraph (knowledge graph) + Temporal (time-aware vector), all fused via Reciprocal Rank Fusion. Pluggable `Retriever` abstraction — add your own sources.
- ⚡ **8-stage recall pipeline** — ClassifyQuery → ExactTokenRescue → GravityDampen → HubDampen → FreshnessBlend → InhibitionOfReturn → CuriosityBoost → QLearningReranker. Opt-in, with persistent state store.
- 🔁 **LLM reranker** — Gemini Flash (or any LLM) reorders top-K by relevance; catches cases where embedding similarity alone is too broad.
- 🔌 **Pluggable embeddings** — Gemini, Ollama, or HuggingFace (local GPU), via provider registry
- 🤖 **Pluggable LLM** — Gemini Flash / Ollama for answer generation and reranking
- 📚 **Temporal knowledge graph** — facts have `valid_from`/`valid_until`, query point-in-time state; graph resurrection stage recovers evicted-but-relevant memories.
- 💬 **Answer mode** — inference layer synthesizes concise factual answers with source citations and confidence
- ✂️ **Chunkers** — plain, fixed-size, and `MessagePairChunker` for chat transcripts (keeps user↔assistant pairs together).
- 🔎 **Query expansion** — optional `QueryExpander` rewrites short queries for better recall before fusion.
- ⚙ **Consolidation runtime** — phase orchestrator for nightly memory lifecycle
- 🔌 **MCP server** — expose memory tools to Claude Desktop, ChatGPT, Cursor, etc.
- 🛡 **Graceful degradation** — retrieval keeps working if graph or any retriever is down

## Environment

| Variable | Purpose | Required for |
| --- | --- | --- |
| `GEMINI_API_KEY` | Google Generative AI key | Gemini embedding + Gemini Flash LLM |
| `OLLAMA_HOST` | Ollama server URL (default `http://localhost:11434`) | Ollama embeddings / LLM |
| `MNEMOSTACK_COLLECTION` | Qdrant collection name (default `mnemostack`) | CLI convenience |
| `MNEMOSTACK_QDRANT_URL` | Qdrant URL (default `http://localhost:6333`) | Remote Qdrant |
| `MNEMOSTACK_GRAPH_URI` | Memgraph bolt URI (default `bolt://localhost:7687`) | Graph retriever / GraphStore |

Only the providers you actually use need their keys. HuggingFace local-GPU embeddings need no keys at all.

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

# Each result: .id .text .score .source ("vector" | "bm25" | "memgraph" | "temporal") .metadata

# Optional: synthesize a concise answer
gen = AnswerGenerator(llm=get_llm("gemini"))
answer = gen.generate("what did we decide", results)
print(answer.text, answer.confidence, answer.sources)
```

#### Full stack: 4-source retrieval + 8-stage pipeline + reranker

This is the configuration that produced the 66.4% / 79.2% LoCoMo numbers above.

```python
from mnemostack.embeddings import get_provider
from mnemostack.llm import get_llm
from mnemostack.vector import VectorStore
from mnemostack.recall import (
    Recaller, Reranker,
    VectorRetriever, BM25Retriever,
    MemgraphRetriever, TemporalRetriever,
    build_full_pipeline,
)
from mnemostack.recall.pipeline import FileStateStore

emb = get_provider("gemini")
store = VectorStore(collection="my-memory", dimension=emb.dimension)

retrievers = [
    VectorRetriever(embedding=emb, vector_store=store),
    BM25Retriever(docs=bm25_docs),                       # see "Building a BM25 corpus" below
    MemgraphRetriever(uri="bolt://localhost:7687"),      # optional
    TemporalRetriever(embedding=emb, vector_store=store),
]
recaller = Recaller(retrievers=retrievers)
raw = recaller.recall("what did we decide", limit=30)

pipeline = build_full_pipeline(state_store=FileStateStore("/tmp/mnemo-state.json"))
reranked = pipeline.apply("what did we decide", raw)
reranker = Reranker(llm=get_llm("gemini"), max_items=20)
final = reranker.rerank("what did we decide", reranked)[:10]
```

##### Building a BM25 corpus

`BM25Retriever` needs a list of `BM25Doc`. Each doc is the atomic unit BM25 will rank — typically a paragraph or chunk of one of your source files:

```python
from mnemostack.recall import BM25Doc
from pathlib import Path

docs = []
for i, path in enumerate(Path("my-notes/").rglob("*.md")):
    text = path.read_text()
    # chunk however you like — here: 800-char windows
    for j in range(0, len(text), 800):
        chunk = text[j : j + 800]
        if chunk.strip():
            docs.append(BM25Doc(
                id=f"{path.name}:{j}",
                text=chunk,
                payload={"source": str(path), "offset": j},
            ))
```

For transcript-like inputs (user↔assistant messages), prefer `MessagePairChunker` so a question and its answer stay in the same chunk. See `mnemostack.chunking`.

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

### Pipeline state

The 8-stage pipeline needs a tiny bit of state between calls (Q-learning weights, inhibition-of-return history, per-document gravity/hub counters). `FileStateStore(path)` persists it to a JSON file. For multi-process servers, implement your own `StateStore` (two methods: `load()` / `save(state)`) backed by Redis or your database.

### Graceful degradation

Any retriever can fail (Memgraph down, Qdrant unreachable, BM25 corpus empty). `Recaller` logs and continues with the remaining sources. The LLM reranker is wrapped in try/except by convention — if the LLM is rate-limited, the pre-rerank order is returned. This is deliberate: a memory stack that goes dark because one component hiccuped is worse than a slightly degraded one.

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
