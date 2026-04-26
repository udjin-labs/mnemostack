# mnemostack

[![PyPI](https://img.shields.io/pypi/v/mnemostack.svg)](https://pypi.org/project/mnemostack/)
[![Python versions](https://img.shields.io/pypi/pyversions/mnemostack.svg)](https://pypi.org/project/mnemostack/)
[![CI](https://github.com/udjin-labs/mnemostack/actions/workflows/ci.yml/badge.svg)](https://github.com/udjin-labs/mnemostack/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> Memory stack for AI agents — durable, structured, semantically searchable.

`mnemostack` is a hybrid memory system combining BM25, vector search (Qdrant), and knowledge graph (Memgraph) with a unified recall pipeline, reranker, and optional LLM inference layer.

**Status:** 🚧 alpha — API may change between 0.2.x releases.

### Who is this for?

Build it in if you need:

- Long-lived agent memory that survives session restarts and doesn't drift into irrelevance as the corpus grows.
- Recall quality on **mixed workloads** — exact-token lookups (IDs, tickers, error strings), semantic queries, temporal questions, multi-hop reasoning — not just one of them.
- A stack you can **plug into your own infrastructure**: bring your own embedding model, LLM, vector store, or graph DB.

Not the best fit if you only need a single call to `text-embedding-3-small` + cosine similarity — something simpler will do. mnemostack earns its complexity on mixed, long-horizon workloads.

### Mental model

Think of it as a storage hierarchy for agent memory:

- **Context window = RAM.** Fast, limited (typically 200K tokens for most agent models; up to 1M in frontier offerings like Claude Opus 4.6/4.7 or GPT-5.4 Codex), often far less usable after MCP/instructions/context rot — ~45K on a 200K window is a realistic working number. Clears on session restart.
- **mnemostack corpus = Disk.** Persistent, searchable, grows forever — every fact the agent has ever seen, queryable on demand.
- **`recall(query)` = page fault handler.** When the agent needs something that isn't in the current context, it pulls the exact fact from storage with a single hybrid query — not a grep, not a reload of the whole corpus.

The practical effect: you stop re-explaining your project to the agent after every `/compact`. You stop losing momentum to the re-orientation tax that shows up in any agent with session compaction. mnemostack solves it at the library level, not tied to any single agent runtime.

### How it works, in one paragraph

On each `recall(query)`: the configured retrievers (Vector and Temporal by default, with BM25 and Memgraph when configured) run in parallel and return ranked lists. Reciprocal Rank Fusion merges them. The optional 8-stage pipeline can reweight results using query classification, exact-token rescue, gravity/hub dampening, freshness, inhibition-of-return, curiosity boosts, Q-learning weights supplied through its state store, and graph resurrection. An optional LLM reranker does a final ordering pass. You get a list of `RecallResult` with source, score, and provenance — ready to hand to a model.

![mnemostack architecture](docs/images/mnemostack-architecture.jpg)

## Benchmarks

Full LoCoMo run (official SNAP-Research dataset, 10 samples / **1986 QA**, clean state, judged by Gemini Flash):

| Metric | mnemostack |
| --- | --- |
| **Correct (strict)** | **66.5%** (1320 / 1986) |
| Partial | 13.3% (265) |
| Wrong | 20.2% (401) |
| **Combined (correct + partial)** | **79.8%** |

By question category:

| Category | Correct |
| --- | --- |
| `cat_5` adversarial open-domain | **90.1%** |
| `cat_4` multi-hop reasoning | **69.3%** |
| `cat_2` temporal | **65.1%** |
| `cat_1` single-hop lists | 33.7% |
| `cat_3` open-domain reasoning | 32.3% |

_Last run: 2026-04-26, mnemostack 0.2.0b1._

> **Honest numbers disclaimer.** The table above is our full-benchmark number across **all 1986 questions and all 5 categories**. Some vendors report their strongest sub-category only; if we did the same we could honestly claim **90.1% on adversarial open-domain** or **69.3% on multi-hop reasoning**. We publish the full aggregate because that's what actually predicts how the system behaves on mixed workloads.

How that compares with reported numbers from other systems on the same benchmark (caveat: different judges, evaluation protocols, and in some cases category cherry-picking):

| System | LoCoMo correct |
| --- | --- |
| Hindsight (leader) | 78–85% |
| Memobase (temporal subset) | 85% |
| Letta filesystem agent | 74% |
| Mem0 graph variant | ~68.5% |
| **mnemostack** | **66.5%** |
| Zep (independently replicated) | 58.4% |

### Real-corpus needle benchmark

LoCoMo measures generic long-term dialogue recall. We also run a private needle-in-haystack benchmark on the production workload that drove the original design — a ~17k-point memory stack indexed from a long-running assistant. Queries mix exact tokens (IP addresses, tickers), telegram IDs, paraphrased facts, and temporal probes.

| Metric | Value |
| --- | --- |
| recall@1 | 90% (9/10) |
| recall@5 | **100%** (10/10) |
| recall@10 | 100% (10/10) |
| Query latency p50 | 1.26 s |
| Query latency max | 1.70 s |

Useful because LoCoMo's failure modes (list exhaustion, open-domain reasoning) are orthogonal to what production memory stacks actually spend time on (find the specific fact the user mentioned weeks ago). This benchmark is not in the public repo; its methodology is in `benchmarks/synthetic_longhorizon.py`, which is the closest reproducible approximation.

### Reproduce LoCoMo from a fresh clone

```bash
pip install -e '.[dev]'
bash benchmarks/download_locomo.sh   # fetches SNAP Research's public dataset
export GEMINI_API_KEY=...
bash benchmarks/run_locomo.sh        # full 10-sample run, writes results/ts.{json,log}
```

Details, category definitions, and notes on the judge protocol: [benchmarks/README.md](benchmarks/README.md).

## Where mnemostack fits

Most memory tools in the agent ecosystem pick one axis and optimize for it: simple vector similarity for RAG, framework-bound memory tied to a specific agent library, platform-level runtimes with audit and compliance features, or CLI wrappers over a single vendor's session store. Each makes sense for its scope.

mnemostack takes a different slice: it is a **recall quality layer**, offered as a plain Python package. Four retrievers (Vector + BM25 + Memgraph + Temporal), RRF fusion, an 8-stage pipeline, and an optional LLM reranker — composed to handle mixed workloads on the same corpus: exact-token lookups, semantic queries, temporal questions, and multi-hop reasoning, without forcing you to choose one mode over another.

We are not a replacement for your agent framework and not a full platform runtime. We are the piece that actually finds the right fact in a growing corpus. Drop mnemostack into your own Python agent, or let a higher-level memory service call `recall()` over a plain function boundary. The retrievers, pipeline, and reranker are individually composable — take only the parts you need.

## Features

- 🧠 **4-source hybrid retrieval** — Vector (Qdrant) + BM25 (exact tokens) + Memgraph (knowledge graph) + Temporal (time-aware vector), all fused via Reciprocal Rank Fusion. Pluggable `Retriever` abstraction — add your own sources.
- ⚖️ **Weighted & adaptive RRF fusion** — `reciprocal_rank_fusion(weights=[...])` lets you lift sources you trust more; `Recaller(adaptive_weights=True)` picks a per-query-shape profile (exact-token / person / temporal / general). See the honest write-up below for where this helps and where it doesn't.
- 🧪 **HyDE retriever (opt-in)** — embeds a hypothetical answer instead of the query. Useful for query↔document vocabulary gaps in documentation-style corpora; does **not** reliably help on dialogue-backed memory and always costs one extra LLM roundtrip per `search()`. Not included in the default `Recaller`.
- ⚡ **8-stage recall pipeline** — ClassifyQuery → ExactTokenRescue → GravityDampen → HubDampen → FreshnessBlend → InhibitionOfReturn → CuriosityBoost → QLearningReranker. Opt-in; stateful HTTP feedback is explicit via `/feedback`, and recall exposure logging is off unless `--auto-record-ior` is enabled.
- 🔁 **LLM reranker** — Gemini Flash (or any LLM) reorders top-K by relevance; catches cases where embedding similarity alone is too broad.
- ⚡ **Async-friendly** — `Recaller.recall_async` dispatches retrievers in parallel; five concurrent HTTP recalls finish in roughly one single-recall wall-clock.
- 🌍 **Unicode-aware entity resolution** — Memgraph retriever probes by `telegram_id`, handle, and precomputed `name_lower` so non-ASCII names match correctly (Memgraph's `toLower()` lower-cases ASCII only).
- 📥 **Streaming `Ingestor` API** — batched, idempotent, LRU-cached ingest from any Python code. Same `(source, offset, text)` → same deterministic UUID-shaped content id, so re-runs are no-ops.
- 🌐 **HTTP API** (optional) — `pip install 'mnemostack[server]'` gives you `/recall`, `/answer`, `/health`, `/docs`, plus `/metrics` in Prometheus text format. See the HTTP server section below.
- 🔌 **Pluggable embeddings** — Gemini, Ollama, or HuggingFace (local GPU), via provider registry
- 🤖 **Pluggable LLM** — Gemini Flash / Ollama for answer generation and reranking
- 📚 **Temporal knowledge graph** — facts have `valid_from`/`valid_until`, query point-in-time state; graph resurrection stage recovers evicted-but-relevant memories.
- 💬 **Answer mode** — inference layer synthesizes concise factual answers with source citations and confidence
- 📏 **Progressive Tiers API** — `search --tier {1,2,3}` and `answer --tier {1,2,3}` bound output size (~50 / ~200 / ~500 tokens) so agents can pay only for the detail they actually need. Omit `--tier` for unchanged full output.
- ✂️ **Chunkers** — plain, fixed-size, and `MessagePairChunker` for chat transcripts (keeps user↔assistant pairs together).
- 🔎 **Query expansion** — optional `QueryExpander` rewrites short queries for better recall before fusion.
- ⚙ **Consolidation runtime** — phase orchestrator for nightly memory lifecycle
- 🔌 **MCP server** — expose memory tools to Claude Desktop, ChatGPT, Cursor, etc.
- 🛡 **Graceful degradation** — retrieval keeps working if graph or any retriever is down

## Utilities

Agent runtimes often wrap transcript messages in metadata envelopes before the real body, which can dominate embeddings and make unrelated turns look similar. Clean messages before chunking/indexing with `strip_metadata_blocks()`:

```python
from mnemostack.utils import strip_metadata_blocks

clean = strip_metadata_blocks(raw_message)
```

Built-in profiles cover OpenClaw webchat and Telegram envelopes; pass `profiles=` or `extra_patterns=` to tune the cleanup for your runtime.

### Fusion weights & HyDE — honest notes

Some of the newer knobs help in specific workloads and do nothing (or mildly hurt) in others. Measured, not promised:

**`Recaller(adaptive_weights=True)`** — picks a weight profile per query shape:

| Query shape | Detection | Profile (bm25 / memgraph / vector / temporal) |
| --- | --- | --- |
| `exact_token` | IPv4 / port / version / UUID / API-style tokens | 1.4 / 1.4 / 1.0 / 0.9 |
| `person` | "who is", `@handle`, `username`, `contact`, etc. | 1.0 / 1.5 / 1.0 / 0.9 |
| `temporal` | "when", "yesterday", "today", dates | 1.0 / 1.0 / 1.0 / 1.4 |
| `general` | everything else | classical equal-weight RRF |

Measured on a real production corpus with 10 needle probes: **recall@1 went 50% → 60%, recall@5 stayed at 90%** (zero regression). On LoCoMo (pure dialogue questions, all classified `general`), adaptive weights had no effect — the profile simply isn't triggered. Rule of thumb: turn it on for production ops-style workloads (IPs, tickers, IDs, named entities); leave it off, or don't expect a lift, for dialogue benchmarks. Static `retriever_weights={...}` always wins over adaptive when both are set.

**`HyDERetriever`** — opt-in. Generates a short hypothetical answer via your LLM and embeds that instead of the raw query, then fuses alongside the other retrievers. Useful when the question and the stored answer use very different vocabulary (documentation corpora, FAQ-style content). On our LoCoMo cat_3 smoke (`conv-43`, 14 reasoning questions) it moved accuracy from 14.3% to 21.4% (+1 correct answer); on dialogue-backed memory overall it's roughly a wash. It **always** costs one extra LLM call per `search()`, so budget accordingly and treat it as a tool for specific workloads rather than a default.

Both knobs are opt-in by design — the default `Recaller` stays classical equal-weight RRF over Vector + BM25 (+ Memgraph + Temporal when supplied).

## Environment

| Variable | Purpose | Required for |
| --- | --- | --- |
| `GEMINI_API_KEY` | Google Generative AI key | Gemini embedding + Gemini Flash LLM |
| `OLLAMA_HOST` | Ollama server URL (default `http://localhost:11434`) | Ollama embeddings / LLM |
| `MNEMOSTACK_COLLECTION` | Qdrant collection name (default `mnemostack`) | CLI convenience |
| `MNEMOSTACK_QDRANT_URL` | Qdrant URL (default `http://localhost:6333`) | Remote Qdrant |
| `MNEMOSTACK_GRAPH_URI` / `MNEMOSTACK_MEMGRAPH_URI` | Memgraph bolt URI | Graph retriever / GraphStore |
| `MNEMOSTACK_PROVIDER` / `MNEMOSTACK_EMBEDDING_PROVIDER` | Embedding provider | CLI / HTTP / MCP |
| `MNEMOSTACK_LLM` / `MNEMOSTACK_LLM_PROVIDER` | LLM provider | Answer generation / reranking |
| `MNEMOSTACK_BM25_PATHS` | BM25 corpus paths separated by `os.pathsep` (`:` on Unix) | CLI / HTTP / MCP BM25 retriever |
| `MNEMOSTACK_AUTO_RECORD_IOR` | `true`/`false` toggle for HTTP recall exposure logging | HTTP stateful pipeline |

Only the providers you actually use need their keys. HuggingFace local-GPU embeddings need no keys at all. `mnemostack init` writes the same settings as YAML; explicit CLI flags override config/env defaults.

## Try it in 30 seconds (Docker)

Fastest way to kick the tyres. No Python install, no manual Qdrant / Memgraph setup.

```bash
git clone https://github.com/udjin-labs/mnemostack && cd mnemostack
cp README.md examples/notes/              # any markdown will do
GEMINI_API_KEY=your-key docker compose -f examples/docker-compose.yml up -d --build

# Index the notes volume and ask a question over HTTP
docker compose -f examples/docker-compose.yml exec mnemostack \
    mnemostack index /data --provider gemini --collection demo

curl -s http://localhost:8000/recall \
    -H 'content-type: application/json' \
    -d '{"query":"what is this about","limit":5}' | jq
```

The mnemostack container runs the HTTP API on port 8000 by default. Interactive docs are at [http://localhost:8000/docs](http://localhost:8000/docs). Use `docker compose exec mnemostack mnemostack <cmd>` for CLI-style operations (`index`, `search`, `health`) against the same stack.

Tear down with `docker compose -f examples/docker-compose.yml down -v` (the `-v` wipes Qdrant + Memgraph state).

Prefer Ollama (no cloud key needed)? Run Ollama on the host, set `OLLAMA_HOST=http://host.docker.internal:11434`, and pass `--provider ollama` everywhere instead of `gemini`.

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

# Record explicit feedback into the same state file used by the HTTP/MCP pipeline
mnemostack feedback <hit-id> --signal clicked --query "what did we decide about auth" \
  --source-list vector --source-list bm25

# MCP server (for Claude Desktop, Cursor, etc.)
mnemostack mcp-serve --provider gemini --collection my-memory
```

#### Progressive tiers — pay only for the detail you need

`search` and `answer` accept an optional `--tier {1,2,3}` flag that bounds how
much output a call produces. Useful when a recall is called from a long-running
agent loop where full recall output would burn context unnecessarily.

```bash
# Tier 1 (~50 tokens) — just "is there anything in memory about this?"
# Returns id, score, source labels; no text.
mnemostack search "VPN failover" --tier 1 --provider gemini

# Tier 2 (~200 tokens) — triage with short snippets (~40 chars each)
mnemostack search "VPN failover" --tier 2 --provider gemini

# Tier 3 (~500 tokens) — full 200-char previews, up to 10 results
mnemostack search "VPN failover" --tier 3 --provider gemini
```

Omit `--tier` to get the full, uncapped output (backward compatible). Rule of
thumb for agents: **tier 1 for navigation / existence checks, tier 3 only when
you actually need to read the memories.** `answer` is already compressed, so it
needs a tier less often — use `--tier 1` there to drop the `SOURCES:` block
when only the answer text is wanted.

### Streaming ingest API

When you want to feed items into mnemostack from code — a chatbot that logs every message, a scraper, a daemon tailing a log — use the `Ingestor`. It handles batching, deduplication, and idempotency for you.

```python
from mnemostack.embeddings import get_provider
from mnemostack.vector import VectorStore
from mnemostack import Ingestor, IngestItem

emb = get_provider("gemini")
store = VectorStore(collection="my-memory", dimension=emb.dimension)
store.ensure_collection()

ing = Ingestor(embedding=emb, vector_store=store, batch_size=64)

stats = ing.ingest([
    IngestItem(text="alice joined acme on 2024-03-01", source="notes/alice.md"),
    IngestItem(text="alice left acme on 2025-06-15", source="notes/alice.md", offset=100),
])
print(stats)  # IngestStats(seen=2, embedded=2, upserted=2, skipped=0, failed=0)
```

Guarantees:

- **Idempotent.** Each item gets a deterministic UUID-shaped content id computed from `(source, offset, text)`. Re-running with the same input is a no-op: Qdrant upsert replaces the point onto itself, and an in-process LRU cache skips even the embedding call for items already seen in this session.
- **Batched.** Items are embedded in batches of `batch_size`, so provider HTTP overhead amortises across many items.
- **Streaming-friendly.** `ing.stream(item_iter)` yields per-batch stats so long feeds can be monitored without waiting for the whole stream to drain.
- **Graceful.** If a single item fails to embed, it is counted as `failed` but the rest of the batch still lands.

```python
# Long-lived feed (e.g. inside a FastAPI or Celery worker)
for item in your_firehose():
    ing.ingest_one(IngestItem(text=item.body, source=item.channel, metadata={
        "user_id": item.user_id,
        "ts": item.ts.isoformat(),
    }))
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

This is the configuration that produced the 66.5% / 79.8% LoCoMo numbers above.

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

If your canonical memory corpus is already stored in Qdrant payloads, build the BM25 corpus from the same collection instead of maintaining a separate markdown export. This keeps exact-token lookup aligned with vector search (message IDs, commit hashes, filenames, quoted phrases):

```python
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from mnemostack.recall import BM25Retriever

client = QdrantClient(host="localhost", port=6333)
bm25 = BM25Retriever.from_qdrant(
    client,
    "memory",
    scroll_filter=Filter(
        must=[FieldCondition(key="chunk_type", match=MatchValue(value="transcript"))]
    ),
    limit=40_000,
)

hits = bm25.search("MERGED 71706", limit=5)
```

You can also call `bm25_docs_from_qdrant(...)` directly if you want to combine Qdrant payload chunks with local `BM25Doc`s before constructing `BM25Retriever`.

### HTTP server (optional)

If you want mnemostack available to callers that aren't Python — any service written in Node, Go, Rust, or a plain `curl` from a shell script — install the server extra and expose it over HTTP:

```bash
pip install 'mnemostack[server]'
export GEMINI_API_KEY=...
mnemostack serve --provider gemini --collection memory --port 8000
```

`mnemostack serve` binds to `127.0.0.1` by default. Use
`--host 0.0.0.0` only behind your own auth/rate-limit layer.

Endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET`  | `/health`  | Qdrant + Memgraph reachability + config summary |
| `POST` | `/recall`  | Hybrid recall with optional 8-stage pipeline |
| `POST` | `/answer`  | Recall + LLM answer synthesis with citations |
| `POST` | `/feedback` | Explicit click/usefulness feedback for stateful learning |
| `GET`  | `/metrics` | Prometheus scrape endpoint (counters + summary histograms) |
| `GET`  | `/docs`    | Interactive OpenAPI UI |

```bash
curl -s http://localhost:8000/recall \
    -H 'content-type: application/json' \
    -d '{"query": "what did we decide about auth", "limit": 10}' | jq
```

Response shape (abridged):

```jsonc
{
  "query": "what did we decide about auth",
  "results": [
    { "id": "...", "text": "...", "score": 0.72, "source": "notes/...md", "metadata": {} }
  ]
}
```

The `/answer` endpoint adds `{ answer, confidence, sources }` alongside the memories. If the LLM isn't configured, `/answer` returns `503` and `/recall` still works — graceful degradation applies at the HTTP layer too.

Stateful learning is explicit. Start the server with `--auto-record-ior` if you want `/recall` and `/answer` responses to update inhibition-of-return state. Send user actions to `/feedback` to update Q-learning:

```bash
curl -s http://localhost:8000/feedback \
    -H 'content-type: application/json' \
    -d '{"hit_id":"...","signal":"clicked","query":"what did we decide about auth","sources":["vector","bm25"]}' | jq
```

`signal` is one of `useful`, `clicked`, or `irrelevant`; pass the `retrievers` list returned by `/recall` as `sources` so Q-learning can update the right source weights.
The same state update is available from CLI as `mnemostack feedback ...` and from MCP as `mnemostack_feedback`.

For production, front this with whichever reverse proxy you already use (nginx, Caddy, Traefik) and set an auth layer — mnemostack's server does not do auth itself on purpose; the goal is to plug into whatever you already have.

### Knowledge graph (optional)

```python
from mnemostack.graph import GraphStore

graph = GraphStore(uri="bolt://localhost:7687")
graph.add_triple("alice", "works_on", "project-x", valid_from="2024-01-01")
graph.add_triple("alice", "works_on", "project-y", valid_from="2024-07-01")

# Who was alice working on in March?
march_facts = graph.query_triples(subject="alice", as_of="2024-03-15")
```

Current graph facts use the explicit `valid_until="current"` marker. If you
created graph data with an older release, run
`mnemostack graph-migrate-current --dry-run` first, then
`mnemostack graph-migrate-current` to backfill legacy `NULL` markers.

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

## Install for an OpenClaw-style agent

If you run an OpenClaw-style assistant with its own `scripts/recall.sh` or similar recall entry point, you don't have to wire mnemostack in by hand. Ask the assistant to do it and it can follow the steps below verbatim.

(These steps are written against the markdown-backed memory layout popularised by [OpenClaw](https://clawhub.ai) / [OpenClaw](https://github.com/openclaw/openclaw) — the reasoning applies to any assistant whose memory lives in `MEMORY.md` + daily notes.)

**Assistant, read this section and follow the steps if your user asks you to install mnemostack.**

1. **Install the package into whichever Python env your recall scripts use.**

   ```bash
   pip install mnemostack
   ```

   Confirm with `python3 -c "import mnemostack; print(mnemostack.__version__)"`.

2. **Make sure Qdrant and (optionally) Memgraph are reachable.** If either is missing, start them:

   ```bash
   docker run -d -p 6333:6333 --name qdrant qdrant/qdrant:latest
   docker run -d -p 7687:7687 --name memgraph memgraph/memgraph:latest   # optional
   ```

   Check with `curl -sf http://localhost:6333/collections` and `echo 'RETURN 1;' | cypher-shell -a bolt://localhost:7687`.

3. **Populate the vector store with the assistant's existing memory.** Almost every OpenClaw-style assistant stores memory as markdown files — `MEMORY.md`, `memory/*.md` daily notes, maybe `USER.md` / `TOOLS.md`. Those files are the ground truth; Qdrant is just a searchable index on top of them. So "migration" here really means a first-time ingest.

   Two cases:

   **Case A — user has no Qdrant yet (default OpenClaw setup, markdown-only memory).**

   Stand up Qdrant (see step 2), then index the workspace in one command:

   ```bash
   mnemostack index ~/.openclaw/workspace/ \
       --provider gemini \
       --collection memory \
       --chunk-size 800
   ```

   This walks every `*.md` / `*.txt` under the workspace, chunks them, embeds each chunk with the chosen provider, and upserts into Qdrant. For a typical workspace (a few megabytes of notes) this finishes in a minute or two and stays well inside free-tier embedding quotas.

   **Re-running is safe.** `mnemostack index` uses a deterministic id per `(source, offset, content)` chunk, so a second run skips unchanged chunks (no duplicates, no wasted embedding calls) and only re-embeds chunks whose content changed or moved. Use `--recreate` if you deliberately want to wipe and rebuild from scratch.

   Memgraph is optional — skip it for a pure markdown-backed memory and `MemgraphRetriever` just won't be part of the retriever list.

   **Case B — user already has Qdrant / Memgraph populated by another plugin or a prior assistant run.** Don't re-index. mnemostack reads existing collections and graphs as-is; payload like `chat_id`, `memory_class`, `valid_from` is preserved. Just point `VectorStore(collection=...)` at the existing collection name and `MemgraphRetriever(uri=...)` at the existing graph.

   **Critical for both cases:** use the *same embedding model* for the query that was used for the stored points. Mixing providers/models across ingest and query silently returns garbage — no error, just bad results.

4. **Point your recall entry at mnemostack.** Swap your search helper to use the 4-retriever `Recaller` from the "Full stack" example above. If your recall script has a `--unified` flag, route to mnemostack first and fall back to the legacy code path on any exception.

5. **Keep the fallback wired.** Don't delete the existing recall implementation. Route mnemostack as the primary, fall back to the old recall on any exception. A bad upgrade becomes a one-minute rollback, not an outage.

6. **Smoke-test on real queries from the user's memory**, not toy strings. Ask at least one exact-token query (an IP, a ticker, an error string), one semantic query, one temporal query ("what did we decide last March"), and one list query. If any regresses vs the old recall, keep both running in parallel until you understand why.

7. **Write a short note in the user's daily memory** describing what changed: which command was used to ingest, which embedding provider, which collection name, and where the fallback lives. Future sessions need this to avoid re-indexing with a different model and trashing recall quality.

If you're a human operator and you'd rather do this yourself, the same steps apply; treat them as a checklist.

## Design

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design: pipeline stages, Qdrant schema, Memgraph temporal model, consolidation runtime, MCP tools.

### Pipeline state

The 8-stage pipeline can use a small state store between calls (Q-learning weights, inhibition-of-return history, per-document gravity/hub counters). `FileStateStore(path)` persists it to a JSON file. HTTP recall applies existing state and can record inhibition-of-return exposure with `--auto-record-ior`; Q-learning updates only through explicit `/feedback` calls. CLI/MCP recall still apply existing state but do not collect feedback automatically. For deterministic benchmarks, call `build_full_pipeline(enable_stateful_stages=False)` so IoR/Q-learning/curiosity state cannot affect scores. For multi-process servers, implement your own `StateStore` (three methods: `get()`, `set()`, `update()`) backed by Redis or your database.

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
- [x] Text → graph triple extractor helpers (`mnemostack.graph.TripleExtractor`)
- [x] Config file support YAML/JSON (`mnemostack.config`, `mnemostack init`/`config` CLI)
- [x] Async variants for high-throughput servers (`mnemostack.vector.AsyncQdrantStore`)
- [x] Docker compose examples (`examples/docker-compose.yml`)
- [x] Reproducible LoCoMo benchmark harness in-tree (`benchmarks/run_locomo.sh`)
- [x] First-class FastAPI/Starlette service wrapper (`pip install 'mnemostack[server]'`, `mnemostack serve`)
- [x] Async `Recaller.recall_async` and parallel retriever dispatch (proven: 5 concurrent HTTP recalls complete in ~1x single-request wall-clock)
- [x] Benchmarks on longer-horizon synthetic corpora (`benchmarks/synthetic_longhorizon.py`)
- [x] Streaming `Ingestor` API (`mnemostack.ingest`)
- [x] Prometheus `/metrics` endpoint on the HTTP server
- [x] Unicode-aware `MemgraphRetriever` probes (`telegram_id`, handle, `name_lower`)
- [x] Community health: Code of Conduct, Security policy, issue/PR templates
- [x] Per-retriever latency in `/metrics` (`mnemostack_recall_<name>_latency_ms`)
- [x] Weighted RRF fusion (`reciprocal_rank_fusion(weights=[...])`)
- [x] Adaptive per-query-shape weights in `Recaller` (`adaptive_weights=True`)
- [x] `HyDERetriever` (opt-in, not in default `Recaller`)
- [x] Two-pass graph extraction (full + detail) in the agent's graph-sync pipeline
- [x] Progressive Tiers API on `search`/`answer` (`--tier {1,2,3}`, backward-compatible)
- [x] MCP integration guides for Claude Desktop/Code, Cursor, OpenClaw (`integrations/`)
- [x] Silent-zero fix in `TemporalRetriever` + dispatch-by-type filter builder (`0.2.0a1`)

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Contributing

Early days. Issues and PRs welcome once API stabilizes.
