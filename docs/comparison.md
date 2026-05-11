# Memory systems comparison

## Why this page exists

The README has a short comparison table. This page adds context: architecture, trade-offs, operational cost, benchmark caveats, and the kinds of workloads each approach fits.

This is not a universal ranking of product quality. Memory systems make different choices. A simple vector store can be the right answer. A managed memory service can be the right answer. A framework-native memory layer can be the right answer. Mnemostack is designed for one particular slice: durable, self-hosted recall for long-running agents with mixed query patterns.

## Summary table

| System | Retrieval shape | Graph support | Feedback loop | Deployment model | MCP / API support | LoCoMo number to treat as directional |
| --- | --- | --- | --- | --- | --- | --- |
| Mnemostack | Hybrid: vector, BM25, temporal, optional graph, RRF, pipeline, optional reranker | Optional Memgraph temporal graph | Explicit feedback; stateful effects depend on the configured recall path | Self-hosted Python package, HTTP server, or MCP stdio server | MCP, HTTP, Python SDK | 82.5% strict in the README setup |
| Plain vector search | Embeddings + nearest-neighbor similarity | No, unless you build it | No, unless you build it | Self-hosted or managed vector DB | Depends on your wrapper | Usually not directly comparable |
| Hand-rolled RAG memory | Whatever the team builds | Whatever the team builds | Whatever the team builds | Custom | Custom | Usually not directly comparable |
| Mem0 | Managed or self-hosted memory layer; primarily vector, with a graph variant | Available in graph variant | Product-managed memory behavior | Managed service or self-hosted | API-oriented | ~68.5% reported for graph variant |
| Zep | Memory server with vector, entity extraction, and temporal behavior | Entity / relationship tracking | Built into the memory server model | Self-hosted or service-oriented, depending on edition | API-oriented | 58.4% independently replicated |
| Letta | Agent framework with memory tiers | Framework-dependent | Integrated into agent memory policies | Run the Letta framework | Framework API | 74% reported for filesystem agent |
| Hindsight | Research / benchmarked memory approach | Depends on implementation | Depends on implementation | Not positioned here as a drop-in service | Not positioned here as a drop-in service | 78-85% reported range |

Numbers in this table come from the Mnemostack README. They are not all from the same harness, judge model, protocol, or deployment setup.

## Comparison axes

**Retrieval method.** Some systems use plain semantic similarity. Others combine semantic search with lexical search, recency, graph traversal, or LLM reranking. This matters because agent memory queries are mixed: "what did we decide about auth" is different from "find error `E_CONN_RESET_42`" or "what changed after last Friday?"

**Graph support.** A graph helps when relationships matter: people, projects, temporal facts, ownership, dependencies, and multi-hop links. It also adds storage and modeling cost. Not every memory workload needs one.

**Feedback loop.** Some systems treat every retrieval as stateless. Others store usefulness signals, exposure history, or ranking weights. Feedback can improve fit over time, but it also creates state that must be backed up, migrated, and reasoned about.

**Deployment model.** Managed services reduce operational work. Self-hosted systems give more control over data, models, cost, and infrastructure, but someone has to run the stores, pin models, monitor health, and handle backups.

**MCP / API support.** MCP is useful when the memory layer is attached directly to agents such as Claude Desktop, Claude Code, Cursor, or other MCP-capable runtimes. HTTP and SDK interfaces matter when memory is part of an application or service.

**Self-hosted vs managed.** This is often the main product decision. If you need zero infrastructure, a managed service is attractive. If you need local control, custom providers, or private deployment, self-hosted systems are easier to reason about.

**Benchmark methodology.** LoCoMo scores depend on dataset version, judge model, answer-generation prompt, scoring rules, category mix, and whether a result is full aggregate or a selected subset. Treat reported numbers as directional unless they were produced by the same harness with the same settings.

## Systems

### Mnemostack

Mnemostack is a self-hosted memory layer for AI agents. It can run as a Python library, an HTTP service, or an MCP stdio server.

On recall, configured retrievers run in parallel. The default shape uses vector and temporal retrieval, with BM25 and Memgraph when configured. Results are fused with Reciprocal Rank Fusion. When enabled, the 8-stage pipeline can classify queries, rescue exact tokens, dampen hubs, blend freshness, apply inhibition-of-return, add curiosity boosts, use Q-learning weights from a state store, and resurrect graph-linked memories. Stateful stages require the configured state store to be persisted if those effects matter across restarts. An optional LLM reranker can reorder the final candidates.

Strengths:

- Hybrid recall across semantic, lexical, temporal, and graph sources.
- Exact-token rescue for IDs, filenames, handles, errors, versions, ports, tickers, and other strings embeddings can blur.
- Temporal and graph-aware retrieval when Memgraph is configured.
- Self-hosted deployment with control over data, providers, and storage.
- Multiple integration paths: MCP, HTTP, and Python SDK.
- Graceful degradation when optional components such as graph retrieval are unavailable.

Costs:

- Requires Qdrant for vector storage.
- Memgraph is optional, but needed for graph-backed memory and MCP graph tools.
- BM25 needs a mounted or indexed corpus to help with exact-token recall.
- Stateful ranking and feedback introduce state management and backup concerns.
- The hybrid pipeline adds latency compared with a single vector lookup.

Good fit for:

- Long-running agents that need memory across compaction, restarts, and sessions.
- Mixed workloads with semantic questions, exact-token lookups, temporal questions, and relationship-heavy queries.
- Production teams that want a self-hosted memory layer rather than a managed memory service.
- Applications that need MCP, HTTP, or SDK access to the same memory backend.

### Plain vector search: Qdrant, Pinecone, Chroma direct

Plain vector search means embedding documents or messages, storing vectors in a vector database, and retrieving nearest neighbors by similarity.

Strengths:

- Simple mental model.
- Fast and well-understood.
- Easy to operate if the corpus and query pattern are straightforward.
- Strong fit for many document Q&A and basic RAG systems.

Weaknesses:

- Exact strings can be missed or ranked too low.
- Time-aware questions need extra metadata logic.
- Graph relationships require a separate layer.
- Mixed agent-memory queries can become embedding roulette: semantically similar chunks may crowd out the specific fact the agent needs.

Good fit for:

- Simple RAG over documents.
- Homogeneous query patterns.
- Teams that want minimal moving parts and can accept semantic-only recall behavior.

### Hand-rolled RAG memory

Hand-rolled RAG memory is a custom retrieval pipeline built around one team's product, corpus, and constraints. It might combine a vector DB, SQL filters, keyword search, custom prompts, rerankers, or bespoke heuristics.

Strengths:

- Maximum control.
- Can match unusual data models or compliance constraints.
- No dependency on a memory-specific library if your team already owns the retrieval stack.

Weaknesses:

- Maintenance becomes the product team's responsibility.
- Benchmarking is easy to postpone or make non-reproducible.
- Common retrieval problems get rediscovered: chunking, exact-token recall, temporal filters, reranking, feedback, deduplication, and evaluation drift.
- Integrations such as MCP, feedback APIs, graph tools, and deployment guides must be built separately.

Good fit for:

- Teams with constraints no existing library covers.
- Products where retrieval behavior is core intellectual property.
- Environments that already have a mature search platform and evaluation harness.

### Mem0

Mem0 is a managed and self-hosted memory layer for LLM applications. The graph variant adds entity extraction and relationship-aware behavior on top of the core memory API.

At a high level, Mem0 exposes a memory API with vector-backed retrieval in common configurations, and graph capabilities available through graph-oriented variants. In the README comparison, Mem0 graph variant is listed at about 68.5% LoCoMo correct, reported externally.

Strengths:

- Managed option for teams that do not want to run the full memory stack themselves.
- Simple API surface.
- Graph variant for entity-oriented memory.
- Practical choice when operational simplicity matters more than owning every retrieval component.

Weaknesses:

- Less control over internals in managed mode.
- Exact-token and temporal behavior depends on product configuration and may not match a purpose-built hybrid pipeline.
- Reported benchmark numbers are not necessarily from the same judge, protocol, or setup as Mnemostack's README numbers.

Good fit for:

- Teams that want memory as a service.
- Apps where a managed API is preferable to running Qdrant, optional Memgraph, and memory service processes.
- Products that need a straightforward memory layer and can work within Mem0's model.

### Zep

Zep is a memory server for LLM applications. It provides memory management, entity extraction, and temporal behavior around application sessions.

The README lists Zep at 58.4% LoCoMo correct from an independently replicated result. Treat that number as directional, not as a universal statement about every Zep deployment.

Strengths:

- Built-in entity extraction.
- Session-oriented memory model.
- Server-style architecture for LLM applications.
- Useful when the application wants memory tied to user or session history.

Weaknesses:

- Lower directional LoCoMo number in the README's mixed-workload comparison; not a universal statement about every Zep deployment.
- Less focused on exact-token rescue than a hybrid lexical + semantic pipeline.
- Fit depends heavily on whether your memory model is session-centric.

Good fit for:

- Applications that need session-scoped memory.
- Teams that want entity tracking without designing that layer from scratch.
- Systems where memory is part of an app server rather than a standalone recall library.

### Letta, formerly MemGPT

Letta is an agent framework with built-in memory management. It is not just a memory retrieval library; memory is part of the framework's agent runtime and policies.

The README lists Letta filesystem agent at 74% LoCoMo correct, reported externally.

Strengths:

- Integrated agent and memory model.
- Memory tiers and policies are part of the framework.
- Good fit when the team is already building on Letta's abstractions.

Weaknesses:

- Framework-coupled by design.
- Not a drop-in standalone recall layer for arbitrary agents or services.
- Less attractive if you already have your own agent runtime and only want memory retrieval.

Good fit for:

- Teams building directly on the Letta framework.
- Projects that want memory policy and agent runtime in one system.
- Agents where adopting the framework is acceptable.

## Benchmarks and methodology notes

Mnemostack's README reports **82.5% strict accuracy** and **92.2% combined accuracy** on LoCoMo in the current evaluation setup. The same section lists externally reported or independently replicated numbers for other systems:

The table below mixes Mnemostack's own current setup with externally reported or independently replicated numbers for other systems, so it should not be read as a same-harness leaderboard.

| System | LoCoMo number from README |
| --- | --- |
| Mnemostack | 82.5% strict |
| Hindsight | 78-85% reported range |
| Letta filesystem agent | 74% reported |
| Mem0 graph variant | ~68.5% reported |
| Zep | 58.4% independently replicated |

Read these as directional. LoCoMo results can move when any of these change:

- judge model;
- answer-generation model;
- scoring rules;
- dataset version;
- whether empty-ground-truth questions are included;
- whether the number is a full aggregate or a selected category;
- retrieval configuration;
- reranker settings;
- prompt and answer-format constraints.

The README also publishes signal-only scores that remove empty-ground-truth `cat_5` questions. That matters because those questions can inflate the full aggregate under the current scorer. For reproduction steps and protocol notes, see [benchmarks/README.md](../benchmarks/README.md).

## When NOT to use Mnemostack

Use something simpler if you only need semantic search over a small or stable document set. A direct Qdrant, Pinecone, or Chroma integration will be easier to run and reason about.

Use a managed memory service if zero-ops deployment matters more than owning the retrieval stack. Mem0's managed option is a better fit for teams that do not want to operate Qdrant, optional Memgraph, state files, backups, and service health checks.

Use Letta's built-in memory if you are already committed to the Letta framework. Adding a separate memory layer may create more integration work than value.

Use no memory layer if the relevant corpus fits comfortably in the model context and does not need to survive across sessions. Retrieval adds moving parts; it should earn its place.

Avoid Mnemostack for sub-10ms retrieval targets. Hybrid recall, fusion, pipeline stages, and optional LLM reranking are built for recall quality and explainability, not ultra-low-latency key-value lookup.

Be cautious if your team cannot own operational state. Qdrant collections, optional Memgraph data, BM25 corpora, and feedback state all need backup and migration plans in production.

## Related docs

- [README](../README.md)
- [MCP server](mcp.md)
- [Deployment guide](deployment.md)
- [Benchmarks](../benchmarks/README.md)
