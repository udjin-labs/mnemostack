# MCP Server

## Overview

Mnemostack exposes durable agent memory through MCP tools. Run the MCP server locally, connect it to an MCP-capable client, and your agent can search indexed memories, synthesize answers, record feedback, and, when Memgraph is configured, query or write temporal graph facts.

The server is stdio-based and uses lazy initialization. It can start even before Qdrant, Memgraph, or provider credentials are reachable; component failures are returned as structured JSON from the tools instead of crashing the MCP process.

## Installation

```bash
pip install 'mnemostack[mcp]'
```

Run Qdrant for vector storage:

```bash
docker run -p 6333:6333 qdrant/qdrant:latest
```

Optional: run Memgraph to enable graph tools:

```bash
docker run -p 7687:7687 memgraph/memgraph:latest
```

> **Note:** These quickstart containers are ephemeral — data is lost on restart unless you mount volumes. For durable deployments, use persistent storage; see the deployment guide when available.

Index something into the same collection before expecting useful recall:

```bash
export GEMINI_API_KEY=your-key-here
mnemostack index ./notes --provider gemini --collection my-memory --recreate
```

## Running the server

Start the stdio MCP server:

```bash
export GEMINI_API_KEY=your-key-here
mnemostack mcp-serve --provider gemini --collection my-memory
```

Useful flags:

```bash
mnemostack mcp-serve \
  --provider gemini \
  --embedding-model text-embedding-004 \
  --collection my-memory \
  --qdrant http://localhost:6333 \
  --llm gemini \
  --llm-model gemini-2.5-flash \
  --bm25-path ./notes \
  --state-path /tmp/mnemostack-server-state.json
```

Enable graph tools by adding Memgraph:

```bash
mnemostack mcp-serve \
  --provider gemini \
  --collection my-memory \
  --memgraph-uri bolt://localhost:7687
```

The underlying server also reads these environment variables:

```bash
export MNEMOSTACK_COLLECTION=my-memory
export MNEMOSTACK_EMBEDDING=gemini
export MNEMOSTACK_EMBEDDING_MODEL=text-embedding-004
export MNEMOSTACK_LLM=gemini
export MNEMOSTACK_LLM_MODEL=gemini-2.5-flash
export MNEMOSTACK_QDRANT_HOST=http://localhost:6333
export MNEMOSTACK_MEMGRAPH_URI=bolt://localhost:7687
export MNEMOSTACK_GRAPH_TIMEOUT=5.0
export MNEMOSTACK_BM25_PATHS="./notes:./docs"
export MNEMOSTACK_STATE_PATH=/tmp/mnemostack-server-state.json
```

## Client configuration

### Claude Desktop

Add a server entry to your Claude Desktop MCP config.

macOS path:

```text
~/Library/Application Support/Claude/claude_desktop_config.json
```

Example:

```json
{
  "mcpServers": {
    "mnemostack": {
      "command": "mnemostack",
      "args": [
        "mcp-serve",
        "--provider", "gemini",
        "--collection", "my-memory",
        "--qdrant", "http://localhost:6333"
      ],
      "env": {
        "GEMINI_API_KEY": "your-key-here"
      }
    }
  }
}
```

Restart Claude Desktop after editing the config. Ask Claude to call `mnemostack_health` first.

### Claude Code

Add Mnemostack as a stdio MCP server:

```bash
claude mcp add mnemostack -- \
  mnemostack mcp-serve \
    --provider gemini \
    --collection my-memory \
    --qdrant http://localhost:6333
```

If the provider needs credentials, make sure they are available in the environment that launches Claude Code:

```bash
export GEMINI_API_KEY=your-key-here
claude
```

### Cursor

Add a stdio MCP server in Cursor settings. The exact UI changes between Cursor versions, but the server shape is:

```json
{
  "mcpServers": {
    "mnemostack": {
      "command": "mnemostack",
      "args": [
        "mcp-serve",
        "--provider", "gemini",
        "--collection", "my-memory",
        "--qdrant", "http://localhost:6333"
      ],
      "env": {
        "GEMINI_API_KEY": "your-key-here"
      }
    }
  }
}
```

After saving, reload Cursor and check that `mnemostack_health`, `mnemostack_search`, `mnemostack_answer`, and `mnemostack_feedback` appear in the MCP tools list.

### Other MCP clients

Use stdio transport:

```json
{
  "command": "mnemostack",
  "args": ["mcp-serve", "--provider", "gemini", "--collection", "my-memory"],
  "env": {
    "GEMINI_API_KEY": "your-key-here"
  }
}
```

For clients that do not support per-server `env`, export variables before starting the client process.

## Available tools

### `mnemostack_health`

**Purpose:** Check health of mnemostack components: embedding provider, vector store, and optional graph.

**Input parameters:** None.

**Return shape:**

```json
{
  "ok": true,
  "components": {
    "embedding": {
      "ok": true,
      "provider": "gemini",
      "dimension": 768,
      "message": "..."
    },
    "vector": {
      "ok": true,
      "collection": "my-memory",
      "exists": true,
      "points": 1234
    },
    "graph": {
      "ok": true,
      "nodes": 42,
      "edges": 57,
      "message": "..."
    }
  }
}
```

`components.graph` is present only when `MNEMOSTACK_MEMGRAPH_URI`, `MNEMOSTACK_GRAPH_URI`, or `--memgraph-uri` is configured.

On failure, a component contains `{"ok": false, "error": "..."}` and top-level `ok` becomes `false`.

**Example usage scenario:** At the start of an agent session, call `mnemostack_health` to verify that the configured collection exists and that provider credentials are valid.

### `mnemostack_search`

**Purpose:** Hybrid recall over indexed memories. Returns top-K results ranked by reciprocal rank fusion across vector recall, temporal recall, optional BM25, and optional Memgraph.

**Input parameters:**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `query` | `string` | Required | Natural-language or exact-token query. |
| `limit` | `integer` | `10` | Maximum number of recall hits to return. |

**Return shape:**

```json
{
  "ok": true,
  "query": "what did we decide about auth?",
  "count": 2,
  "results": [
    {
      "id": "2f6a...",
      "text": "We decided to keep OAuth behind the API gateway...",
      "score": 0.0328,
      "sources": ["vector", "bm25"],
      "payload": {
        "source": "notes/auth.md",
        "offset": 120,
        "ts": "2026-04-20T10:15:00Z"
      }
    }
  ]
}
```

On failure:

```json
{
  "ok": false,
  "error": "...",
  "query": "what did we decide about auth?"
}
```

**Example usage scenario:** Before editing an old subsystem, search `"what did we decide about auth gateway"` and read the top 3 results before proposing changes.

### `mnemostack_answer`

**Purpose:** Generate a concise factual answer from retrieved memories. It uses hybrid recall, then an LLM layer to synthesize a short answer with confidence and citations.

**Input parameters:**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `query` | `string` | Required | Question to answer from memory. |
| `limit` | `integer` | `10` | Number of memories to retrieve before answer synthesis. |

**Return shape:**

```json
{
  "ok": true,
  "query": "Which database did we pick for graph memory?",
  "answer": "We picked Memgraph, with temporal facts stored as subject-predicate-object triples.",
  "confidence": 0.86,
  "sources": ["architecture-notes.md"],
  "fallback_recommended": false,
  "error": null
}
```

If the answer generator cannot produce a reliable answer, `ok` may be `false`, `confidence` may be low, and `fallback_recommended` may be `true`. On exceptions, the tool returns:

```json
{
  "ok": false,
  "error": "...",
  "query": "Which database did we pick for graph memory?"
}
```

**Example usage scenario:** Ask `"What deployment steps did we use last time for the staging agent?"` after a context compaction, then follow the citations if confidence is low.

### `mnemostack_feedback`

**Purpose:** Record explicit feedback into the configured state store for Q-learning weights and inhibition-of-return tracking. In the default MCP server configuration, feedback is persisted to the state file but the default `mnemostack_search` does not apply stateful pipeline stages. This tool is most useful for deployments that wire the same state store into their recall pipeline, and for observability (tracking which results agents find useful).

**Input parameters:**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `hit_id` | `string` | Required | Result id from `mnemostack_search` or a cited source id. |
| `signal` | `string` | Required | One of `useful`, `irrelevant`, or `clicked`. |
| `query` | nullable `string` | `null` | Original query. Used to infer query type when `query_type` is not supplied. |
| `query_type` | nullable `string` | `null` | Explicit query type override, such as `general`, `exact_token`, `person`, or `temporal`. |
| `source` | nullable `string` | `null` | Single retriever/source label to update. |
| `sources` | nullable `string[]` | `null` | Multiple retriever/source labels, usually copied from a search result's `sources`. |
| `reward` | nullable `number` | `null` | Optional override in `[0.0, 1.0]`. Defaults: `useful=1.0`, `clicked=0.7`, `irrelevant=0.0`. |

Use `sources` when copying labels from a `mnemostack_search` result. Use `source` only when you want to update one explicit retriever label.

**Return shape:**

```json
{
  "ok": true,
  "hit_id": "2f6a...",
  "signal": "useful",
  "reward": 1.0,
  "query_type": "general",
  "ior_recorded": false,
  "q_learning_updates": 2
}
```

Validation failures return structured errors:

```json
{
  "ok": false,
  "error": "signal must be one of: useful, irrelevant, clicked"
}
```

**Example usage scenario:** After the agent uses a result from `mnemostack_search`, call feedback with the hit id, `signal: "useful"`, the original query, and the result's `sources` array. Note: in the default MCP server, this records the signal for observability and external pipeline consumers; it does not directly alter subsequent `mnemostack_search` ranking unless the deployment wires the state store into the Recaller pipeline.

### `mnemostack_graph_query`

**Availability:** Only registered when Memgraph is configured with `--memgraph-uri`, `MNEMOSTACK_MEMGRAPH_URI`, or `MNEMOSTACK_GRAPH_URI`.

**Purpose:** Query the knowledge graph with optional subject-predicate-object filters and point-in-time validity.

**Input parameters:**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `subject` | nullable `string` | `null` | Optional subject filter. |
| `predicate` | nullable `string` | `null` | Optional predicate filter. |
| `obj` | nullable `string` | `null` | Optional object filter. |
| `as_of` | nullable `string` | `null` | Optional ISO date. Returns only facts valid at that date. |
| `limit` | `integer` | `50` | Maximum number of triples to return. |

**Return shape:**

```json
{
  "ok": true,
  "count": 1,
  "triples": [
    {
      "subject": "alice",
      "predicate": "works_on",
      "obj": "mnemostack",
      "valid_from": "2026-04-01",
      "valid_until": null
    }
  ]
}
```

On failure:

```json
{
  "ok": false,
  "error": "..."
}
```

**Example usage scenario:** Query `subject: "alice", predicate: "works_on", as_of: "2026-05-01"` to find what Alice was working on at a specific date.

### `mnemostack_graph_add_triple`

**Availability:** Only registered when Memgraph is configured with `--memgraph-uri`, `MNEMOSTACK_MEMGRAPH_URI`, or `MNEMOSTACK_GRAPH_URI`.

**Purpose:** Add a temporal fact `(subject, predicate, object)` to the graph. Nodes are created on demand.

**Input parameters:**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `subject` | `string` | Required | Subject node, such as a person, project, or service. |
| `predicate` | `string` | Required | Relationship name, such as `works_on`, `owns`, `depends_on`, or `prefers`. |
| `obj` | `string` | Required | Object node or value. |
| `valid_from` | nullable `string` | `null` | Optional ISO date when the fact starts being valid. |
| `valid_until` | nullable `string` | `null` | Optional ISO date when the fact stops being valid. |

**Return shape:**

```json
{
  "ok": true,
  "subject": "alice",
  "predicate": "works_on",
  "obj": "mnemostack"
}
```

On failure:

```json
{
  "ok": false,
  "error": "..."
}
```

**Example usage scenario:** When a project handoff happens, write `subject: "alice", predicate: "owns", obj: "billing-api", valid_from: "2026-05-11"` so future temporal graph queries can recover ownership.

## Recommended agent instructions

Add a short instruction block to agents that can use the MCP tools:

```text
You have access to Mnemostack durable memory.

Use mnemostack_search when the answer may depend on prior sessions, project decisions,
user preferences, exact identifiers, incidents, or anything not present in the current
context. Use mnemostack_answer for concise factual answers grounded in memory. Prefer
mnemostack_search when you need to inspect raw sources or exact wording.

After using a memory result, call mnemostack_feedback to record usefulness signals for observability and for deployments that apply stateful recall stages:
- signal="useful" when a result helped the final answer or action
- signal="irrelevant" when a result looked related but was not useful
- signal="clicked" when you inspect a result and want to record exposure

Do not invent memories. If recall is empty or confidence is low, say so and ask a
focused follow-up or proceed from current context only.
```

If graph tools are available, add:

```text
Use mnemostack_graph_query for relationship or point-in-time questions, such as who
owned a service on a date. Use mnemostack_graph_add_triple only for explicit facts
from the user, project docs, or tool output; do not write inferred facts as truth.
```

## Common workflows

### Project recall

1. Search the project name plus the decision or component.
2. Read the top hits.
3. Use feedback on the hit that informed the answer.

Example:

```json
{
  "query": "mnemostack auth gateway decision",
  "limit": 5
}
```

### Compaction recovery

After an agent context compaction or restart, ask broad orientation questions first:

```json
{
  "query": "current status blockers and next steps for billing-api migration",
  "limit": 10
}
```

Then narrow to exact decisions:

```json
{
  "query": "billing-api migration final database choice",
  "limit": 5
}
```

### Personal preferences lookup

Use `mnemostack_search` before making style, tooling, communication, or workflow assumptions:

```json
{
  "query": "user preference for PR review style and commit messages",
  "limit": 5
}
```

### Exact-token lookup: IPs, IDs, error strings

Use the exact token in the query. If BM25 paths are configured, exact-token retrieval can rescue strings that embeddings may blur.

```json
{
  "query": "10.42.7.19 ECONNRESET invoice-worker",
  "limit": 10
}
```

### Temporal questions

Ask with explicit dates when possible:

```json
{
  "query": "who owned billing-api on 2026-04-15",
  "limit": 10
}
```

If graph tools are enabled, query the graph directly:

```json
{
  "subject": "billing-api",
  "predicate": "owned_by",
  "as_of": "2026-04-15",
  "limit": 10
}
```

### Feedback after useful or irrelevant recall

Useful result:

```json
{
  "hit_id": "2f6a...",
  "signal": "useful",
  "query": "what did we decide about auth",
  "sources": ["vector", "bm25"]
}
```

Irrelevant result:

```json
{
  "hit_id": "9b18...",
  "signal": "irrelevant",
  "query": "what did we decide about auth",
  "sources": ["vector"]
}
```

Clicked/read result:

```json
{
  "hit_id": "2f6a...",
  "signal": "clicked",
  "query": "what did we decide about auth",
  "sources": ["vector", "bm25"]
}
```

## Environment variables

These are the MCP-relevant environment variables read by `mnemostack mcp-serve` / `mnemostack.mcp.server`.

| Variable | Default | Description |
| --- | --- | --- |
| `MNEMOSTACK_COLLECTION` | `mnemostack` | Qdrant collection name. Alias for `MNEMOSTACK_VECTOR_COLLECTION`. |
| `MNEMOSTACK_EMBEDDING` | `gemini` | Embedding provider. Alias of `MNEMOSTACK_EMBEDDING_PROVIDER` / `MNEMOSTACK_PROVIDER`. |
| `MNEMOSTACK_EMBEDDING_MODEL` | Provider default | Embedding model override. |
| `MNEMOSTACK_LLM` | `gemini` | LLM provider for answer generation. Alias of `MNEMOSTACK_LLM_PROVIDER`. |
| `MNEMOSTACK_LLM_MODEL` | Provider default | LLM model override. |
| `MNEMOSTACK_QDRANT_HOST` | `http://localhost:6333` | Qdrant URL. Alias of `MNEMOSTACK_VECTOR_HOST` / `MNEMOSTACK_QDRANT_URL`. |
| `MNEMOSTACK_MEMGRAPH_URI` | unset | Memgraph Bolt URI. If unset, graph tools are not registered. Alias of `MNEMOSTACK_GRAPH_URI`. |
| `MNEMOSTACK_GRAPH_TIMEOUT` | `5.0` | Memgraph operation timeout in seconds. |
| `MNEMOSTACK_BM25_PATHS` | unset | File or directory paths for BM25 exact-token retrieval, separated by `os.pathsep` (`:` on Unix, `;` on Windows). |
| `MNEMOSTACK_STATE_PATH` | `/tmp/mnemostack-server-state.json` | JSON state file for feedback and stateful recall stages. |

Common provider variables:

| Variable | Used for |
| --- | --- |
| `GEMINI_API_KEY` | Gemini embeddings and Gemini LLM. |
| `OLLAMA_HOST` | Ollama embeddings or LLM, defaulting to `http://localhost:11434` in typical Ollama setups. |

## Troubleshooting

### Tool not visible in agent

- Confirm the client is configured for stdio with `command: "mnemostack"` and `args: ["mcp-serve", ...]`.
- Restart or reload the MCP client after changing config.
- Run the command manually in a terminal. If it exits immediately, fix the printed error first.
- Make sure you installed the MCP extra: `pip install 'mnemostack[mcp]'`.

### Missing API key

Symptoms: `mnemostack_health` reports `components.embedding.ok=false`, or `mnemostack_answer` returns an LLM/provider error.

Fix: export the provider key in the same environment that launches the MCP client:

```bash
export GEMINI_API_KEY=your-key-here
```

For desktop apps, put the key in the server `env` block if the client supports it.

### Qdrant unreachable

Symptoms: `components.vector.ok=false`, connection refused, timeout, or DNS errors.

Fix:

```bash
docker run -p 6333:6333 qdrant/qdrant:latest
mnemostack health --provider gemini --collection my-memory --qdrant http://localhost:6333
```

If Qdrant runs remotely, set `--qdrant`, `MNEMOSTACK_QDRANT_HOST`, or `MNEMOSTACK_QDRANT_URL` to the reachable URL.

### Wrong collection

Symptoms: health is green, but `points` is `0` or search returns unrelated memories.

Fix: use the same collection name for indexing and MCP serving:

```bash
mnemostack index ./notes --provider gemini --collection my-memory --recreate
mnemostack mcp-serve --provider gemini --collection my-memory
```

Call `mnemostack_health` and check `components.vector.collection` and `components.vector.points`.

### Embedding provider mismatch

Symptoms: search quality is poor, or Qdrant errors mention vector dimensions.

Cause: the collection was indexed with one embedding provider/model and served with another provider/model with a different dimension.

Fix: serve with the same provider/model used for indexing, or recreate the collection:

```bash
mnemostack index ./notes \
  --provider gemini \
  --embedding-model text-embedding-004 \
  --collection my-memory \
  --recreate

mnemostack mcp-serve \
  --provider gemini \
  --embedding-model text-embedding-004 \
  --collection my-memory
```

### Graph tools not appearing: Memgraph not configured

`mnemostack_graph_query` and `mnemostack_graph_add_triple` are registered only when a Memgraph URI is configured at server startup.

Fix:

```bash
docker run -p 7687:7687 memgraph/memgraph:latest
mnemostack mcp-serve \
  --provider gemini \
  --collection my-memory \
  --memgraph-uri bolt://localhost:7687
```

Then restart the MCP client. If the tools still do not appear, call `mnemostack_health` and check `components.graph`.

### Search finds memories, but answer is poor

Symptoms: `mnemostack_search` returns relevant memories, but `mnemostack_answer` gives a weak answer, low confidence, missing citations, or says it cannot answer.

Cause: retrieval is working; the issue is above retrieval — answer synthesis, LLM provider behavior, prompt fit, reranking, context budget, or confidence policy.

Fix:

- Inspect the raw `mnemostack_search` results first to confirm retrieval quality.
- Increase `limit` for `mnemostack_answer` to give the LLM more context.
- Check that `MNEMOSTACK_LLM`, `--llm`, and provider credentials are correctly configured.
- Try a more specific query.
- If the raw memories are correct, prefer using `mnemostack_search` results directly in the agent prompt instead of treating this as an indexing problem.

### Search returns empty after indexing

Symptoms: indexing succeeds and `mnemostack_health` shows `points > 0`, but `mnemostack_search` returns no useful hits.

Check:

- Use the same `--collection` for indexing and MCP serving.
- Use the same `--provider` and `--embedding-model` for indexing and querying. Mismatched embedding models produce incompatible vectors.
- Make sure `--qdrant`, `MNEMOSTACK_QDRANT_HOST`, or `MNEMOSTACK_QDRANT_URL` points to the same Qdrant instance used during indexing.
- Try a broader query.
- Configure `--bm25-path` / `MNEMOSTACK_BM25_PATHS` if you rely on exact-token lookup for IDs, IPs, filenames, or error strings.
