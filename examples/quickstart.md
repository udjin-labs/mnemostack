# mnemostack quickstart

Retrieve a fact from a synthetic message, inspect its source, then connect an
agent to the same collection. This path uses local Ollama embeddings with
`nomic-embed-text` and the `archive-demo` collection throughout. No API key or
Memgraph is needed. Pull both models: `llama3.2:3b` is used for search reranking
and for optional answer generation, while `nomic-embed-text` creates embeddings.

## 1. Prerequisites and installation

Use Python 3.10+, Docker, and an installed, running Ollama service on
`http://127.0.0.1:11434`. These commands assume a POSIX shell and a fresh demo
directory. Use a fresh Python environment and default mnemostack configuration;
existing config files and `MNEMOSTACK_*` variables can change retrieval behavior.
Choose an unused collection name if `archive-demo` already contains data, and
substitute that name in every command and MCP configuration below.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install 'mnemostack[mcp]'
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

## 2. Start local storage

```bash
# Qdrant 1.20+ is required. Match qdrant-client: same major, minor within 1.
docker run -d --name mnemostack-demo-qdrant \
  -p 127.0.0.1:6333:6333 \
  -v mnemostack-demo-qdrant:/qdrant/storage \
  qdrant/qdrant:v1.20.0
```

Wait for Qdrant to be ready before indexing. If a compatible Qdrant already runs
at this address, reuse it and skip the container command.

```bash
curl --fail http://127.0.0.1:6333/collections
export MNEMOSTACK_GRAPH_URI=""
```

Keep services local for this walkthrough. The full development Compose example
is separate: [docker-compose.yml](docker-compose.yml) builds `../Dockerfile`
and mounts `./notes`, so downloading that YAML alone is not a working setup.
It requires the repository checkout and a review of its service bindings before use.

## 3. Index one synthetic message

```bash
mkdir -p demo-notes
cat > demo-notes/workshop-chat.txt <<'MESSAGE'
Synthetic messenger export. No real people or messages.
2026-10-01 09:15 | Mira | message demo-001
For the Cedar workshop, the equipment pickup code is CEDAR-482.
MESSAGE

mnemostack index ./demo-notes/ \
  --provider ollama --embedding-model nomic-embed-text \
  --ollama-host http://127.0.0.1:11434 \
  --qdrant http://127.0.0.1:6333 --collection archive-demo
```

This explicitly imports the file; it does not watch a messenger or capture future
messages. Keep the same embedding model for indexing and querying. `--recreate`
is unnecessary here and would drop the existing collection.

## 4. Retrieve the fact and its source

```bash
mnemostack search "What is the equipment pickup code for the Cedar workshop?" \
  --provider ollama --embedding-model nomic-embed-text \
  --ollama-host http://127.0.0.1:11434 \
  --qdrant http://127.0.0.1:6333 --collection archive-demo \
  --llm ollama --llm-model llama3.2:3b --rerank-mode full_reorder --json
```

Check the returned `text` for `CEDAR-482` and `payload.source` for
`workshop-chat.txt`. The message identifier `demo-001` is in the text, not a
separately parsed messenger field. The top-level `sources` field lists retrieval
methods, such as `vector`; it is not the original filename. These are the evidence fields to check in your run.

This walkthrough explicitly uses `--rerank-mode full_reorder`, which asks the
LLM to order all candidate IDs. In the local smoke-test, `llama3.2:3b` with the
default `relevant_only` mode returned the answer text instead of IDs, triggering
a reranker fallback. `full_reorder` produced valid IDs for this synthetic case;
this does not guarantee ranking quality on a larger archive.

For compact agent output, add `--tier 1`, `--tier 2`, or `--tier 3` to search.
Tier 1 omits text, tier 2 includes short snippets, and tier 3 includes longer
previews. Omit `--tier` when inspecting full text and payload source evidence.

## 5. Optional: generate an answer

Ask about the same indexed fact using the LLM already pulled in step 1:

```bash
mnemostack answer "What is the equipment pickup code for the Cedar workshop?" \
  --provider ollama --embedding-model nomic-embed-text \
  --ollama-host http://127.0.0.1:11434 \
  --qdrant http://127.0.0.1:6333 --collection archive-demo \
  --llm ollama --llm-model llama3.2:3b --rerank-mode full_reorder
```

The expected fact is `CEDAR-482`, supported by the workshop message. Inspect the
answer's sources rather than treating a plausible sentence as proof of recall.
LLM wording and results can vary. `answer --tier 1` hides the sources block, so
leave it off for this check. CLI search and MCP search also use the configured
LLM for reranking; a missing or failing reranker can fall back to retrieval
results, but that fallback is not a successful test of the configured path.

## 6. Connect an MCP client

After indexing, add this to your MCP client's configuration. Replace `command`
with the absolute path to `mnemostack` in the virtual environment if the client
does not inherit your shell's PATH. The client launches the stdio server; there
is no separate HTTP listener to start for this configuration.

```json
{
  "mcpServers": {
    "mnemostack": {
      "command": "mnemostack",
      "args": [
        "mcp-serve", "--transport", "stdio",
        "--provider", "ollama", "--embedding-model", "nomic-embed-text",
        "--ollama-host", "http://127.0.0.1:11434",
        "--qdrant", "http://127.0.0.1:6333", "--collection", "archive-demo",
        "--llm", "ollama", "--llm-model", "llama3.2:3b",
        "--rerank-mode", "full_reorder"
      ],
      "env": {"MNEMOSTACK_GRAPH_URI": ""}
    }
  }
}
```

Ask the agent to call `mnemostack_search` for the Cedar workshop pickup code and
include the retrieved filename and message identifier in its reply.
`mnemostack_answer` uses the same model. MCP search can call that model for
reranking even if you never request an answer. Inspect `degraded` in MCP results
(and use `include_trace: true` for diagnostics) rather than treating a fallback
as proof that the model was available.
`mnemostack_remember` is also available for explicit writes; this demonstration
only reads the message you indexed. Automatic capture or recall injection is
integration-specific and is not enabled merely by adding this MCP configuration.

Next: try the [same archive from two local clients](two-local-clients-demo.md), or read the
[HTTP API and tenant-auth setup](../README.md#http-server-optional) for a service
with separate consumers.
