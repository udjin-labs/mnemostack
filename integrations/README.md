# mnemostack integrations

MCP (Model Context Protocol) integrations for popular agent hosts. Each
sub-directory has a self-contained `SKILL.md` with install, verify, and
uninstall instructions.

## Which setup to use

Start with the built-in server. It already exposes the native tools
(`mnemostack_search`, `mnemostack_remember` and the others below), so no tool
has to be written.

| Your deployment | Use |
|---|---|
| Built-in embedding provider (`gemini`, `ollama`, `huggingface`, `openai`), client on a host that reaches Qdrant | `mnemostack mcp-serve` over stdio. See the host table below |
| Collection that still requires a custom embedding provider | The launcher in [Existing collection with a custom embedding provider](../docs/mcp.md#existing-collection-with-a-custom-embedding-provider). The MCP process registers the provider even when the HTTP service already uses it |
| Deployment on another host, several clients | `mnemostack mcp-serve --transport http --auth` next to the stores, and each client connects to `/mcp` with its own service key, see [docs/mcp.md: Authenticated HTTP](../docs/mcp.md#authenticated-http) |

### Remote deployments: keep the native tools

A client gets the native tool names and schemas only from the MCP server
itself. For a remote deployment, run the server with the Streamable HTTP
transport on the deployment host, and give each client the endpoint URL and a
Bearer service key (`mnemostack keys add`):

```bash
# on the deployment host
mnemostack mcp-serve --transport http --auth --host 127.0.0.1 --port 8001

# on each client
claude mcp add --transport http mnemostack https://memory.example.com/mcp \
  --header "Authorization: Bearer $MNEMOSTACK_API_KEY"
codex mcp add mnemostack --url https://memory.example.com/mcp \
  --bearer-token-env-var MNEMOSTACK_API_KEY
```

The transport carries the MCP protocol unchanged, so the client lists the
same tools with the same schemas as over stdio. A deployment with a custom
embedding provider runs the launcher from docs/mcp.md with the same
transport.

The REST API of `mnemostack serve` (`/recall`, `/memories`) is for
applications. Wrapping it in MCP tools of your own gives an agent other tool
names and other schemas than the ones the documentation and the agent
instructions describe. Do that only for a requirement that the built-in
server cannot meet.

## Hosts

| Host | Path | Transport |
|---|---|---|
| Claude Desktop / Claude Code | [`claude-code/`](claude-code/SKILL.md) | stdio |
| Codex | [`codex/`](codex/SKILL.md) | stdio, HTTP |
| Cursor | [`cursor/`](cursor/SKILL.md) | stdio |
| OpenClaw | [`openclaw/SKILL.md`](openclaw/SKILL.md) | stdio |

All of them use the same `mnemostack mcp-serve` binary. They differ only in
where the host's MCP config lives and how it discovers servers.

## Prerequisites (all hosts)

```bash
pip install 'mnemostack[mcp]'
mnemostack health --provider gemini  # or ollama, huggingface, openai
```

If `health` passes and you have indexed content (`mnemostack index ./notes/`),
the integration only wires the host up to the existing server process.
For OpenAI embeddings, set `OPENAI_API_KEY` in the MCP server's environment.

## Verify from a fresh session

1. The tool list holds `mnemostack_search` and `mnemostack_remember`.
2. `mnemostack_search` returns a record that you know is in the collection.
3. `mnemostack_remember` stores a test fact.
4. A second, new session finds that fact with `mnemostack_search`.

## What the MCP server exposes

Tools are prefixed with `mnemostack_` in the wire protocol:

- `health` — config + reachability summary
- `search` — hybrid recall over indexed content
- `answer` — LLM-synthesized answer with confidence + source citations
- `feedback` — record explicit click/usefulness feedback into the stateful pipeline
- `resolve`: verify a citation against its source
- `invalidate`: retract stored memories
- `remember`: store a memory
- `graph_query` *(only when `--memgraph-uri` is set)* — point-in-time Cypher over the graph
- `graph_add_triple` *(only when `--memgraph-uri` is set)* — add a temporal fact

`search` and `answer` accept `limit` and `token_budget` to bound output size;
`search` returns full result text and neither tool accepts `tier`.
