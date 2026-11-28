# Codex

Wire mnemostack into the Codex CLI as an MCP server.

## Install

1. Make sure the binary and a working backend are ready:

   ```bash
   pip install 'mnemostack[mcp]'
   mnemostack health --provider gemini
   ```

2. Add the server. Codex stores it in `~/.codex/config.toml`:

   ```bash
   codex mcp add mnemostack --env GEMINI_API_KEY=your-key-here -- \
     mnemostack mcp-serve --provider gemini --collection my-memory
   ```

   The resulting entry:

   ```toml
   [mcp_servers.mnemostack]
   command = "mnemostack"
   args = ["mcp-serve", "--provider", "gemini", "--collection", "my-memory"]

   [mcp_servers.mnemostack.env]
   GEMINI_API_KEY = "your-key-here"
   ```

   Swap `--provider gemini` for `ollama` or `huggingface` if that's what
   `health` validated. For graph tools, add `--memgraph-uri bolt://localhost:7687`.

3. For a deployment on another host, connect to its authenticated HTTP
   endpoint (`mnemostack mcp-serve --transport http --auth`) instead. Codex
   reads the service key from the named environment variable:

   ```bash
   codex mcp add mnemostack --url https://memory.example.com/mcp \
     --bearer-token-env-var MNEMOSTACK_API_KEY
   ```

4. For a collection built with a custom embedding provider, use the launcher
   from [docs/mcp.md](../../docs/mcp.md#existing-collection-with-a-custom-embedding-provider):

   ```bash
   codex mcp add mnemostack -- /path/to/venv/bin/python /path/to/mnemostack_mcp.py
   ```

5. Start a new Codex session.

## Verify

`codex mcp list` shows the entry as enabled. In a new session, ask Codex:

> List the MCP tools you have available from mnemostack.

You should see `mnemostack_health`, `mnemostack_search`, `mnemostack_answer`,
`mnemostack_feedback`, `mnemostack_resolve`, `mnemostack_invalidate`, and
`mnemostack_remember` (plus the graph tools if a Memgraph URI was passed). Then:

> Use mnemostack to search for "VPN failover" with limit 1.

The reply should include at most one result with its text and sources.

## Uninstall

```bash
codex mcp remove mnemostack
```

No other state lives on the host side. The vector store and the graph are
owned by the mnemostack process, not by the integration.

## Troubleshooting

- **Tools don't appear**: run `codex mcp list` and check that the entry is
  enabled. If `mnemostack` is not on the PATH of the Codex process, use an
  absolute path as the command.
- **`fastmcp not installed`**: the `[mcp]` extra is missing:
  `pip install 'mnemostack[mcp]'`.
- **`Unknown embedding provider`**: the collection uses a custom provider.
  Use the launcher from step 4; the provider must be registered in the MCP
  process itself.
- **`health` says DOWN**: the MCP server will start but every tool call will
  fail until Qdrant and the embedding provider are reachable.
