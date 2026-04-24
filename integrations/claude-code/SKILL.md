# Claude Desktop / Claude Code

Wire mnemostack into Claude Desktop (and by extension Claude Code, which
reads the same MCP config) as an MCP server.

## Install

1. Make sure the binary and a working backend are ready:

   ```bash
   pip install 'mnemostack[mcp]'
   mnemostack health --provider gemini
   ```

2. Locate the Claude Desktop MCP config:

   - macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
   - Linux (unofficial/Claude Code): `~/.config/claude/claude_desktop_config.json`

3. Merge this entry into the `mcpServers` block (create the file if needed):

   ```json
   {
     "mcpServers": {
       "mnemostack": {
         "command": "mnemostack",
         "args": [
           "mcp-serve",
           "--provider", "gemini",
           "--collection", "my-memory"
         ],
         "env": {
           "GEMINI_API_KEY": "your-key-here"
         }
       }
     }
   }
   ```

   Swap `--provider gemini` for `ollama` or `openai` if that's what `health`
   validated. For graph tools, add `--memgraph-uri bolt://localhost:7687`.

4. Restart Claude Desktop (fully quit, not just close the window).

## Verify

In a new chat, ask Claude:

> List the MCP tools you have available from mnemostack.

You should see `search` and `answer` (plus `graph_query` if Memgraph URI was
passed). Then:

> Use mnemostack to search for "VPN failover" with tier 1.

The reply should cite sources and not leak any snippets (tier 1 = list only).

## Uninstall

Remove the `mnemostack` entry from `mcpServers` in the config file and restart
Claude Desktop. No other state lives on the host side — the vector store and
the memgraph are owned by the mnemostack process, not by the integration.

## Troubleshooting

- **Tools don't appear**: check Claude Desktop's MCP logs
  (`~/Library/Logs/Claude/mcp*.log` on macOS). A common cause is `mnemostack`
  not being on PATH for the GUI app — use an absolute path in `command`.
- **`fastmcp not installed`**: the `[mcp]` extra is missing:
  `pip install 'mnemostack[mcp]'`.
- **`health` says DOWN**: the MCP server will start but every tool call will
  fail until Qdrant/embedding provider is reachable.
