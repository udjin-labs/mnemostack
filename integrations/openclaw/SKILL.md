# OpenClaw

Wire mnemostack into OpenClaw as an MCP server, alongside the native OpenClaw
memory tools (`memory_search` / `memory_get`). The two coexist — mnemostack
handles the hybrid pipeline (Vector + BM25 + Memgraph + Temporal + 8-stage
rerank), while the native tools do fast file-scoped retrieval.

## Install

1. Binary and backend:

   ```bash
   pip install 'mnemostack[mcp]'
   mnemostack health --provider gemini
   ```

2. Open OpenClaw's config (`~/.openclaw/openclaw.json` on the current host) and
   add an MCP server under the `mcp` plugin block. Exact schema path:
   `plugins.entries.mcp.config.servers` (confirm the current field name with
   `config.schema.lookup mcp.servers` — OpenClaw's schema evolves).

   ```json
   {
     "mnemostack": {
       "command": "mnemostack",
       "args": [
         "mcp-serve",
         "--provider", "gemini",
         "--collection", "memory",
         "--memgraph-uri", "bolt://localhost:7687"
       ],
       "env": {
         "GEMINI_API_KEY": "${env:GEMINI_API_KEY}"
       }
     }
   }
   ```

3. Apply config and restart OpenClaw gateway:

   ```bash
   # Restart takes ~1 minute on this VPS — don't poll too early.
   systemctl --user restart openclaw-gateway
   ```

## Verify

After gateway restart, in any OpenClaw session:

> List the MCP servers you can see.

Expected: `mnemostack` among the available servers. Then:

> Call mnemostack's search tool with query "VPN failover" and tier 1.

You should get id+score+source list, no text. The existing
`scripts/recall-selfeval.sh` and the `auto-recall-mnemostack` plugin are
**unaffected** — they call mnemostack via its Python API directly, not via
MCP. The MCP route is for cases where an outer agent host wants tool-style
access.

## When to use which

- **Native `memory_search` / `memory_get`** — cheap file-scoped lookups in
  MEMORY.md and memory/*.md, already in OpenClaw's system prompt.
- **`scripts/recall-selfeval.sh`** — the default recall forcing function
  (also what the auto-recall plugin invokes).
- **mnemostack over MCP** — when you want a spawned sub-agent or an external
  host to call into the same pipeline without shelling out to bash.

## Uninstall

Remove the `mnemostack` entry from `plugins.entries.mcp.config.servers` and
restart the gateway. Qdrant/Memgraph data is not touched.

## Troubleshooting

- **Server not listed after restart**: check
  `journalctl --user -u openclaw-gateway -n 200` for MCP init errors. A common
  cause is the binary not being on PATH for the gateway's systemd unit —
  replace `"mnemostack"` with the absolute path from `which mnemostack`.
- **Tool calls hang**: mnemostack's first call warms up the embedding
  provider; Ollama cold-start can take seconds. Switch to `gemini` for
  faster response if this matters.
- **Schema field name drift**: OpenClaw's plugin config schema is not yet
  stable. If `plugins.entries.mcp.config.servers` is rejected, run
  `config.schema.lookup mcp` to see the current shape.
