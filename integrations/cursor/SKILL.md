# Cursor

Wire mnemostack into Cursor as an MCP server. Cursor reads MCP servers from
its own settings, not from Claude Desktop's config.

## Install

1. Make sure the binary and a working backend are ready:

   ```bash
   pip install 'mnemostack[mcp]'
   mnemostack health --provider gemini
   ```

2. Open Cursor → Settings → MCP (or edit `~/.cursor/mcp.json` directly).

3. Add the server entry:

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

4. Reload Cursor (`Cmd/Ctrl+Shift+P` → *Reload Window*). The server should
   appear in the MCP panel with a green dot.

## Verify

Open the Cursor chat and ask:

> What MCP tools do you have from mnemostack?

Expected: `search`, `answer` (and `graph_query` if Memgraph URI was set).

Quick smoke test:

> Search mnemostack for "VPN failover", tier 2.

The model should invoke the tool and return short (~40 char) snippets with
source ids.

## Uninstall

Remove the `mnemostack` block from `~/.cursor/mcp.json` (or toggle it off in
the Cursor MCP settings UI) and reload the window.

## Troubleshooting

- **Server stays red**: Cursor doesn't always show stderr — run the same
  command manually in a terminal to see what's wrong:
  `mnemostack mcp-serve --provider gemini --collection my-memory`.
- **PATH differs in GUI**: Cursor on macOS inherits a different PATH than
  Terminal. Use an absolute path for `command` (e.g. output of
  `which mnemostack`).
- **Model can't find the tool**: some Cursor versions gate MCP behind per-chat
  tool toggles. Check the tools icon in the chat input.
