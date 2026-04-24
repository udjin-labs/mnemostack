# mnemostack integrations

MCP (Model Context Protocol) integrations for popular agent hosts. Each
sub-directory has a self-contained `SKILL.md` with install, verify, and
uninstall instructions.

| Host | Path | Transport |
|---|---|---|
| Claude Desktop / Claude Code | [`claude-code/`](claude-code/SKILL.md) | stdio |
| Cursor | [`cursor/`](cursor/SKILL.md) | stdio |
| OpenClaw | [`openclaw/SKILL.md`](openclaw/SKILL.md) | stdio |

All three use the same `mnemostack mcp-serve` binary. They differ only in
where the host's MCP config lives and how it discovers servers.

## Prerequisites (all hosts)

```bash
pip install 'mnemostack[mcp]'
mnemostack health --provider gemini  # or ollama, openai
```

If `health` passes and you have indexed content (`mnemostack index ./notes/`),
the integration only wires the host up to the existing server process.

## What the MCP server exposes

Tools (names may evolve; check `mnemostack mcp-serve --help` for the current list):

- `search` — hybrid recall over indexed content
- `answer` — LLM-synthesized answer with citations
- `graph_query` *(if `--memgraph-uri` is set)* — direct Cypher over the graph

All tools accept the same `--tier {1,2,3}` semantics that the CLI exposes, so
agents can bound output size per call.
