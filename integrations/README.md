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
mnemostack health --provider gemini  # or ollama, huggingface
```

If `health` passes and you have indexed content (`mnemostack index ./notes/`),
the integration only wires the host up to the existing server process.

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
