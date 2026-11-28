# Instructions for coding agents

## Connecting an agent to a mnemostack deployment

Read [integrations/README.md](integrations/README.md) first. It says which
setup fits a deployment: the built-in `mnemostack mcp-serve` command, the
launcher for a custom embedding provider, or the HTTP transport for a remote
deployment. The tool reference is [docs/mcp.md](docs/mcp.md).

Before implementing a custom MCP adapter, test the built-in server and
identify a specific requirement it cannot meet. Document how to reproduce
that limitation.
