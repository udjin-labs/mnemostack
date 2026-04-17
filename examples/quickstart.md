# mnemostack quickstart

## 1. Install

```bash
pip install mnemostack
```

## 2. Start Qdrant (+ optional Memgraph)

```bash
curl -O https://raw.githubusercontent.com/udjin-labs/mnemostack/main/examples/docker-compose.yml
docker compose up -d
```

## 3. Pick an embedding provider

### Option A: Ollama (local, no API key)

```bash
# Install ollama from ollama.com, then:
ollama pull nomic-embed-text
```

### Option B: Gemini (cloud, best quality)

```bash
export GEMINI_API_KEY=your-key-here
```

## 4. Create a config (optional but recommended)

```bash
mnemostack init
# edit ~/.config/mnemostack/config.yaml
mnemostack config       # show resolved config
```

## 5. Index your notes

```bash
mnemostack index ./my-notes/ --provider ollama --recreate
```

## 6. Search

```bash
mnemostack search "what did we decide about auth" --provider ollama
```

## 7. Answer mode (needs LLM)

```bash
mnemostack answer "what is the capital of France" --provider gemini
```

## 8. MCP server (Claude Desktop / Cursor)

Add to your MCP config:

```json
{
  "mcpServers": {
    "mnemostack": {
      "command": "mnemostack",
      "args": ["mcp-serve", "--provider", "gemini", "--collection", "my-memory"],
      "env": {
        "GEMINI_API_KEY": "your-key"
      }
    }
  }
}
```
