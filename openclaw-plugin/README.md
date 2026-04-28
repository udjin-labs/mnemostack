# mnemostack OpenClaw auto-recall plugin

Companion OpenClaw plugin for [mnemostack](../README.md). It connects OpenClaw's `before_prompt_build` hook to a mnemostack recall backend and injects a bounded `<active_memory>` block before the model sees the prompt.

mnemostack provides the durable recall backend: vector retrieval, BM25/exact-token rescue, graph retrieval, temporal retrieval, fusion, reranking, and answer synthesis. This plugin only decides when to ask for recall and how to add the answer to OpenClaw context.

## Quick start

1. Install and configure mnemostack.
2. Start the mnemostack HTTP daemon/server with its default answer endpoint:

   ```bash
   mnemostack serve --host 127.0.0.1 --port 18793
   ```

3. Install this plugin in OpenClaw from the `openclaw-plugin/` directory.
4. Enable the plugin.
5. Ask OpenClaw a recall-style question such as "what did we decide about the rollout?" or "что мы решили про релиз?".

With the daemon on the default port, no plugin configuration is required. The bundled `openclaw.plugin.json` and runtime defaults use:

```json
{
  "type": "http",
  "name": "mnemostack-daemon",
  "url": "http://127.0.0.1:18793/answer",
  "method": "POST",
  "requestMode": "default",
  "timeoutMs": 7000,
  "allowedHosts": ["127.0.0.1", "localhost"]
}
```

## What it does

- Runs from OpenClaw's `before_prompt_build` hook.
- Detects recall-style user messages with language-agnostic trigger lists.
- Ships with English and Russian defaults; custom triggers can extend or replace them.
- Calls the configured backend chain until it gets a confident answer.
- Injects only a bounded answer and source list into prompt context.
- Caches answers per backend/config/session and de-duplicates in-flight requests.

It does **not** store memories itself. mnemostack owns ingestion, storage, retrieval, ranking, and answer generation.

## HTTP backend contract

By default, the plugin POSTs JSON to `http://127.0.0.1:18793/answer`:

```json
{
  "query": "user text",
  "normalizedQuery": "user text normalized for cache",
  "limit": 5,
  "maxResults": 5,
  "minConfidence": 0.4,
  "trigger": { "patternId": "memory_keyword", "matchedText": "remember" },
  "session": { "agentId": "main", "provider": "telegram", "chatType": "direct" }
}
```

Expected response:

```json
{
  "status": "ok",
  "answer": "The thing you decided was...",
  "confidence": 0.86,
  "sources": [{ "title": "notes.md", "uri": "file:///notes.md", "score": 0.74 }]
}
```

`status` can be `ok`, `not_found`, `degraded`, or `error`.

## Script fallback

If you are not running the daemon, configure a script backend after or instead of HTTP. This is useful for `recall-selfeval.sh`-style local integrations:

```json
{
  "enabled": true,
  "backends": [
    {
      "type": "script",
      "name": "recall-selfeval",
      "command": "/path/to/recall-selfeval.sh",
      "args": ["--answer"],
      "queryMode": "stdin",
      "protocol": "text",
      "timeoutMs": 10000
    }
  ]
}
```

Script backends never use `shell: true`. Configure `command` and `args`; the user query is passed through stdin by default or as one argument when `queryMode: "arg"` is set.

## Config reference

| Key | Default | Description |
| --- | --- | --- |
| `enabled` | `true` | Register the hook when plugin is enabled. |
| `agents` | `[]` | Optional allow-list of OpenClaw agent ids. Empty means all. |
| `allowedChatTypes` | `["direct","group"]` | Chat types allowed for recall. |
| `timeoutMs` | `7000` | Per-hook timeout passed to backends. |
| `maxResults` | `5` | Hint for backend result count. |
| `minConfidence` | `0.4` | Minimum confidence required for injection. |
| `allowDegraded` | `false` | Allow `degraded` backend results to inject. |
| `logPath` | unset | Optional JSONL diagnostics path. |
| `recallMinChars` / `recallMaxChars` | `8` / `8000` | Bounds input sent to backends. |
| `backends` | mnemostack HTTP daemon | Ordered backend chain. |
| `triggers.customPath` | unset | Optional `triggers.json`; reload with `SIGUSR2`. |
| `triggers.reloadSignal` | `SIGUSR2` | Reload signal; set to `false` to avoid signal handlers. |
| `cache.enabled` | `true` | In-memory cache. |
| `cache.okTtlMs` | `1800000` | TTL for successful answers. |
| `cache.notFoundTtlMs` | `600000` | TTL for `not_found` negative cache. |
| `cache.degradedTtlMs` | `1800000` | TTL for `degraded` candidates. |
| `cache.errorTtlMs` | `0` | Error cache TTL. Timeout/infra failures are not cached. |
| `cache.maxEntries` | `500` | Max cache entries; expired/LRU sweep. |
| `injection.maxLength` | `2000` | Maximum injected body length. |
| `injection.tag` | `active_memory` | XML-ish envelope tag. |

## Custom triggers

Trigger defaults are intentionally language-agnostic lists rather than hardcoded grammar. English and Russian phrases are included. Add a JSON file and point `triggers.customPath` at it to extend deployment-specific names, keywords, or false-positive allow-lists.

## Security notes

HTTP backends call the configured URL from the OpenClaw Gateway process. Treat this as SSRF-capable configuration: do not expose plugin configuration to untrusted users, and prefer loopback/private endpoints you control. The default backend is loopback-only and uses `allowedHosts` to reject unexpected hosts and redirects.

## Test

```bash
npm test
```
