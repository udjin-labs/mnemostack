# Deployment Guide

## Overview

This guide covers running Mnemostack in production with durable storage.

The main production rule is simple: durable memory must be actually durable. Do not run Qdrant or Memgraph as throwaway containers without persistent volumes. Do not expose the HTTP API without authentication. Do not change embedding models on an existing collection without a migration plan.

Mnemostack can run as either:

- an **MCP stdio server** for local agent integrations such as Claude Desktop, Claude Code, Cursor, or other MCP-capable clients; or
- an **HTTP server** for applications and services that call `/recall`, `/answer`, `/feedback`, `/health`, and `/metrics`.

Both modes use the same backing stores:

- **Qdrant** stores vector memory and payloads.
- **Memgraph** optionally stores temporal graph facts.
- **BM25** optionally indexes a file-based corpus for exact-token retrieval.
- **FileStateStore** stores recall feedback and stateful pipeline data in a JSON file.

## Recommended architecture

```text
             +------------------+
             |      Agent       |
             | Claude / app /   |
             | service runtime  |
             +---------+--------+
                       |
          MCP stdio or HTTP API
                       |
             +---------v--------+
             |   Mnemostack     |
             | MCP or HTTP      |
             | recall service   |
             +----+--------+----+
                  |        |
        vector +  |        | graph facts
        payloads  |        | optional
                  |        |
        +---------v--+  +--v----------+
        |  Qdrant    |  |  Memgraph   |
        | persistent |  | persistent  |
        | volume     |  | volume      |
        +------------+  +-------------+
                  |
                  | optional exact-token corpus
                  v
        +----------------+
        | BM25 files     |
        | mounted read   |
        +----------------+
```

Recommended production shape:

1. Run Qdrant with a persistent volume or managed disk.
2. Run Memgraph only if you need graph-backed memory or MCP graph tools.
3. Run Mnemostack as a small stateless service, except for the configured state file.
4. Put a reverse proxy with authentication in front of the HTTP server.
5. Keep the embedding provider and embedding model pinned for each collection.

For MCP-only local use, the MCP server uses stdio and should not be exposed on the network. Qdrant and Memgraph still need durable storage if the memory matters.

## Production Docker Compose

The compose file below runs:

- Qdrant with a persistent named volume.
- Memgraph with a persistent named volume.
- Mnemostack HTTP server on an internal Docker network.
- A mounted state directory for `FileStateStore`.
- A mounted read-only corpus directory for BM25.

Save as `docker-compose.yml` and keep secrets in `.env` or your deployment secret manager.

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.12.6
    restart: unless-stopped
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      QDRANT__LOG_LEVEL: INFO
      # Prefer enabling Qdrant API keys when your Mnemostack deployment can
      # pass the key to Qdrant, or when a private proxy injects it.
      # Current Mnemostack CLI/server flags configure the Qdrant URL, not an
      # api_key parameter, so do not enable this line unless your deployment
      # has that wiring in place.
      # QDRANT__SERVICE__API_KEY: ${QDRANT_API_KEY:?set QDRANT_API_KEY}
    networks:
      - mnemostack
    # Healthcheck omitted from this example; the official qdrant/qdrant image
    # may not include bash or timeout. Use restart: unless-stopped instead.

  memgraph:
    # Pin this to a tested Memgraph version in production; avoid floating `latest`.
    image: memgraph/memgraph:latest
    restart: unless-stopped
    volumes:
      - memgraph_data:/var/lib/memgraph
    command:
      - "--bolt-server-name-for-init=Neo4j/5.11.0"
    networks:
      - mnemostack
    # Expose Bolt only to trusted networks if external tools need it.
    # ports:
    #   - "127.0.0.1:7687:7687"

  mnemostack:
    build:
      context: .
      dockerfile: Dockerfile
    restart: unless-stopped
    depends_on:
      - qdrant
      - memgraph
    environment:
      GEMINI_API_KEY: ${GEMINI_API_KEY:?set GEMINI_API_KEY}
      # CLI flags in `command` below take precedence over env vars.
      # Env vars listed here for visibility; remove duplicates if you prefer.
    volumes:
      - ./corpus:/data/corpus:ro
      - mnemostack_state:/var/lib/mnemostack
    networks:
      - mnemostack
      - edge
    expose:
      - "8000"
    command:
      - serve
      - --host
      - 0.0.0.0
      - --port
      - "8000"
      - --provider
      - gemini
      - --embedding-model
      - text-embedding-004
      - --llm
      - gemini
      - --llm-model
      - gemini-2.5-flash
      - --collection
      - production-memory
      - --qdrant
      - http://qdrant:6333
      - --memgraph-uri
      - bolt://memgraph:7687
      - --bm25-path
      - /data/corpus
      - --state-path
      - /var/lib/mnemostack/state.json

  # Example reverse proxy. Configure TLS and real auth before exposing this.
  caddy:
    image: caddy:2
    restart: unless-stopped
    depends_on:
      - mnemostack
    ports:
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
      - caddy_config:/config
    networks:
      - edge

volumes:
  qdrant_storage:
  memgraph_data:
  mnemostack_state:
  caddy_data:
  caddy_config:

networks:
  mnemostack:
    internal: true
  edge:
```

Example `.env`:

```bash
GEMINI_API_KEY=...
# Required only if you enable Qdrant API-key authentication.
QDRANT_API_KEY=change-me-to-a-long-random-value
```

Example `Caddyfile` with basic auth:

```caddyfile
memory.example.com {
  encode zstd gzip

  basicauth /* {
    ops-user $2a$14$REPLACE_WITH_CADDY_HASH_PASSWORD
  }

  reverse_proxy mnemostack:8000
}
```

Generate a Caddy password hash with:

```bash
docker run --rm caddy:2 caddy hash-password --plaintext 'your-long-password'
```

`depends_on` controls startup order, not readiness. Keep `restart: unless-stopped`; if Qdrant or Memgraph is still starting, Mnemostack health checks may fail briefly until dependencies become reachable.

If you do not need graph memory, remove the `memgraph` service and explicitly pass `--memgraph-uri ""` to disable graph. **Important:** the `mnemostack serve` CLI defaults `--memgraph-uri` to `bolt://localhost:7687` when omitted, so simply removing the flag does **not** disable graph — the server will attempt Bolt connections and log repeated timeouts. Pass an empty string to force it off. The `mnemostack mcp-serve` entrypoint reads from config where the default is `None` (disabled), so omitting is safe there. Verify with `/health` after startup.

## Persistence model

### Qdrant

Qdrant is the primary durable store for vector memory. It stores embeddings, IDs, text payloads, source metadata, and collection state.

In Docker, persist:

```yaml
volumes:
  - qdrant_storage:/qdrant/storage
```

If this volume is missing, removed, or mounted to the wrong path, the collection will disappear after container recreation. A common symptom is:

```text
collection exists=false, points=0
```

Use one collection per memory namespace, for example:

```bash
mnemostack index ./corpus \
  --provider gemini \
  --embedding-model text-embedding-004 \
  --collection production-memory \
  --qdrant http://localhost:6333
```

### Memgraph

Memgraph is optional. It stores temporal graph facts for graph recall and MCP graph tools.

Persist:

```yaml
volumes:
  - memgraph_data:/var/lib/memgraph
```

If Memgraph is down, Mnemostack can still recall from Qdrant and BM25. Graph-backed results and graph tools will fail or degrade until Memgraph returns.

### State store

The recall pipeline can use `FileStateStore(path)` for Q-learning weights, inhibition-of-return history, and other stateful recall data. For the HTTP server, pass:

```bash
--state-path /var/lib/mnemostack/state.json
```

Persist the parent directory:

```yaml
volumes:
  - mnemostack_state:/var/lib/mnemostack
```

The state file is small JSON. It is not the memory corpus. Losing it resets learned feedback and stateful ranking behavior, but does not delete indexed memories from Qdrant.

For multiple HTTP server processes writing feedback concurrently, do not share one JSON file over an unsafe filesystem. Use one writer, disable stateful writes, or implement a custom `StateStore` backed by Redis or your database.

### BM25 corpus

BM25 is file-based. Mnemostack builds BM25 docs from configured paths:

```bash
--bm25-path /data/corpus
```

or:

```bash
export MNEMOSTACK_BM25_PATHS=/data/corpus:/data/extra-notes
```

Mount the corpus read-only in production:

```yaml
volumes:
  - ./corpus:/data/corpus:ro
```

BM25 helps with exact strings: IDs, filenames, error messages, user handles, ticket numbers, and uncommon names. Keep the mounted corpus in sync with what you expect exact-token recall to find.

## Backups and restore

Backups are not optional. If agents depend on memory, memory is production data.

### Qdrant snapshots

The examples below include an `api-key` header. If Qdrant API-key authentication is disabled in your deployment, omit `-H "api-key: $QDRANT_API_KEY"`.

Create a collection snapshot:

```bash
curl -sS -X POST \
  -H "api-key: $QDRANT_API_KEY" \
  http://localhost:6333/collections/production-memory/snapshots | jq
```

List snapshots:

```bash
curl -sS \
  -H "api-key: $QDRANT_API_KEY" \
  http://localhost:6333/collections/production-memory/snapshots | jq
```

Download a snapshot:

```bash
curl -fL \
  -H "api-key: $QDRANT_API_KEY" \
  -o production-memory.snapshot \
  http://localhost:6333/collections/production-memory/snapshots/SNAPSHOT_NAME
```

Restore according to the Qdrant version you run. Test restore on a staging Qdrant before you trust the backup. A backup that has never been restored is only a hope.

Also back up the raw Qdrant volume if your infrastructure supports crash-consistent volume snapshots.

### Memgraph dump/load

For graph data, export a Cypher dump:

```bash
docker compose exec -T memgraph mgconsole --execute "DUMP DATABASE;" > memgraph.dump.cypherl
```

Restore into a clean Memgraph instance:

```bash
docker compose exec -T memgraph mgconsole < memgraph.dump.cypherl
```

Validate after restore:

```bash
docker compose exec -T memgraph mgconsole --execute "MATCH (n) RETURN count(n);"
docker compose exec -T memgraph mgconsole --execute "MATCH ()-[r]->() RETURN count(r);"
```

For large graphs, prefer storage-level snapshots or the backup mechanism recommended for your Memgraph edition and version. Always test restore.

### State store

The state store is a JSON file. Copy it while the service is stopped, or use a filesystem snapshot:

```bash
docker compose stop mnemostack
docker run --rm \
  -v mnemostack_mnemostack_state:/state:ro \
  -v "$PWD/backups:/backup" \
  busybox cp /state/state.json /backup/state-$(date +%F).json
docker compose start mnemostack
```

Restore by copying the JSON file back to the configured `--state-path`.

### Recommended schedule

A practical baseline:

- Qdrant snapshots: hourly for active systems, daily for low-write systems.
- Qdrant volume snapshots: daily.
- Memgraph dump or volume snapshot: daily, or more often if graph writes matter.
- State store copy: daily.
- Retention: 7 daily, 4 weekly, 3 monthly, adjusted for your compliance needs.
- Restore drill: at least once per month.

## Security

### Qdrant API key

Use a Qdrant API key when your deployment can pass the key from Mnemostack to Qdrant, or when a private proxy injects it:

```yaml
environment:
  QDRANT__SERVICE__API_KEY: ${QDRANT_API_KEY:?set QDRANT_API_KEY}
```

Clients must send:

```bash
-H "api-key: $QDRANT_API_KEY"
```

The current public Mnemostack CLI/server flags configure the Qdrant URL (`--qdrant`, `MNEMOSTACK_QDRANT_URL`, `MNEMOSTACK_QDRANT_HOST`) and do not expose a Qdrant `api_key` flag. If you have not added that wiring, keep Qdrant on an internal network instead of turning on Qdrant auth and breaking the service.

Do not expose ports `6333` or `6334` to the public internet.

### Memgraph auth

Keep Memgraph on a private network by default. Only expose Bolt (`7687`) to trusted hosts that need direct graph access.

If your Memgraph edition/version supports authentication, enable it and create a least-privilege user for Mnemostack. Store credentials in your secret manager, not in the compose file. If you run without Memgraph auth, compensate with network isolation: Docker internal networks, firewall rules, and no public port binding.

### HTTP server

`mnemostack serve` intentionally does not implement authentication. It is meant to sit behind your existing auth layer.

By default it binds to `127.0.0.1`. Binding to all interfaces prints a warning:

```bash
mnemostack serve --host 0.0.0.0 --port 8000
```

Only do this behind a reverse proxy with:

- TLS,
- authentication,
- request size limits,
- rate limits,
- access logs.

Treat `/recall`, `/answer`, and `/feedback` as sensitive. They can reveal indexed memory and influence future ranking state.

### MCP

The MCP server is stdio-based:

```bash
mnemostack mcp-serve --provider gemini --collection production-memory
```

It should be launched by the local MCP client. It does not need a listening TCP port. Do not wrap MCP stdio in a network service unless you add a proper transport security and auth model.

### `GEMINI_API_KEY` handling

Only set provider keys for providers you use. For Gemini:

```bash
export GEMINI_API_KEY=...
```

In production:

- inject it through a secret manager or `.env` file excluded from git;
- do not bake it into the Docker image;
- avoid logging the environment;
- rotate it if it appears in logs, shell history, screenshots, or issue reports.

## Monitoring

### Mnemostack health

HTTP:

```bash
curl -fsS http://localhost:8000/health | jq
```

CLI:

```bash
mnemostack health \
  --provider gemini \
  --embedding-model text-embedding-004 \
  --collection production-memory \
  --qdrant http://localhost:6333
```

MCP clients can call `mnemostack_health`.

Health checks differ by entrypoint:

- **HTTP `/health`** checks Qdrant endpoint reachability and Memgraph ping (when configured). It does **not** validate embedding provider, collection existence, or point count.
- **CLI `mnemostack health`** performs a deeper check: embedding provider reachable and correct dimension, Qdrant collection exists, point count non-zero, and Memgraph reachable when configured.
- **MCP `mnemostack_health`** performs the same deep check as the CLI.

For production monitoring, combine HTTP `/health` for liveness with periodic CLI `mnemostack health` for full validation.

### Prometheus metrics

The HTTP server exposes Prometheus text metrics:

```bash
curl -fsS http://localhost:8000/metrics
```

Useful alerts:

- elevated `/recall` or `/answer` latency;
- rising error rate;
- zero successful recalls over an expected traffic window;
- embedding provider failures;
- Qdrant unavailable;
- Memgraph unavailable when graph is required.

### Qdrant dashboard

If you expose Qdrant dashboard/API for operators, bind it to localhost or a VPN-only interface and require the API key. Watch:

- collection point count;
- disk usage;
- segment count and optimizer activity;
- snapshot success/failure;
- latency under recall load.

### Memgraph monitoring

Watch:

- process RSS and container memory;
- Bolt connection failures;
- node and relationship counts;
- query latency;
- disk usage under `/var/lib/memgraph`.

### Log levels

Set Qdrant logging with:

```yaml
environment:
  QDRANT__LOG_LEVEL: INFO
```

For Mnemostack, start with normal application logs. Increase verbosity only while debugging because recall payloads may contain sensitive memory text.

## Scaling

### Single-node vs distributed Qdrant

Start single-node unless you already know you need more. Single-node Qdrant with persistent disk is simpler to back up, restore, and reason about.

Move to distributed Qdrant when:

- collection size exceeds a single node's disk or memory budget;
- recall latency is constrained by vector search load;
- you need high availability across nodes.

Plan this as an infrastructure migration. Test snapshot and restore before moving production traffic.

### Read replicas

If your workload is read-heavy, use read replicas or a managed Qdrant topology that supports your availability target. Keep writes and snapshot behavior clear: agents are sensitive to stale memory when they expect recently indexed facts to appear immediately.

### Multiple MCP server instances

MCP servers are normally launched per client over stdio. They are mostly stateless and can point at the same Qdrant collection and Memgraph instance.

Be careful with `MNEMOSTACK_STATE_PATH` if multiple instances record feedback. A local JSON state file is not safe as a shared concurrent write target. For multi-process learning state, use one writer or implement a shared `StateStore`.

### Multiple HTTP server instances

The HTTP server can be replicated behind a load balancer because Qdrant and Memgraph hold the durable corpus. The exception is the file state store. Options:

1. Disable stateful writes and use stateless recall only.
2. Use sticky routing to a single stateful instance.
3. Replace `FileStateStore` with a database-backed store.

Do not put the same JSON state file on a shared network mount and assume concurrent writes are safe.

## Upgrades and rollback

### Mnemostack

Mnemostack's public API is intended to evolve additively in minor releases. Still, treat upgrades like production changes:

```bash
pip install --upgrade 'mnemostack[server]'
# or rebuild the Docker image from a pinned git tag / release
```

Before upgrade:

1. Back up Qdrant, Memgraph, and state store.
2. Record current package version and image digest.
3. Run smoke tests in staging against a copy of production data.
4. Check `/health`, `/metrics`, and representative recall queries.

### Qdrant

Pin Qdrant versions in compose instead of using `latest` in production:

```yaml
image: qdrant/qdrant:v1.12.6
```

Read Qdrant release notes before upgrading. Test snapshots and collection compatibility on staging.

### Re-indexing when changing embedding model

Changing the embedding model changes vector dimensions and vector geometry. Existing vectors are not compatible with queries embedded by a different model.

Safe migration pattern:

1. Pick a new collection name, for example `production-memory-v2`.
2. Index the same source corpus with the new model.
3. Run side-by-side recall quality checks.
4. Switch Mnemostack to the new collection.
5. Keep the old collection until rollback is no longer needed.

Example:

```bash
mnemostack index ./corpus \
  --provider gemini \
  --embedding-model text-embedding-004 \
  --collection production-memory-v2 \
  --qdrant http://localhost:6333
```

### Rollback

Rollback means restoring the whole memory contract:

- Mnemostack version or image digest;
- Qdrant version;
- collection name;
- embedding provider and model;
- Memgraph data if used;
- state file if feedback/ranking state matters.

If a deployment corrupts or deletes data, stop writes first. Restore Qdrant from a snapshot or volume backup, restore Memgraph if needed, copy back the state JSON, then restart Mnemostack.

## Embedding model consistency

This is critical: use the same embedding provider and model for indexing and querying a collection.

Example consistent setup:

```bash
mnemostack index ./corpus \
  --provider gemini \
  --embedding-model text-embedding-004 \
  --collection production-memory

mnemostack serve \
  --provider gemini \
  --embedding-model text-embedding-004 \
  --collection production-memory
```

If you index with one model and query with another, one of two things usually happens:

- the vector dimension differs and Qdrant rejects the query/upsert; or
- the dimension matches but the vector space differs, producing low-quality or misleading recall.

Do not rely on provider defaults for production unless you pin the package version and understand the default. Prefer explicit flags or environment variables:

```bash
export MNEMOSTACK_PROVIDER=gemini
export MNEMOSTACK_EMBEDDING_MODEL=text-embedding-004
export MNEMOSTACK_COLLECTION=production-memory
```

Migration to a new model should use a new collection, not in-place mutation.

## Smoke tests

Run these after first deploy, after every upgrade, and after restore.

### CLI health

```bash
mnemostack health \
  --provider gemini \
  --embedding-model text-embedding-004 \
  --collection production-memory \
  --qdrant http://localhost:6333
```

Expected: embedding OK, vector OK, collection exists.

### CLI search

Index a small known document first:

```bash
mkdir -p /tmp/mnemostack-smoke
printf 'smoke-memory: the deployment color is blue.\n' > /tmp/mnemostack-smoke/smoke.md

mnemostack index /tmp/mnemostack-smoke \
  --provider gemini \
  --embedding-model text-embedding-004 \
  --collection production-memory \
  --qdrant http://localhost:6333
```

Search:

```bash
mnemostack search "deployment color" \
  --provider gemini \
  --embedding-model text-embedding-004 \
  --collection production-memory \
  --qdrant http://localhost:6333 \
  --limit 5
```

Expected: the result includes `deployment color is blue`.

### HTTP health

```bash
curl -fsS http://localhost:8000/health | jq
```

### HTTP recall

```bash
curl -fsS http://localhost:8000/recall \
  -H 'content-type: application/json' \
  -d '{"query":"deployment color","limit":5}' | jq
```

Expected: non-empty `results` containing the smoke document.

### Metrics

```bash
curl -fsS http://localhost:8000/metrics | head
```

Expected: Prometheus text output.

## Common production failures

### Qdrant disk full

Symptoms:

- indexing fails;
- Qdrant logs write or segment errors;
- recall may still work for older data;
- snapshots fail.

Fix:

1. Stop or reduce writes.
2. Increase disk or move Qdrant storage to a larger volume.
3. Confirm snapshots work again.
4. Re-run failed indexing jobs.

Prevention: alert on disk usage before 80% and keep snapshot storage separate from the Qdrant data volume.

### Memgraph OOM

Symptoms:

- container restarts;
- graph tools fail;
- recall falls back to vector/BM25 only;
- Bolt connection errors in health checks.

Fix:

1. Increase memory limit or reduce graph size/query load.
2. Check expensive graph queries.
3. Restart Memgraph and run `mnemostack health`.
4. If data is corrupted or missing, restore from dump or volume snapshot.

Prevention: monitor RSS and relationship count growth.

### Embedding API rate limits

Symptoms:

- indexing slows or fails;
- `/recall` fails when query embedding cannot be generated;
- `/answer` may also fail if the LLM provider is rate-limited.

Fix:

1. Back off indexing jobs.
2. Reduce concurrency.
3. Retry with exponential backoff.
4. Consider local embeddings through Ollama or HuggingFace if external limits are unacceptable.

Prevention: separate bulk indexing windows from interactive recall traffic.

### State store permissions

Symptoms:

- `/feedback` fails;
- `--auto-record-ior` requests fail or log write errors;
- state file remains empty or missing.

Fix:

```bash
docker compose exec mnemostack sh -lc 'id && ls -ld /var/lib/mnemostack && touch /var/lib/mnemostack/.write-test'
```

Make the mounted directory writable by the container user. The project Dockerfile runs as user `mnemos`.

### Collection missing after Qdrant restart without volume

Symptoms:

- `/health` reports collection missing;
- point count is zero;
- previous recall results disappear after container recreation.

Cause: Qdrant was started without a persistent `/qdrant/storage` mount, or the wrong volume was mounted.

Fix:

1. Stop Qdrant.
2. Attach the correct volume or restore a snapshot.
3. Re-index if no backup exists.
4. Add a smoke test that restarts Qdrant and verifies the collection still exists.

## Production checklist

Before going live:

- [ ] Qdrant uses a persistent volume mounted at `/qdrant/storage`.
- [ ] Qdrant is not publicly exposed; Qdrant API-key auth is enabled if your deployment can pass the key.
- [ ] Memgraph is either disabled or stored on a persistent volume.
- [ ] Memgraph is network-isolated or authenticated according to your deployment model.
- [ ] Mnemostack HTTP is behind TLS and authentication before any external exposure.
- [ ] MCP is used over stdio only, not exposed as a network service.
- [ ] `GEMINI_API_KEY` and other provider keys are injected as secrets, not committed.
- [ ] Embedding provider and model are pinned for each collection.
- [ ] Indexing and serving use the same collection and embedding model.
- [ ] `FileStateStore` path is on persistent storage if feedback state matters.
- [ ] BM25 corpus paths are mounted read-only and match the intended exact-token corpus.
- [ ] Qdrant, Memgraph, and state store backups are scheduled.
- [ ] Restore has been tested in staging.
- [ ] `/health` returns OK.
- [ ] `/metrics` is scraped or checked.
- [ ] A known smoke query returns the expected memory.
- [ ] Disk, memory, latency, and error alerts are configured.
