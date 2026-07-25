# Deployment Guide

## Overview

This guide covers running Mnemostack in production with durable storage.

The main production rule is simple: durable memory must be actually durable. Do not run Qdrant or Memgraph as throwaway containers without persistent volumes. Do not expose the HTTP API without authentication. Do not change embedding models on an existing collection without a migration plan.

Mnemostack can run as either:

- an **MCP stdio server** for local agent integrations such as Claude Desktop, Claude Code, Cursor, or other MCP-capable clients; or
- an **HTTP server** for applications and services that call `/recall`, `/answer`, `/feedback`, `/health` (plus `/healthz` / `/readyz` / `/status` probes), and `/metrics`.

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
    image: qdrant/qdrant:v1.18.3
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

Since 0.5.0 the default state location is `$XDG_STATE_HOME/mnemostack/server-state.json` (falling back to `~/.local/state/...`; a legacy `/tmp` state file is migrated once, atomically), and `FileStateStore` takes a cross-process `flock` around reads and writes — CLI, HTTP and MCP processes on the same host can safely share one state file on a local POSIX filesystem. The caveat that remains: do not share the state file over a network filesystem (NFS and friends — advisory locks are unreliable there) and do not rely on `flock` on non-POSIX platforms. For those cases use one writer, disable stateful writes, or implement a custom `StateStore` backed by Redis or your database.

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

`mnemostack serve` does not require authentication **by default** — a single-tenant deployment is meant to sit behind your existing auth layer. For multi-tenant deployments it has a built-in **service-key auth** mode (`--auth`, described below); enable it, or keep the server behind your own auth layer, before exposing it.

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

The `filters` parameter on `/recall` and `/answer` provides data isolation inside the retrievers (no point outside the filtered scope is returned), but it is **caller-supplied — not an authorization boundary**. A client that can reach the endpoint can pass any filter, or none.

For multi-tenant deployments, use the built-in **service-key auth** instead of trusting client filters: start the server with `mnemostack serve --auth`, issue per-tenant keys with `mnemostack keys add --tenant <id> --scopes read,write`, and clients present the key via `Authorization: Bearer <key>` or `X-API-Key`. The **tenant is resolved from the key** (a client can't assert another tenant) and enforced through the vector store's tenant filter plus a post-fusion `filter_by_tenant` backstop, so an authenticated caller only ever sees its own tenant's data. `/recall` and `/answer` require `read`, `/feedback` requires `write`; a missing/invalid key is `401`, an insufficient scope `403` (default-deny). The operator endpoints stay **unauthenticated even under `--auth`** — `/health`, `/healthz`, `/readyz`, `/status`, `/metrics` expose version, dependency reachability, and counters; protect them at your proxy (or bind to a trusted interface) if that's sensitive. (Auth is off by default — single-tenant deployments are unaffected.) You can still additionally inject a tenant filter at a reverse proxy for defense-in-depth. Upgrade note: on 0.5.0 and earlier, several retrievers ignored `filters=` entirely — see the Security section of the changelog.

### External key store (OpenBao / Vault-compatible)

By default service keys live in a local hashed-JSON file (`FileKeyStore`). If you
already run a secret store, point verification at it instead — key lifecycle
(rotation, audit, replication across `serve` nodes) then belongs to the store's
own tooling rather than a flat file you replicate by hand:

```bash
export MNEMOSTACK_KEYSTORE=openbao
export MNEMOSTACK_OPENBAO_URL=https://bao.internal:8200
export BAO_TOKEN=...                       # or MNEMOSTACK_OPENBAO_TOKEN, or AppRole:
# export MNEMOSTACK_OPENBAO_ROLE_ID=...    #   MNEMOSTACK_OPENBAO_SECRET_ID=...
mnemostack serve --auth                    # mcp-serve --auth reads the same env
```

Records live in the KV-v2 engine at `<mount>/<path_prefix>/<sha256-of-key>`
(defaults `secret` / `mnemostack/keys`, override with `MNEMOSTACK_OPENBAO_MOUNT`
/ `MNEMOSTACK_OPENBAO_PATH_PREFIX`) with the same shape the file store uses —
the plaintext never reaches the store, only its hash:

```bash
KEY=msk_$(openssl rand -hex 24)            # you mint the key; give it to the client
HASH=$(printf %s "$KEY" | sha256sum | cut -d' ' -f1)
bao kv put secret/mnemostack/keys/$HASH tenant=acme scopes=read,write
bao kv delete secret/mnemostack/keys/$HASH   # revoke
```

Semantics to know before switching:

- **Verify-only.** The adapter only ever *reads* (a read-capable token/AppRole is
  all it needs). `mnemostack keys` and the inspector's Keys panel manage the
  local file store only — under an external backend the panel reports keys as
  externally managed (`501`) and the CLI warns; quotas stay locally manageable.
- **Fail closed.** An unreachable/misconfigured store denies every key (and a
  selected-but-misconfigured backend fails the server at boot). Positive lookups
  are cached briefly (`MNEMOSTACK_OPENBAO_CACHE_TTL`, default 5s) so the MCP
  server's per-tool-call re-verification doesn't pay a network round trip each
  time; the TTL bounds how long **any** grant change (revocation, scope
  downgrade, tenant move) takes effect. Misses are never cached — a just-added
  key works immediately. Only the KV **v2** engine is supported; pointing at a
  v1 mount denies every key, so verify with one `bao kv get` before switching.
  Redirect responses from the store are refused (never followed), so the store
  token can't be re-sent to another host.
- No new dependency: the adapter uses the standard library's HTTP client with
  TLS verification on.

For a single box that doesn't want another daemon, encrypt the key file at rest
instead — see the sops + age recipe in [recipes.md](recipes.md).

### Per-tenant quotas and rate limits

Under `--auth` you can cap each tenant's resource use from a small per-tenant quota store (`mnemostack quota set --tenant <id> …`; JSON file at `MNEMOSTACK_QUOTAS_FILE`, default `~/.config/mnemostack/quotas.json`):

- **Storage** — `--max-points N` caps how many vector points a tenant may store. `mnemostack index-markdown --tenant <id>` resolves the cap from the quota store automatically and refuses a write that would exceed it. The library `Ingestor` enforces a cap only when the application passes it (`Ingestor(tenant=…, max_points=N)`) — a plain `Ingestor(tenant=…)` is **not** auto-capped from the store, so resolve `FileQuotaStore().get(tenant)` yourself in a custom ingest path. The quota store **fails open**: a corrupt/unreadable file logs loudly and imposes no limit rather than blocking ingest, so monitor the log if you rely on it.
- **Request rate** — `--max-rps R` (with an optional `--burst N`) throttles a tenant's requests on the authenticated HTTP surface; over the rate the server returns **`429`** with a `Retry-After` header. The limiter is **per-process** (a token bucket), so with N workers the effective ceiling is N×`max_rps`; for a hard global limit, enforce it at your reverse proxy instead. Tenants without a rate quota, and unauthenticated single-tenant deployments, are never throttled.

`quota set` is a partial update (only the fields you pass change; pass `none` to clear one), and `quota list` / `quota rm` round it out.

### Tenant offboarding and backup

When a tenant leaves (customer churn, an erasure request), remove it across every store in one command instead of manual surgery:

```bash
mnemostack tenant-rm --tenant acme --memgraph-uri bolt://localhost:7687 --dry-run
# tenant 'acme':                      <- per-store counts, nothing deleted yet
#   vector points:    5120
#   graph nodes:      312
#   ...
mnemostack tenant-rm --tenant acme --memgraph-uri bolt://localhost:7687 --yes
```

This deletes the tenant's vector points (a server-side filter delete — unscoped/legacy points and other tenants are never matched), its graph nodes **and** edges (`--memgraph-uri` defaults to the graph configured in your config/env, so a graph deployment is swept automatically; pass `--no-graph` to opt out on a vector-only deployment — and if the stack config can't be loaded, one of `--memgraph-uri` / `--no-graph` is required, since a config-file graph would otherwise be invisible), its service keys, its quota, and its learning-state partitions.

**Do not issue keys for a tenant while it is being offboarded**, and for a hard guarantee pause the servers (or wait past `MNEMOSTACK_OPENBAO_CACHE_TTL` for an external key store). `tenant-rm` revokes the tenant's keys before sweeping and re-scans after, but a single CLI pass cannot lock out key issuance cluster-wide: a key minted mid-sweep is caught by the re-scan and **reported as a partial removal** (so you re-run), not silently ignored — but the window is only fully closed by not issuing keys during the operation. Because keys go first, each data store is then swept from **live state** (never a stale preflight count), so a point, collection, or quota an *authenticated* writer created just before revocation is still caught — what's left after that is only out-of-band/unauthenticated writers, which is why the command reports a best-effort sweep rather than claiming the tenant is provably gone. The deletion is gated behind `--yes` and is **best-effort per store**: a failing store is reported, the rest proceed, and a nonzero exit lists what remains — re-run after fixing it.

Keys are handled **first** so a still-valid key can't re-write data into a store that was just cleaned. On a local key store the tenant's keys are revoked (and if that would remove the last usable admin key, or the revocation fails, the whole sweep aborts with nothing deleted — issue another admin key / fix the store and re-run). With an external key store (`MNEMOSTACK_KEYSTORE=openbao`) `tenant-rm` **cannot** revoke — revoke the tenant's key(s) there first (e.g. `bao kv delete ...`), then re-run with `--external-keys-revoked` to confirm and sweep the data stores.

Before removing (or for a plain backup), dump the tenant's data:

```bash
mnemostack tenant-export --tenant acme -o acme.jsonl
```

The export is the tenant's vector points (vectors + payloads, JSONL) — the portable source of truth. The graph is derived (rebuild by re-indexing), and keys/quotas/learning state are deliberately excluded (keys are secrets; copy the key store explicitly if that's really intended).

### Audit log (control-plane operations)

Set `MNEMOSTACK_AUDIT_FILE=/var/log/mnemostack/audit.jsonl` (one env var, all surfaces) to record every control-plane operation that touches a tenant's resources as one JSON line — who (`operator:cli`, or the admin key's public id at the `inspect --auth` console), what (`keys.issue`/`keys.revoke`, `quota.set`/`quota.remove`, `tenant.rm`/`tenant.export`/`tenant.migrate`, `auth.denied`), which tenant, and the outcome (`success` / `error` / `partial` / `aborted` / `denied`). Unset (the default) nothing is written. The admin console shows the trail in an **Audit** tab.

Know what it is and isn't:

- **Best-effort, never blocking.** A failed audit write logs a loud error and the operation proceeds — a broken log must not take key management or an emergency offboarding down. It is an *operational trail*, not a tamper-evident compliance ledger: anyone who can write the file can edit it. For tamper evidence, ship the JSONL to an external collector.
- **No key material, ever.** Events carry public key ids only — never a plaintext key or hash. Admin-console denials are logged for *presented-but-rejected* credentials (with the client IP); anonymous no-key probes are not (access-log noise).
- **Data plane is out of scope.** Recall/answer/feedback requests belong to your HTTP access logs; the audit trail is for operations that change a tenant's resources. Input-validation rejections and `--dry-run` passes (nothing attempted) are likewise not recorded.
- **Rotate externally.** The file grows without bound — point logrotate at it. Reading the trail (the Audit tab / `GET /api/audit`) scans the whole file each time (memory-bounded, but time grows with the file), so lapsed rotation shows up as a slower tab, another reason to rotate. Appends are `flock`-serialized with a bounded ~2s wait (CLI + console never interleave, and a hung lock-holder can't stall an audited operation), the file is created `0o640`, and readers skip-and-count a line truncated by a copytruncate rotation. Both reads and writes refuse symlinks in **every** path component and require a regular file owned by the running user (or root) that is no looser than `0640` — a planted link, FIFO, or pre-created world-accessible file fails loudly instead of diverting, hanging, or leaking the trail. So point `MNEMOSTACK_AUDIT_FILE` at a real path (a deliberately symlinked log directory is refused too), and if you pre-provision the file (e.g. via logrotate `create`), make it `0640` and owned by the service user.

### MCP

The MCP server is stdio-based:

```bash
mnemostack mcp-serve --provider gemini --collection production-memory
```

It should be launched by the local MCP client. It does not need a listening TCP port. Do not wrap MCP stdio in a network service unless you add a proper transport security and auth model.

For multi-tenant use, `mcp-serve --auth` runs the process as one authenticated tenant: launch it with a service key (`--api-key` / `MNEMOSTACK_API_KEY`) that resolves the tenant + scopes for the whole process (the natural fit for how an MCP client passes secrets in its server config). The key is **re-verified on every tool call**, so revoking it stops the data/graph tools immediately; they are scope-gated (`search`/`answer` need `read`, `invalidate`/`feedback` need `write`) and confined to the key's tenant — including the tenant-scoped graph tools. (`mnemostack_health` is public — a liveness check that still responds after the key is revoked, so it isn't a tenant boundary.)

### Web inspector / admin console

`mnemostack inspect` serves a small **read-only** operator console (default `127.0.0.1:8100`) for browsing tenants, per-tenant size, records, and dependency reachability. It never writes to the data, so it's safe to point at production — but it exposes memory contents, so bind it to a trusted interface (localhost or a proxy).

`mnemostack inspect --auth` turns it into a **tenant-administration console**: every `/api` call then requires an **admin**-scoped service key, and management panels appear for issuing/revoking service keys (a new key's plaintext is shown once) and setting/removing quotas. Auth is by header (no cookies, so no CSRF); revoking the last admin key is refused so the console can't lock itself out. Because it now exposes key metadata and mutation, keep it on a trusted interface / behind TLS. Without `--auth` the management endpoints return `403` and the console stays the read-only browser.

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
- **HTTP `/healthz`** (liveness) returns `200` whenever the process is up and checks no backend — a failure means "restart the pod", not "a dependency is down". Point a Kubernetes `livenessProbe` here.
- **HTTP `/readyz`** (readiness) checks the two hard dependencies recall needs — Qdrant (bounded ping; timeout `vector.health_timeout` / `--qdrant-health-timeout`, default 2s) and the embedding provider (reachability cached and refreshed in the background, so probes don't pay a live embedding call each time and a slow/wedged provider can't hang the probe) — and returns `503 not_ready` when either is down, so a load balancer stops routing traffic. The graph is optional/fail-soft and is deliberately **not** pinged here — a live graph check would let a slow/blackholed Memgraph add latency to the probe and trip its timeout; graph reachability is on `/health` and `/status` instead. Point a Kubernetes `readinessProbe` here.
- **HTTP `/status`** returns an operator snapshot: version, configured provider/LLM/collection/Qdrant URL, live dependency reachability, and headline counters (recall volume + total degradation events).
- **CLI `mnemostack health`** performs a deeper check: embedding provider reachable and correct dimension, Qdrant collection exists, point count non-zero, and Memgraph reachable when configured.
- **CLI `mnemostack doctor`** is the full read-only diagnostic: it validates config (provider names, `rerank_mode`), probes the embedding provider (a live reachability call — embeds a tiny string), Qdrant (reachability + collection + point count + a stored-vs-provider **vector-dimension mismatch** check that catches a collection built with a different embedding model), the LLM (config/construction check by default; `--check-llm` adds a live billable generation probe), and the graph (or `disabled`), printing a remediation hint per failure. Exit codes: `0` healthy / `1` a core recall dependency (embedding or Qdrant) down / `2` config invalid — usable directly in a CI or deploy gate. The graph is optional/fail-soft, so a configured-but-unreachable graph is a warning that does not fail the exit code. `--json` for machine consumption. It's read-only — never creates or modifies a collection — so it's safe against production.
- **MCP `mnemostack_health`** performs the same deep check as the CLI.

For production monitoring, use `/healthz` for liveness and `/readyz` for readiness probes, and combine with periodic CLI `mnemostack health` for full validation (collection exists, point count non-zero).

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

### Lexical search beyond the in-process BM25

The default lexical arm (in-process BM25 over `bm25_paths`) holds its corpus
in client memory — fine to roughly 100K documents, not for a million-chunk
collection. `recall.text_search` (env `MNEMOSTACK_TEXT_SEARCH`) selects the
arm; pick by collection size and whether you can re-index:

| Mode | Corpus lives | Scoring | Scale | Reindex needed |
|---|---|---|---|---|
| `bm25` (auto-default with `bm25_paths`) | client RAM, from files | true BM25 | <~100K | — |
| `qdrant_bm25` | client RAM, scrolled from payloads | true BM25 | <~100K | no |
| `lexical` | Qdrant full-text index | dense (gate filters, doesn't score) | any | **no** |
| `sparse` | Qdrant sparse vectors | server tf·idf-like (IDF modifier) | any | **yes** (space written at ingest) |

- `lexical` needs the full-text payload index on a real server: run
  `mnemostack text-index` once (idempotent, non-destructive — safe on a
  mounted pre-existing collection).
- `sparse` is the strongest lexical arm at scale: new collections indexed
  under this mode get the sparse space automatically. Switching an existing
  dense collection requires a **re-index into a collection created under
  sparse mode** — Qdrant servers cannot add a new sparse space to an existing
  collection (CI-verified against real servers v1.15.4 and v1.18.3; the
  attempt fails loudly, never silently). `mnemostack sparse-backfill` is for the OTHER gap: a collection
  that already has the space but holds points written dense-only (e.g.
  through the async store) — it writes the missing encodings without
  re-embedding anything.
- Server version minimums for the new arms: full-text `MatchText` filtering
  is long-standing, the sparse IDF modifier needs Qdrant server ≥ 1.10, and
  the coverage check (`has_vector`) needs ≥ 1.13 — the compose example pins
  `v1.18.3`, aligned with the `qdrant-client` version a fresh
  `pip install mnemostack` resolves to today; CI replays the real-server
  scenarios against both `v1.15.4` and `v1.18.3`.
- All modes respect the tenant boundary and validity view exactly like dense
  search, and follow the configured payload schema.
- **`sparse` writes are sync-only today**: `AsyncVectorStore` does not
  maintain the sparse space — a deployment ingesting through the async store
  under `text_search: sparse` would write dense-only points that never
  surface from sparse recall. Ingest through the sync `VectorStore`/CLI, or
  run `mnemostack sparse-backfill` after async writes.
- Editing a chunk's text via `set_payload` (`index --refresh-payloads`)
  leaves its sparse encoding stale (like the dense vector): re-upsert or run
  `mnemostack sparse-backfill` after bulk text edits.

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

Multiple MCP instances on the same host can share one `MNEMOSTACK_STATE_PATH`: since 0.5.0 `FileStateStore` locks the file across processes (`flock`) for every read and write. Do not share the file over a network filesystem — for multi-host learning state, use one writer or implement a shared `StateStore`.

### Multiple HTTP server instances

The HTTP server can be replicated behind a load balancer because Qdrant and Memgraph hold the durable corpus. The exception is the file state store. Options:

1. Disable stateful writes and use stateless recall only.
2. Use sticky routing to a single stateful instance.
3. Replace `FileStateStore` with a database-backed store.

Same-host replicas can share the state file (cross-process `flock` since 0.5.0). Do not put it on a shared network mount and assume concurrent writes are safe — advisory locks are unreliable over NFS.

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
image: qdrant/qdrant:v1.18.3
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

## Mounting a pre-existing collection (foreign payload schema)

Recall can run over a Qdrant collection **written by another system** — no
re-embedding, no payload rewrite. Two things usually differ in such a
collection: the payload field names, and how timestamps are stored. Both are
configuration:

```yaml
recall:
  text_key: content            # payload key holding the chunk text
  timestamp_key: updated_at    # payload key holding the timestamp
  timestamp_format: epoch      # iso (RFC3339 strings) | epoch (seconds) | epoch_ms
```

(or `MNEMOSTACK_TEXT_KEY` / `MNEMOSTACK_TIMESTAMP_KEY` /
`MNEMOSTACK_TIMESTAMP_FORMAT` — one deployment, one schema; the CLI, HTTP
server, and MCP all read the same setting).

Notes for this mode:

- **`timestamp_format` matters for temporal recall correctness**, not just
  display: window filters are sent to Qdrant in the field's own domain. With
  `iso` configured against a numeric field (or vice versa) the range condition
  matches nothing and temporal recall silently contributes zero — which is
  also the symptom to check first if date-scoped queries return nothing.
- Timestamp **values** are parsed tolerantly everywhere regardless of the
  format setting (ISO strings, numeric epochs — milliseconds detected by
  magnitude — datetimes); unparseable values degrade (no freshness decay, no
  date prefix) instead of erroring.
- Validity/invalidation keys (`invalidated_at`, `valid_from`, `valid_until`)
  need no mapping: absent fields mean "current", so a foreign collection is
  fully visible by default.
- This is a **read-side mount**. mnemostack's own ingest keeps writing the
  standard schema (`text`/`timestamp`) — don't point a writing deployment and
  a foreign-schema mount at the same collection.
- The embedding-consistency rule above still applies: query with the same
  model that produced the collection's vectors.

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
